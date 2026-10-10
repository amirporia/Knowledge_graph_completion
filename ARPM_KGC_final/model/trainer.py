import json
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from .models import build_model, ModelOutput
from ..evaluation.metric import accuracy
from ..setting.logger_config import logger
from ..utils.dict_hub import init_tokenizer
from ..utils.doc import Dataset, collate
from ..utils.utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    load_checkpoint,
    delete_old_checkpoints,
    report_num_trainable_parameters,
    move_to_cuda,
    get_model_obj
)

_NON_MODEL_KEYS = ('triplet_mask', 'self_negative_mask', 'related_triplet_mask', 'batch_data')


class Trainer:
    """Handles model training, evaluation, and checkpointing.

    Loss (baseline):  L = alpha * L_hrta + L_hr                                   (Eq. 6)
    Loss (ARPM):      L = alpha * L_hrta + L_hr
                          + eta_p L_proto + eta_s L_struct + eta_c L_combined
                          + eta_div L_div + eta_pdiv L_pdiv
    The ARPM terms see detached query / entity vectors, so ONLY the memory modules receive their
    gradient: the encoders are trained exactly as in the baseline.
    """

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        self.use_memory = bool(args.use_memory)
        self.best_metric = None
        self.start_epoch = 0

        self._initialize_tokenizer()
        self._build_model()
        self._setup_device()
        self._init_optimizer_and_criterion()
        self._init_data_loaders()
        self._init_scheduler()
        self._init_amp()
        self._maybe_resume()

    def _initialize_tokenizer(self):
        init_tokenizer(self.args)

    def _build_model(self):
        if self.args.rank == 0:
            logger.info(f"=> creating model (ARPM memory extension: {'ON' if self.use_memory else 'OFF = baseline'})")
        self.model = build_model(self.args)
        if self.args.rank == 0:
            logger.info(self.model)

    def _setup_device(self):
        if self.args.distributed:
            self.device = torch.device(f'cuda:{self.args.local_rank}')
            torch.cuda.set_device(self.device)
        elif torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.args.gpu}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)

        if self.args.distributed:
            # --fixed-lambda-* ablations bypass MemoryGate in the forward pass, so DDP must tolerate
            # unused parameters when the memory extension is on. Baseline keeps the original setting.
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                broadcast_buffers=False, find_unused_parameters=self.use_memory,
            )
        elif torch.cuda.device_count() > 1:
            logger.info(f'Using nn.DataParallel across {torch.cuda.device_count()} GPUs; '
                        f'global batch size stays {self.args.batch_size}.')
            self.model = nn.DataParallel(self.model)

    def _init_optimizer_and_criterion(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        model_obj = get_model_obj(self.model)

        if self.use_memory:
            # memory modules are trained from scratch -> own (larger) LR, no weight decay
            main_params, memory_params = [], []
            for name, p in model_obj.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith(('hr_bert.', 'tail_bert.')) or name == 'log_inv_t':
                    main_params.append(p)
                else:
                    memory_params.append(p)
            groups = [{'params': main_params}]
            if memory_params:
                groups.append({'params': memory_params, 'lr': self.args.memory_lr, 'weight_decay': 0.0})
            self.optimizer = AdamW(groups, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            self.optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        if self.args.rank == 0:
            report_num_trainable_parameters(model_obj)

    def _init_data_loaders(self):
        self.train_dataset = Dataset(path=self.args.train_path, test_set=False)

        self.train_loader, self.train_sampler = self._create_data_loader(
            self.train_dataset, shuffle=True, drop_last=True, distributed=self.args.distributed
        )

        # In-batch validation (baseline checkpoint selection) only runs on rank 0.
        # With --checkpoint-metric != acc the full filtered-ranking pass is used instead,
        # which builds its own loaders, so no in-batch validation set is needed.
        self.valid_dataset = None
        self.valid_loader = None
        self.has_valid = bool(self.args.valid_path) and self.args.rank == 0
        if self.has_valid and self.args.checkpoint_metric == 'acc':
            self.valid_dataset = Dataset(path=self.args.valid_path, test_set=False)
            self.valid_loader, _ = self._create_data_loader(
                self.valid_dataset, shuffle=False, distributed=False
            )

    def _create_data_loader(self, dataset, shuffle, drop_last=False, distributed=False):
        sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            if distributed else None
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collate,
            num_workers=self.args.workers,
            pin_memory=False,
            drop_last=drop_last
        )
        return loader, sampler

    def _init_scheduler(self):
        world_size = self.args.world_size if self.args.distributed else 1
        steps_per_epoch = len(self.train_dataset) // world_size // max(self.args.batch_size, 1)
        num_training_steps = self.args.epochs * steps_per_epoch

        self.args.warmup = min(self.args.warmup, num_training_steps // 10)
        if self.args.rank == 0:
            logger.info(
                f'Total training steps: {num_training_steps}, '
                f'warmup steps: {self.args.warmup}'
            )
        self.scheduler = self._create_lr_scheduler(num_training_steps)

    def _create_lr_scheduler(self, num_training_steps):
        schedulers = {
            'linear': get_linear_schedule_with_warmup,
            'cosine': get_cosine_schedule_with_warmup
        }

        scheduler_type = self.args.lr_scheduler
        if scheduler_type not in schedulers:
            raise ValueError(f'Unknown lr scheduler: {scheduler_type}')

        return schedulers[scheduler_type](
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup,
            num_training_steps=num_training_steps
        )

    def _init_amp(self):
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

    def _maybe_resume(self):
        if not self.args.resume:
            return

        checkpoint = load_checkpoint(self.args.resume_path, map_location='cpu')

        get_model_obj(self.model).load_state_dict(checkpoint['state_dict'])

        if checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.args.use_amp and checkpoint.get('scaler') is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.best_metric = checkpoint.get('best_metric')
        self.start_epoch = checkpoint.get('epoch', -1) + 1

        if self.args.rank == 0:
            logger.info(
                f'Resumed from {self.args.resume_path} '
                f'(checkpoint epoch {checkpoint.get("epoch")}, resuming at epoch {self.start_epoch})'
            )

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ training
    def train_loop(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    def train_epoch(self, epoch):
        if self.args.distributed:
            self.train_sampler.set_epoch(epoch)

        meters = self._init_training_meters()
        shown = [meters['losses'], meters['inv_t'], meters['top1'], meters['top3'],
                 meters['hr_losses'], meters['related_losses']]
        if self.use_memory:
            shown += [meters['proto_losses'], meters['struct_losses'], meters['combined_losses'],
                      meters['lam_p'], meters['lam_s']]
        progress = ProgressMeter(len(self.train_loader), shown, prefix=f"Epoch: [{epoch}]")

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()
            batch_dict = self._move_batch_to_device(batch_dict)

            outputs = self._forward_pass(batch_dict)
            loss_components = self._compute_losses(outputs, batch_dict)

            self._update_meters(meters, loss_components, outputs)
            self._backward_pass(loss_components['total_loss'])
            self.scheduler.step()

            if i % self.args.print_freq == 0 and self.args.rank == 0:
                progress.display(i)
            if self.args.eval_every_n_step > 0 and (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)

        if self.args.rank == 0:
            logger.info(f'Learning rate: {self.scheduler.get_last_lr()[0]}')

    def _init_training_meters(self):
        return {
            'losses': AverageMeter('Loss', ':.4'),
            'related_losses': AverageMeter('RelatedLoss', ':.4'),
            'hr_losses': AverageMeter('HRLoss', ':.4'),
            'proto_losses': AverageMeter('L_proto', ':.4'),
            'struct_losses': AverageMeter('L_struct', ':.4'),
            'combined_losses': AverageMeter('L_comb', ':.4'),
            'lam_p': AverageMeter('lam_p', ':.3f'),
            'lam_s': AverageMeter('lam_s', ':.3f'),
            'top1': AverageMeter('Acc@1', ':6.2f'),
            'top3': AverageMeter('Acc@3', ':6.2f'),
            'inv_t': AverageMeter('InvT', ':6.2f')
        }

    def _move_batch_to_device(self, batch_dict):
        if torch.cuda.is_available():
            return move_to_cuda(batch_dict)
        return batch_dict

    def _forward_pass(self, batch_dict):
        """Execute forward pass with optional AMP."""
        model_kwargs = {k: v for k, v in batch_dict.items() if k not in _NON_MODEL_KEYS}
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(**model_kwargs)
        return self.model(**model_kwargs)

    def _compute_losses(self, outputs, batch_dict):
        """Baseline Eq.(6): L_cls = alpha * L_hrta + L_hr  (+ ARPM memory losses if enabled)."""
        model_obj = get_model_obj(self.model)
        logit_dict = ModelOutput(**model_obj.compute_logits(output_dict=outputs, batch_dict=batch_dict))

        related_loss = self._compute_bidirectional_loss(
            logit_dict.related_logits, logit_dict.related_labels
        )
        hr_loss = self._compute_bidirectional_loss(
            logit_dict.hr_logits, logit_dict.hr_labels
        )

        total_loss = self.args.alpha * related_loss + hr_loss

        result = {
            'total_loss': total_loss,
            'related_loss': related_loss,
            'hr_loss': hr_loss,
            'hr_logits': logit_dict.hr_logits,
            'hr_labels': logit_dict.hr_labels
        }

        if self.use_memory:
            mem = self._compute_memory_losses(outputs, batch_dict)
            result['total_loss'] = (
                total_loss
                + self.args.eta_proto * mem['proto_loss']
                + self.args.eta_struct * mem['struct_loss']
                + self.args.eta_combined * mem['combined_loss']
                + self.args.eta_div * mem['div_loss']
                + self.args.eta_pdiv * mem['pdiv_loss']
            )
            result.update(mem)
        return result

    def _compute_memory_losses(self, outputs, batch_dict) -> Dict:
        """ARPM auxiliary losses. q / tails / base score are detached -> memory modules only.

        L_proto / L_struct : in-batch InfoNCE of S_p / S_struct
        L_combined         : InfoNCE of S_base.detach() + lambda_p S_p + lambda_s S_struct,
                             the only loss that trains the gate (it learns the residual over baseline)
        """
        model_obj = get_model_obj(self.model)

        tail_d = outputs['tail_vector'].detach().float()
        q_d = outputs['hr_vector'].detach().float()
        q_hrta_d = outputs['related_hr_vector'].detach().float()
        prototypes = outputs['prototypes']
        m_struct = outputs['m_struct']
        lambda_p, lambda_s = outputs['lambda_p'], outputs['lambda_s']

        batch_size = q_d.size(0)
        labels = torch.arange(batch_size, device=q_d.device)
        inv_t = model_obj.log_inv_t.exp().detach()
        triplet_mask = batch_dict.get('triplet_mask')

        S_p = model_obj.score_prototypes(prototypes, tail_d)
        S_s = model_obj.score_struct(m_struct, tail_d)
        S_base = q_d.mm(tail_d.t()) + q_hrta_d.mm(tail_d.t())   # baseline Eq.(9) score

        def _masked(score):
            logits = score * inv_t
            if triplet_mask is not None:
                logits = logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)
            return logits

        L_proto = self._compute_bidirectional_loss(_masked(S_p), labels)
        L_struct = self._compute_bidirectional_loss(_masked(S_s), labels)

        combined_score = model_obj.combined_score(S_base, S_p, S_s, lambda_p, lambda_s)
        L_combined = self._compute_bidirectional_loss(_masked(combined_score), labels)

        return {
            'proto_loss': L_proto,
            'struct_loss': L_struct,
            'combined_loss': L_combined,
            'div_loss': outputs['div_loss'].mean(),
            'pdiv_loss': outputs['proto_div'].mean(),
        }

    def _compute_bidirectional_loss(self, logits, labels):
        """InfoNCE in both directions (query->tail and tail->query), as in SimKGC."""
        assert logits.size(0) == self.args.batch_size

        loss = self.criterion(logits, labels)
        loss += self.criterion(logits[:, :self.args.batch_size].t(), labels)

        return loss

    def _update_meters(self, meters, loss_components, outputs):
        batch_size = self.args.batch_size

        acc1, acc3 = accuracy(
            loss_components['hr_logits'],
            loss_components['hr_labels'],
            topk=(1, 3)
        )

        meters['losses'].update(loss_components['total_loss'].item(), batch_size)
        meters['related_losses'].update(loss_components['related_loss'].item(), batch_size)
        meters['hr_losses'].update(loss_components['hr_loss'].item(), batch_size)
        meters['top1'].update(acc1.item(), batch_size)
        meters['top3'].update(acc3.item(), batch_size)
        meters['inv_t'].update(get_model_obj(self.model).log_inv_t.exp().item(), batch_size)

        if self.use_memory:
            meters['proto_losses'].update(loss_components['proto_loss'].item(), batch_size)
            meters['struct_losses'].update(loss_components['struct_loss'].item(), batch_size)
            meters['combined_losses'].update(loss_components['combined_loss'].item(), batch_size)
            meters['lam_p'].update(outputs['lambda_p'].mean().item(), batch_size)
            meters['lam_s'].update(outputs['lambda_s'].mean().item(), batch_size)

    def _backward_pass(self, loss):
        self.optimizer.zero_grad()

        if self.args.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )
            self.optimizer.step()

    # ------------------------------------------------------------------ evaluation / checkpointing
    def _metric_key(self) -> str:
        return 'Acc@1' if self.args.checkpoint_metric == 'acc' else self.args.checkpoint_metric

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        """Validation + checkpointing on rank 0; other ranks wait at the barrier.

        checkpoint-metric == 'acc' : baseline in-batch Acc@1 (cheap, every call)
        otherwise                  : full filtered-ranking validation at epoch boundaries
        """
        if self.args.rank == 0:
            is_best = False
            if self.args.skip_valid_eval or not self.has_valid:
                # No validation: keep the latest weights as "best" so evaluate.py still finds model_best.mdl
                is_best = True
            elif self.args.checkpoint_metric == 'acc':
                metric_dict = self.eval_epoch(epoch)
                is_best = self._check_best_metric(metric_dict)
                if is_best:
                    self.best_metric = metric_dict
            else:
                due = (epoch + 1) % max(self.args.full_eval_every_n_epochs, 1) == 0
                if step == 0 and due:
                    metric_dict = self._compute_full_validation_metrics()
                    logger.info(
                        f'Epoch {epoch} full filtered-ranking validation '
                        f'(selection metric: {self.args.checkpoint_metric}): {json.dumps(metric_dict)}'
                    )
                    is_best = self._check_best_metric(metric_dict)
                    if is_best:
                        self.best_metric = metric_dict

            self._save_checkpoint(epoch, step, is_best)

        if self.args.distributed:
            dist.barrier()

    def _check_best_metric(self, metric_dict):
        if not metric_dict:
            return False
        if self.best_metric is None:
            return True
        key = self._metric_key()
        return metric_dict.get(key, 0) > self.best_metric.get(key, 0)

    def _save_checkpoint(self, epoch, step, is_best):
        if step == 0:
            filename = f'{self.args.model_dir}/checkpoint_epoch{epoch}.mdl'
        else:
            filename = f'{self.args.model_dir}/checkpoint_{epoch}_{step}.mdl'

        model_state = get_model_obj(self.model).state_dict()

        full_state = {
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.args.use_amp else None,
            'best_metric': self.best_metric,
        }
        eval_state = {
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': model_state,
        }

        save_checkpoint(full_state, is_best=is_best, filename=filename, eval_state=eval_state)

        delete_old_checkpoints(
            path_pattern=f'{self.args.model_dir}/checkpoint_*.mdl',
            keep=self.args.max_to_keep
        )

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        """Baseline in-batch validation (rank 0 only)."""
        if not self.valid_loader:
            return {}

        meters = {
            'losses': AverageMeter('Loss', ':.4'),
            'top1': AverageMeter('Acc@1', ':6.2f'),
            'top3': AverageMeter('Acc@3', ':6.2f')
        }

        # Unwrapped module: under DDP only rank 0 runs this.
        model = get_model_obj(self.model)
        model.eval()

        for batch_dict in self.valid_loader:
            batch_dict = self._move_batch_to_device(batch_dict)

            outputs = model(**batch_dict)
            outputs = model.compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)

            loss = self.criterion(outputs.hr_logits, outputs.hr_labels)
            acc1, acc3 = accuracy(outputs.hr_logits, outputs.hr_labels, topk=(1, 3))

            batch_size = outputs.hr_logits.size(0)
            meters['losses'].update(loss.item(), batch_size)
            meters['top1'].update(acc1.item(), batch_size)
            meters['top3'].update(acc3.item(), batch_size)

        metric_dict = self._format_metrics(meters)
        logger.info(f'Epoch {epoch}, valid metric: {json.dumps(metric_dict)}')

        return metric_dict

    def _format_metrics(self, meters):
        return {
            'Acc@1': round(meters['top1'].avg, 3),
            'Acc@3': round(meters['top3'].avg, 3),
            'loss': round(meters['losses'].avg, 3)
        }

    @torch.no_grad()
    def _compute_full_validation_metrics(self) -> Dict[str, float]:
        """Same filtered-ranking code path as evaluation/evaluate.py (identical for baseline & ARPM)."""
        from ..evaluation.predict import BertPredictor
        from ..evaluation.evaluate import evaluate_predictor
        from ..utils.dict_hub import get_entity_dict

        model_obj = get_model_obj(self.model)
        was_training = model_obj.training
        was_is_test = self.args.is_test

        model_obj.eval()
        self.args.is_test = True
        try:
            predictor = BertPredictor.from_model(
                model_obj, device=self.device, use_cuda=torch.cuda.is_available(),
                batch_size=self.args.full_eval_batch_size,
            )
            entity_dict = get_entity_dict()
            entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)
            result = evaluate_predictor(
                predictor, entity_tensor=entity_tensor,
                batch_size=self.args.full_eval_batch_size, save_details=False,
            )
        finally:
            self.args.is_test = was_is_test
            model_obj.train(was_training)

        return result['average']
