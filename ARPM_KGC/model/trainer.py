import json
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from .models import build_model
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


class Trainer:
    """Handles ARPM-KGC model training, evaluation, and checkpointing.
        L = L_query + eta_p * L_proto + eta_s * L_struct + eta_div * L_div
    """

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
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
            logger.info("=> creating ARPM-KGC model")
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
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    def _init_optimizer_and_criterion(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        if self.args.rank == 0:
            report_num_trainable_parameters(get_model_obj(self.model))

    def _init_data_loaders(self):
        self.train_dataset = Dataset(path=self.args.train_path, test_set=False)

        self.train_loader, self.train_sampler = self._create_data_loader(
            self.train_dataset, shuffle=True, drop_last=True, distributed=self.args.distributed
        )

        self.valid_dataset = None
        self.valid_loader = None
        if self.args.valid_path and self.args.rank == 0:
            self.valid_dataset = Dataset(path=self.args.valid_path, test_set=False)
            self.valid_loader, _ = self._create_data_loader(
                self.valid_dataset, shuffle=True, distributed=False
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

        checkpoint = load_checkpoint(self.args.resume_path, map_location=self.device)

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

    def train_loop(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    def train_epoch(self, epoch):
        if self.args.distributed:
            self.train_sampler.set_epoch(epoch)

        meters = self._init_training_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [meters['losses'], meters['query_losses'], meters['proto_losses'],
             meters['struct_losses'], meters['div_losses'], meters['combined_losses'],
             meters['inv_t'], meters['top1'], meters['top3']],
            prefix=f"Epoch: [{epoch}]"
        )

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()
            batch_dict = self._move_batch_to_device(batch_dict)

            outputs = self._forward_pass(batch_dict)
            loss_components = self._compute_losses(outputs, batch_dict)

            self._update_meters(meters, loss_components)
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
            'query_losses': AverageMeter('L_query', ':.4'),
            'proto_losses': AverageMeter('L_proto', ':.4'),
            'struct_losses': AverageMeter('L_struct', ':.4'),
            'div_losses': AverageMeter('L_div', ':.4'),
            'combined_losses': AverageMeter('L_combined', ':.4'),
            'top1': AverageMeter('Acc@1', ':6.2f'),
            'top3': AverageMeter('Acc@3', ':6.2f'),
            'inv_t': AverageMeter('InvT', ':6.2f')
        }

    def _move_batch_to_device(self, batch_dict):
        if torch.cuda.is_available():
            return move_to_cuda(batch_dict)
        return batch_dict

    def _forward_pass(self, batch_dict):
        model_kwargs = {k: v for k, v in batch_dict.items()
                        if k not in ('triplet_mask', 'self_negative_mask', 'batch_data', 'test_forward')}
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(**model_kwargs)
        return self.model(**model_kwargs)

    def _compute_losses(self, outputs, batch_dict) -> Dict:
        """In-batch negatives (batch tail vectors act as the candidate entity set).
         The same false-negative `triplet_mask` is applied
        to all three logit matrices (S_q, S_p, S_struct) since an unmasked false
        negative would corrupt every one of them, not just S_q.

        A shared inverse-temperature `exp(log_inv_t)` (and, for S_q only, the
        additive margin + self-negative term) is applied ONLY to the copies of
        S_q/S_p/S_struct/S(t|h,r) used to form the CE logits below -- never to the
        raw values returned for ranking or analysis (see model/models.py scoring
        docstring).

        L_combined (weighted by --eta-combined) is a bidirectional CE loss on the
        scaled combined score S(t|h,r) = S_q + lambda_p*S_p + lambda_s*S_struct.
        It exists purely so MemoryGate (G_lambda, producing lambda_p/lambda_s)
        sits on a path back to total_loss -- L_query/L_proto/L_struct/L_div never
        touch lambda_p/lambda_s, so without this term the gate is never trained
        and, under DDP, triggers "unused parameter" errors.
        """
        model_obj = get_model_obj(self.model)

        q = outputs['q']
        tail_vector = outputs['tail_vector']
        head_vector = outputs['head_vector']
        prototypes = outputs['prototypes']
        m_struct = outputs['m_struct']
        lambda_p = outputs['lambda_p']
        lambda_s = outputs['lambda_s']
        div_loss = outputs['div_loss']
        slot_gate = outputs.get('slot_gate')

        batch_size = q.size(0)
        labels = torch.arange(batch_size, device=q.device)
        inv_t = model_obj.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask')

        S_q = model_obj.score_query(q, tail_vector)
        S_p = model_obj.score_prototypes(prototypes, tail_vector, slot_gate=slot_gate)
        S_s = model_obj.score_struct(m_struct, tail_vector)

        # ---- L_query: additive margin + self-negative + inv-temperature scaling ----
        q_logits = S_q - torch.diag_embed(
            torch.full((batch_size,), self.args.additive_margin, device=q.device)
        )
        q_logits = q_logits * inv_t
        if triplet_mask is not None:
            q_logits = q_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)

        if self.args.use_self_negative:
            self_neg_logits = torch.sum(q * head_vector, dim=1) * inv_t
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits = self_neg_logits.masked_fill(~self_negative_mask, model_obj.NEGATIVE_INF)
            q_logits = torch.cat([q_logits, self_neg_logits.unsqueeze(1)], dim=-1)

        L_query = self._bidirectional_ce(q_logits, labels, batch_size)

        # ---- L_proto, L_struct: inv-temperature scaling only (no margin/self-neg) ----
        p_logits = S_p * inv_t
        s_logits = S_s * inv_t
        if triplet_mask is not None:
            p_logits = p_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)
            s_logits = s_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)

        L_proto = self._bidirectional_ce(p_logits, labels, batch_size)
        L_struct = self._bidirectional_ce(s_logits, labels, batch_size)

        # ---- L_combined: the only term touching lambda_p/lambda_s (MemoryGate) ----
        combined_score = model_obj.combined_score(S_q, S_p, S_s, lambda_p, lambda_s)
        combined_logits = combined_score * inv_t
        if triplet_mask is not None:
            combined_logits = combined_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)
        L_combined = self._bidirectional_ce(combined_logits, labels, batch_size)

        total_loss = L_query + self.args.eta_proto * L_proto + \
                     self.args.eta_struct * L_struct + self.args.eta_div * div_loss + \
                     self.args.eta_combined * L_combined

        return {
            'total_loss': total_loss,
            'query_loss': L_query,
            'proto_loss': L_proto,
            'struct_loss': L_struct,
            'div_loss': div_loss,
            'combined_loss': L_combined,
            'combined_score': combined_score,
            'labels': labels,
        }

    @staticmethod
    def _bidirectional_ce(logits: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        """CE(logits, labels) in both directions (t->prediction and prediction->t).
        `logits` may have an extra trailing self-negative column (only ever present for S_q);
        the backward direction always uses only the square (batch_size, batch_size)
        block."""
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        loss = loss + criterion(logits[:, :batch_size].t(), labels)
        return loss

    def _update_meters(self, meters, loss_components):
        batch_size = self.args.batch_size

        acc1, acc3 = accuracy(
            loss_components['combined_score'],
            loss_components['labels'],
            topk=(1, 3)
        )

        meters['losses'].update(loss_components['total_loss'].item(), batch_size)
        meters['query_losses'].update(loss_components['query_loss'].item(), batch_size)
        meters['proto_losses'].update(loss_components['proto_loss'].item(), batch_size)
        meters['struct_losses'].update(loss_components['struct_loss'].item(), batch_size)
        meters['div_losses'].update(loss_components['div_loss'].item(), batch_size)
        meters['combined_losses'].update(loss_components['combined_loss'].item(), batch_size)
        meters['top1'].update(acc1.item(), batch_size)
        meters['top3'].update(acc3.item(), batch_size)

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

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        """Rank-0-only validation + checkpointing after each epoch (step=0) or
        every `--eval-every-n-step` steps mid-epoch (step=i+1).
        """
        if self.args.rank == 0:
            # in_batch_metric_dict = self.eval_epoch(epoch) logger.info(f'Epoch {epoch} in-batch diagnostic (NOT used
            # for checkpoint selection): {json.dumps(in_batch_metric_dict)}')

            is_epoch_boundary = (step == 0)
            due_for_full_eval = (epoch + 1) % max(self.args.full_eval_every_n_epochs, 1) == 0
            run_full_eval = self.valid_loader is not None and is_epoch_boundary and due_for_full_eval

            is_best = False
            if run_full_eval:
                full_metric_dict = self._compute_full_validation_metrics()
                logger.info(
                    f'Epoch {epoch} full filtered-ranking validation (used for '
                    f'checkpoint selection via --checkpoint-metric='
                    f'{self.args.checkpoint_metric}): {json.dumps(full_metric_dict)}'
                )
                is_best = self._check_best_metric(full_metric_dict)
                if is_best:
                    self.best_metric = full_metric_dict

            self._save_checkpoint(epoch, step, is_best)

        if self.args.distributed:
            dist.barrier()

    def _check_best_metric(self, metric_dict):
        if not metric_dict:
            return False
        if self.best_metric is None:
            return True
        key = self.args.checkpoint_metric
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
        """Cheap in-batch diagnostic: Acc@1/Acc@3/loss on S(t|h,r) among only
        the batch's own tails."""
        if not self.valid_loader:
            return {}

        meters = {
            'losses': AverageMeter('Loss', ':.4'),
            'top1': AverageMeter('Acc@1', ':6.2f'),
            'top3': AverageMeter('Acc@3', ':6.2f')
        }

        for batch_dict in self.valid_loader:
            self.model.eval()
            batch_dict = self._move_batch_to_device(batch_dict)

            outputs = self._forward_pass(batch_dict)
            loss_components = self._compute_losses(outputs, batch_dict)

            batch_size = self.args.batch_size
            meters['losses'].update(loss_components['total_loss'].item(), batch_size)

            acc1, acc3 = accuracy(
                loss_components['combined_score'], loss_components['labels'], topk=(1, 3)
            )
            meters['top1'].update(acc1.item(), batch_size)
            meters['top3'].update(acc3.item(), batch_size)

        return self._format_metrics(meters)

    def _format_metrics(self, meters):
        return {
            'Acc@1': round(meters['top1'].avg, 3),
            'Acc@3': round(meters['top3'].avg, 3),
            'loss': round(meters['losses'].avg, 3)
        }

    @torch.no_grad()
    def _compute_full_validation_metrics(self) -> Dict[str, float]:

        from ..evaluation.predict import ARPMPredictor
        from ..evaluation.evaluate import evaluate_predictor
        from ..utils.dict_hub import get_entity_dict

        model_obj = get_model_obj(self.model)
        was_training = model_obj.training
        was_is_test = self.args.is_test

        model_obj.eval()
        self.args.is_test = True
        try:
            predictor = ARPMPredictor.from_model(
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
