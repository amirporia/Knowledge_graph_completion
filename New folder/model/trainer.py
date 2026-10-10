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

_NON_MODEL_KEYS = ('triplet_mask', 'self_negative_mask', 'batch_data', 'test_forward')


class Trainer:
    """Trains ARPM-KGC v2.

        L = L_query + alpha * L_hrta                      (encoders; identical to the RAA-KGC baseline)
          + eta_p L_proto + eta_s L_struct + eta_c L_combined + eta_div L_div + eta_pdiv L_pdiv
                                                           (memory modules only: inputs are detached)
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
            logger.info("=> creating ARPM-KGC v2 model")
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
                broadcast_buffers=False, find_unused_parameters=True,
            )
        elif torch.cuda.device_count() > 1:
            # Every tensor returned by ARPMModel.forward is a per-example (B, ...) tensor, so
            # DataParallel gathers them along dim 0 and all losses still see the full batch.
            logger.info(f'Using nn.DataParallel across {torch.cuda.device_count()} GPUs; '
                        f'global batch size stays {self.args.batch_size}.')
            self.model = nn.DataParallel(self.model)

    def _init_optimizer_and_criterion(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        model_obj = get_model_obj(self.model)

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

        if self.args.rank == 0:
            report_num_trainable_parameters(model_obj)

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
            dataset, batch_size=self.args.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler, collate_fn=collate, num_workers=self.args.workers,
            pin_memory=False, drop_last=drop_last
        )
        return loader, sampler

    def _init_scheduler(self):
        world_size = self.args.world_size if self.args.distributed else 1
        steps_per_epoch = len(self.train_dataset) // world_size // max(self.args.batch_size, 1)
        num_training_steps = self.args.epochs * steps_per_epoch

        self.args.warmup = min(self.args.warmup, num_training_steps // 10)
        if self.args.rank == 0:
            logger.info(f'Total training steps: {num_training_steps}, warmup steps: {self.args.warmup}')
        self.scheduler = self._create_lr_scheduler(num_training_steps)

    def _create_lr_scheduler(self, num_training_steps):
        schedulers = {'linear': get_linear_schedule_with_warmup, 'cosine': get_cosine_schedule_with_warmup}
        if self.args.lr_scheduler not in schedulers:
            raise ValueError(f'Unknown lr scheduler: {self.args.lr_scheduler}')
        return schedulers[self.args.lr_scheduler](
            optimizer=self.optimizer, num_warmup_steps=self.args.warmup,
            num_training_steps=num_training_steps
        )

    def _init_amp(self):
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

    def _maybe_resume(self):
        if not self.args.resume:
            return

        # Load on CPU first: avoids a transient double allocation of weights + AdamW state on the GPU.
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
            logger.info(f'Resumed from {self.args.resume_path} '
                        f'(checkpoint epoch {checkpoint.get("epoch")}, resuming at epoch {self.start_epoch})')

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            [meters['losses'], meters['query_losses'], meters['hrta_losses'], meters['proto_losses'],
             meters['struct_losses'], meters['combined_losses'], meters['lam_p'], meters['lam_s'],
             meters['inv_t'], meters['top1'], meters['top3']],
            prefix=f"Epoch: [{epoch}]"
        )

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()
            batch_dict = self._move_batch_to_device(batch_dict)

            outputs, loss_components = self._forward_and_compute_losses(batch_dict)

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
            'query_losses': AverageMeter('L_query', ':.4'),
            'hrta_losses': AverageMeter('L_hrta', ':.4'),
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

    def _forward_and_compute_losses(self, batch_dict):
        """Forward AND loss computation inside the same autocast region (see v1 notes)."""
        model_kwargs = {k: v for k, v in batch_dict.items() if k not in _NON_MODEL_KEYS}
        with torch.amp.autocast('cuda', enabled=self.args.use_amp):
            outputs = self.model(**model_kwargs)
            loss_components = self._compute_losses(outputs, batch_dict)
        return outputs, loss_components

    def _compute_losses(self, outputs, batch_dict) -> Dict:
        model_obj = get_model_obj(self.model)

        q = outputs['q']
        q_hrta = outputs['q_hrta']
        tail_vector = outputs['tail_vector']
        head_vector = outputs['head_vector']
        prototypes = outputs['prototypes']
        m_struct = outputs['m_struct']
        lambda_p = outputs['lambda_p']
        lambda_s = outputs['lambda_s']

        div_loss = outputs['div_loss'].mean()
        pdiv_loss = outputs['proto_div'].mean()

        batch_size = q.size(0)
        labels = torch.arange(batch_size, device=q.device)
        inv_t = model_obj.log_inv_t.exp()
        inv_t_aux = inv_t.detach()
        triplet_mask = batch_dict.get('triplet_mask')
        margin = torch.diag_embed(torch.full((batch_size,), self.args.additive_margin, device=q.device))

        # ---- L_query (encoders): additive margin + self-negative, as in the baseline ----
        S_q = model_obj.score_query(q, tail_vector)
        q_logits = (S_q - margin) * inv_t
        if triplet_mask is not None:
            q_logits = q_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)
        if self.args.use_self_negative:
            self_neg_logits = torch.sum(q * head_vector, dim=1) * inv_t
            self_neg_logits = self_neg_logits.masked_fill(~batch_dict['self_negative_mask'],
                                                          model_obj.NEGATIVE_INF)
            q_logits = torch.cat([q_logits, self_neg_logits.unsqueeze(1)], dim=-1)
        L_query = self._bidirectional_ce(q_logits, labels, batch_size)

        # ---- L_hrta (encoders): anchor-enhanced query, as in the baseline (Eq. 7) ----
        if model_obj.use_raa:
            S_h = model_obj.score_query(q_hrta, tail_vector)
            h_logits = (S_h - margin) * inv_t
            if triplet_mask is not None:
                h_logits = h_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)
            L_hrta = self._bidirectional_ce(h_logits, labels, batch_size)
            S_base = S_q.detach() + S_h.detach()
        else:
            L_hrta = q.new_zeros(())
            S_base = S_q.detach()

        # ---- memory losses: only memory modules receive gradient (q and tails are detached) ----
        tail_d = tail_vector.detach()
        S_p = model_obj.score_prototypes(prototypes, tail_d)
        S_s = model_obj.score_struct(m_struct, tail_d)

        p_logits = S_p * inv_t_aux
        s_logits = S_s * inv_t_aux
        if triplet_mask is not None:
            p_logits = p_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)
            s_logits = s_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)
        L_proto = self._bidirectional_ce(p_logits, labels, batch_size)
        L_struct = self._bidirectional_ce(s_logits, labels, batch_size)

        # ---- L_combined: trains the gate (base score is frozen, so the gate learns the residual) ----
        combined_score = model_obj.combined_score(S_base, S_p, S_s, lambda_p, lambda_s)
        combined_logits = combined_score * inv_t_aux
        if triplet_mask is not None:
            combined_logits = combined_logits.masked_fill(~triplet_mask, model_obj.NEGATIVE_INF)
        L_combined = self._bidirectional_ce(combined_logits, labels, batch_size)

        total_loss = (L_query + self.args.alpha * L_hrta
                      + self.args.eta_proto * L_proto + self.args.eta_struct * L_struct
                      + self.args.eta_combined * L_combined
                      + self.args.eta_div * div_loss + self.args.eta_pdiv * pdiv_loss)

        return {
            'total_loss': total_loss,
            'query_loss': L_query,
            'hrta_loss': L_hrta,
            'proto_loss': L_proto,
            'struct_loss': L_struct,
            'div_loss': div_loss,
            'combined_loss': L_combined,
            'combined_score': combined_score,
            'labels': labels,
        }

    @staticmethod
    def _bidirectional_ce(logits: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        loss = loss + criterion(logits[:, :batch_size].t(), labels)
        return loss

    def _update_meters(self, meters, loss_components, outputs):
        batch_size = self.args.batch_size
        acc1, acc3 = accuracy(loss_components['combined_score'], loss_components['labels'], topk=(1, 3))

        meters['losses'].update(loss_components['total_loss'].item(), batch_size)
        meters['query_losses'].update(loss_components['query_loss'].item(), batch_size)
        meters['hrta_losses'].update(loss_components['hrta_loss'].item(), batch_size)
        meters['proto_losses'].update(loss_components['proto_loss'].item(), batch_size)
        meters['struct_losses'].update(loss_components['struct_loss'].item(), batch_size)
        meters['combined_losses'].update(loss_components['combined_loss'].item(), batch_size)
        meters['lam_p'].update(outputs['lambda_p'].mean().item(), batch_size)
        meters['lam_s'].update(outputs['lambda_s'].mean().item(), batch_size)
        meters['top1'].update(acc1.item(), batch_size)
        meters['top3'].update(acc3.item(), batch_size)
        meters['inv_t'].update(get_model_obj(self.model).log_inv_t.exp().item(), batch_size)

    def _backward_pass(self, loss):
        self.optimizer.zero_grad()

        if self.args.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        if self.args.rank == 0:
            is_epoch_boundary = (step == 0)
            due_for_full_eval = (epoch + 1) % max(self.args.full_eval_every_n_epochs, 1) == 0
            run_full_eval = self.valid_loader is not None and is_epoch_boundary and due_for_full_eval

            is_best = False
            if run_full_eval:
                full_metric_dict = self._compute_full_validation_metrics()
                logger.info(
                    f'Epoch {epoch} full filtered-ranking validation (used for checkpoint selection via '
                    f'--checkpoint-metric={self.args.checkpoint_metric}): {json.dumps(full_metric_dict)}'
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
        eval_state = {'epoch': epoch, 'args': self.args.__dict__, 'state_dict': model_state}

        save_checkpoint(full_state, is_best=is_best, filename=filename, eval_state=eval_state)
        delete_old_checkpoints(path_pattern=f'{self.args.model_dir}/checkpoint_*.mdl',
                               keep=self.args.max_to_keep)

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
