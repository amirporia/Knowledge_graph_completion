import json
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
                          get_constant_schedule_with_warmup)

from .models import build_model, ModelOutput
from .negative_sampler import HaSaNegativeSampler
from ..evaluation.metric import accuracy
from ..setting.config import uses_hasa
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

# Keys that must NOT go through nn.DataParallel's scatter / the model's forward: (B,B) masks, python objects
# and sampler bookkeeping that is only consumed by the loss after the gather.
_NON_MODEL_KEYS = ('triplet_mask', 'self_negative_mask', 'batch_data',
                   'prob', 'f_prob', 'hard_idx', 'false_idx', 'head_idx', 'tail_idx')


class Trainer:
    """Handles model training, evaluation, and checkpointing for every --method."""

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        self.best_metric = None
        self.start_epoch = 0
        self.sampler = None
        # HaSa's raw-dot-product losses can overflow; skip such steps instead of corrupting the weights
        self.skip_nonfinite = uses_hasa(args.method)

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
            logger.info("=> creating model")
        self.model = build_model(self.args)
        if self.args.rank == 0:
            logger.info(self.model)

    def _setup_device(self):
        """Place the model on this process's device, wrapping in DDP if distributed."""
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
            )
        elif torch.cuda.device_count() > 1:
            logger.info(f'Using nn.DataParallel across {torch.cuda.device_count()} GPUs; '
                        f'global batch size stays {self.args.batch_size}.')
            self.model = nn.DataParallel(self.model)

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
        self.train_dataset = Dataset(path=self.args.train_path)

        self.train_loader, self.train_sampler = self._create_data_loader(
            self.train_dataset, batch_size=self.args.batch_size,
            shuffle=True, drop_last=True, distributed=self.args.distributed
        )

        # Validation only ever runs on rank 0 (see _run_eval).
        self.valid_dataset = None
        self.valid_loader = None
        if self.args.valid_path and self.args.rank == 0:
            self.valid_dataset = Dataset(path=self.args.valid_path)
            self.valid_loader, _ = self._create_data_loader(
                self.valid_dataset, batch_size=self.args.batch_size * 2,
                shuffle=True, distributed=False
            )

    def _create_data_loader(self, dataset, batch_size, shuffle, drop_last=False, distributed=False):
        sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            if distributed else None
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collate,
            num_workers=self.args.workers,
            pin_memory=torch.cuda.is_available(),
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
        scheduler_type = self.args.lr_scheduler
        if scheduler_type == 'constant':
            return get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.args.warmup)

        schedulers = {
            'linear': get_linear_schedule_with_warmup,
            'cosine': get_cosine_schedule_with_warmup
        }
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
        """Restore model/optimizer/scheduler/scaler state from a checkpoint if --resume was passed
        (epoch granularity)."""
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

    def _init_negative_sampler(self):
        """Create the HaSa embedding bank (needs the final -- possibly resumed -- weights)."""
        if getattr(get_model_obj(self.model), 'needs_negative_sampler', False):
            self.sampler = HaSaNegativeSampler(self.args, self.device)
            self.sampler.init_bank(self.model)

    def train_loop(self):
        """Main training loop over epochs."""
        self._init_negative_sampler()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    def train_epoch(self, epoch):
        if self.args.distributed:
            self.train_sampler.set_epoch(epoch)

        meters = self._init_training_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [meters['losses'], meters['inv_t'], meters['top1'], meters['top3']],
            prefix=f"Epoch: [{epoch}]"
        )

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()
            batch_dict = self._move_batch_to_device(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            if self.sampler is not None:
                batch_dict.update(self.sampler.sample(self.model, batch_dict))

            outputs = self._forward_pass(batch_dict)
            if self.sampler is not None:
                self.sampler.update_bank(batch_dict, outputs)
            loss, outputs = self._compute_loss(outputs, batch_dict, batch_size)

            self._update_meters(meters, loss, outputs, batch_size)
            self._backward_pass(loss)
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
            'top1': AverageMeter('Acc@1', ':6.2f'),
            'top3': AverageMeter('Acc@3', ':6.2f'),
            'inv_t': AverageMeter('InvT', ':6.2f')
        }

    def _move_batch_to_device(self, batch_dict):
        if torch.cuda.is_available():
            return move_to_cuda(batch_dict)
        return batch_dict

    def _forward_pass(self, batch_dict):
        model_kwargs = {k: v for k, v in batch_dict.items() if k not in _NON_MODEL_KEYS}
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(**model_kwargs)
        return self.model(**model_kwargs)

    def _compute_loss(self, outputs, batch_dict, batch_size):
        """Method-specific training loss (SimKGC: bidirectional InfoNCE, HaSa: hardness-aware InfoNCE)."""
        loss, outputs = get_model_obj(self.model).compute_loss(outputs, batch_dict)
        assert outputs.logits.size(0) == batch_size
        return loss, outputs

    def _update_meters(self, meters, loss, outputs, batch_size):
        acc1, acc3 = accuracy(outputs.logits, outputs.labels, topk=(1, 3))

        meters['top1'].update(acc1.item(), batch_size)
        meters['top3'].update(acc3.item(), batch_size)
        meters['inv_t'].update(outputs.inv_t.item(), 1)
        if torch.isfinite(loss):
            meters['losses'].update(loss.item(), batch_size)

    def _backward_pass(self, loss):
        self.optimizer.zero_grad()

        if self.args.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )
            self.scaler.step(self.optimizer)  # skipped automatically when grads are inf/nan
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )
            if self.skip_nonfinite and not torch.isfinite(grad_norm):
                logger.warning('Non-finite loss/gradient, skipping this step')
                self.optimizer.zero_grad()
                return
            self.optimizer.step()

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        """Run validation and handle checkpointing (rank 0 only under DDP)."""
        if self.args.rank == 0:
            metric_dict = self.eval_epoch(epoch)
            is_best = self._check_best_metric(metric_dict)
            if is_best:
                self.best_metric = metric_dict

            self._save_checkpoint(epoch, step, is_best)

        if self.args.distributed:
            dist.barrier()

    def _check_best_metric(self, metric_dict):
        if not self.valid_loader:
            return False
        if self.best_metric is None:
            return True
        return metric_dict.get('Acc@1', 0) > self.best_metric.get('Acc@1', 0)

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
        # checkpoint_*.mdl / model_best.mdl are only read by eval/predict scripts
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
        """Evaluate the model on the validation set (in-batch ranking, same for all methods)."""
        if not self.valid_loader:
            return {}

        meters = {
            'losses': AverageMeter('Loss', ':.4'),
            'top1': AverageMeter('Acc@1', ':6.2f'),
            'top3': AverageMeter('Acc@3', ':6.2f')
        }

        model = get_model_obj(self.model)
        model.eval()

        for batch_dict in self.valid_loader:
            batch_dict = self._move_batch_to_device(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = model(**batch_dict)
            outputs = model.compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)

            loss = self.criterion(outputs.logits, outputs.labels)
            acc1, acc3 = accuracy(outputs.logits, outputs.labels, topk=(1, 3))

            meters['losses'].update(loss.item(), batch_size)
            meters['top1'].update(acc1.item(), batch_size)
            meters['top3'].update(acc3.item(), batch_size)

        metric_dict = self._format_metrics(meters)
        logger.info(f'Epoch {epoch}, valid metric: {json.dumps(metric_dict)}')

        return metric_dict

    @staticmethod
    def _format_metrics(meters):
        return {
            'Acc@1': round(meters['top1'].avg, 3),
            'Acc@3': round(meters['top3'].avg, 3),
            'loss': round(meters['losses'].avg, 3)
        }
