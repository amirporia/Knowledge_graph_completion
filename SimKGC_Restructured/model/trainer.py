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


class Trainer:
    """Handles model training, evaluation, and checkpointing."""

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
        """Initialize the tokenizer based on args."""
        init_tokenizer(self.args)

    def _build_model(self):
        """Build and log the model architecture (rank 0 only, to avoid duplicate log spam)."""
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

    def _init_optimizer_and_criterion(self):
        """Initialize loss function, optimizer, and log trainable parameters."""
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        if self.args.rank == 0:
            report_num_trainable_parameters(get_model_obj(self.model))

    def _init_data_loaders(self):
        """Initialize training and validation data loaders."""
        self.train_dataset = Dataset(path=self.args.train_path)

        self.train_loader, self.train_sampler = self._create_data_loader(
            self.train_dataset, batch_size=self.args.batch_size,
            shuffle=True, drop_last=True, distributed=self.args.distributed
        )

        # Validation only ever runs on rank 0 (see _run_eval), so other ranks skip
        # loading/tokenizing it entirely instead of duplicating that work per GPU.
        self.valid_dataset = None
        self.valid_loader = None
        if self.args.valid_path and self.args.rank == 0:
            self.valid_dataset = Dataset(path=self.args.valid_path)
            self.valid_loader, _ = self._create_data_loader(
                self.valid_dataset, batch_size=self.args.batch_size * 2,
                shuffle=True, distributed=False
            )

    def _create_data_loader(self, dataset, batch_size, shuffle, drop_last=False, distributed=False):
        """Create a DataLoader, using a DistributedSampler when sharding across ranks."""
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
        """Initialize learning rate scheduler."""
        # Under DDP, each rank only ever sees 1/world_size of the dataset (via the
        # DistributedSampler), so the scheduler -- which steps once per local optimizer
        # step -- needs the per-rank step count, not the full dataset's.
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
        """Create learning rate scheduler based on config."""
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
        """Initialize the AMP gradient scaler (created before resume so its state can be restored)."""
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

    def _maybe_resume(self):
        """Restore model/optimizer/scheduler/scaler state from a checkpoint if --resume was passed.

        Resumption is at epoch granularity: training continues from the epoch after the
        one recorded in the checkpoint. If the checkpoint was written mid-epoch (from an
        --eval-every-n-step save), the remainder of that particular epoch is not replayed.
        """
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
        """Main training loop over epochs."""
        for epoch in range(self.start_epoch, self.args.epochs):
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        if self.args.distributed:
            # Reshuffles each rank's shard differently per epoch.
            self.train_sampler.set_epoch(epoch)

        meters = self._init_training_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [meters['losses'], meters['inv_t'], meters['top1'], meters['top3']],
            prefix=f"Epoch: [{epoch}]"
        )

        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()
            batch_dict = self._move_batch_to_device(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self._forward_pass(batch_dict)
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
        """Initialize AverageMeter objects for tracking metrics."""
        return {
            'losses': AverageMeter('Loss', ':.4'),
            'top1': AverageMeter('Acc@1', ':6.2f'),
            'top3': AverageMeter('Acc@3', ':6.2f'),
            'inv_t': AverageMeter('InvT', ':6.2f')
        }

    def _move_batch_to_device(self, batch_dict):
        """Move batch to GPU if available."""
        if torch.cuda.is_available():
            return move_to_cuda(batch_dict)
        return batch_dict

    def _forward_pass(self, batch_dict):
        """Execute forward pass with optional AMP."""
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(**batch_dict)
        return self.model(**batch_dict)

    def _compute_loss(self, outputs, batch_dict, batch_size):
        """Compute the bidirectional InfoNCE loss."""
        outputs = get_model_obj(self.model).compute_logits(
            output_dict=outputs, batch_dict=batch_dict
        )
        outputs = ModelOutput(**outputs)
        assert outputs.logits.size(0) == batch_size

        # head + relation -> tail
        loss = self.criterion(outputs.logits, outputs.labels)
        # tail -> head + relation
        loss = loss + self.criterion(outputs.logits[:, :batch_size].t(), outputs.labels)

        return loss, outputs

    def _update_meters(self, meters, loss, outputs, batch_size):
        """Update tracking meters with current batch results."""
        acc1, acc3 = accuracy(outputs.logits, outputs.labels, topk=(1, 3))

        meters['top1'].update(acc1.item(), batch_size)
        meters['top3'].update(acc3.item(), batch_size)
        meters['inv_t'].update(outputs.inv_t.item(), 1)
        meters['losses'].update(loss.item(), batch_size)

    def _backward_pass(self, loss):
        """Execute backward pass with gradient clipping and optional AMP."""
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
        """Run validation and handle checkpointing (rank 0 only under DDP).

        Only rank 0 evaluates and writes checkpoint files -- running this on every rank
        would duplicate work and risk multiple processes writing the same file at once.
        The barrier keeps other ranks from racing ahead while rank 0 is still evaluating/saving.
        """
        if self.args.rank == 0:
            metric_dict = self.eval_epoch(epoch)
            is_best = self._check_best_metric(metric_dict)
            if is_best:
                self.best_metric = metric_dict

            self._save_checkpoint(epoch, step, is_best)

        if self.args.distributed:
            dist.barrier()

    def _check_best_metric(self, metric_dict):
        """Check if current metrics are the best so far."""
        if not self.valid_loader:
            return False
        if self.best_metric is None:
            return True
        return metric_dict.get('Acc@1', 0) > self.best_metric.get('Acc@1', 0)

    def _save_checkpoint(self, epoch, step, is_best):
        """Save model checkpoint and clean up old ones."""
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
        # (BertPredictor.load), which need just 'args' and 'state_dict'.
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
        """Evaluate the model on the validation set."""
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
        """Format meter values into a metric dictionary."""
        return {
            'Acc@1': round(meters['top1'].avg, 3),
            'Acc@3': round(meters['top3'].avg, 3),
            'loss': round(meters['losses'].avg, 3)
        }
