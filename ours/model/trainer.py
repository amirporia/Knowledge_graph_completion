import json
from typing import Dict

import torch
import torch.nn as nn
import torch.utils.data
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

        self._initialize_tokenizer()
        self._build_model()
        self._setup_device()
        self._init_optimizer_and_criterion()
        self._init_data_loaders()
        self._init_scheduler()

    def _initialize_tokenizer(self):
        """Initialize the tokenizer based on args."""
        init_tokenizer(self.args)

    def _build_model(self):
        """Build and log the model architecture."""
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)

    def _setup_device(self):
        """Setup the training device (CPU/GPU)."""
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _init_optimizer_and_criterion(self):
        """Initialize loss function, optimizer, and log trainable parameters."""
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        report_num_trainable_parameters(self.model)

    def _init_data_loaders(self):
        """Initialize training and validation data loaders."""
        self.train_dataset = Dataset(path=self.args.train_path, test_set=False)
        self.valid_dataset = (
            Dataset(path=self.args.valid_path, test_set=False)
            if self.args.valid_path else None
        )

        self.train_loader = self._create_data_loader(self.train_dataset, shuffle=True, drop_last=True)
        self.valid_loader = self._create_data_loader(self.valid_dataset, shuffle=True) if self.valid_dataset else None

    def _create_data_loader(self, dataset, shuffle, drop_last=False):
        """Create a DataLoader with standard configuration."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.args.workers,
            pin_memory=False,
            drop_last=drop_last
        )

    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        num_training_steps = (
                self.args.epochs * len(self.train_dataset) // max(self.args.batch_size, 1)
        )
        self.args.warmup = min(self.args.warmup, num_training_steps // 10)
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

    def train_loop(self):
        """Main training loop over epochs."""
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        meters = self._init_training_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [meters['losses'], meters['inv_t'], meters['top1'], meters['top3'],
             meters['hr_losses'], meters['related_losses']],
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

            if i % self.args.print_freq == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)

        logger.info(f'Learning rate: {self.scheduler.get_last_lr()[0]}')

    def _init_training_meters(self):
        """Initialize AverageMeter objects for tracking metrics."""
        return {
            'losses': AverageMeter('Loss', ':.4'),
            'related_losses': AverageMeter('RelatedLoss', ':.4'),
            'hr_losses': AverageMeter('HRLoss', ':.4'),
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

    def _compute_losses(self, outputs, batch_dict):
        """Compute total loss and component losses."""
        outputs = get_model_obj(self.model).compute_logits(
            output_dict=outputs, batch_dict=batch_dict
        )
        outputs = ModelOutput(**outputs)

        # Compute related loss (head+relation -> tail and tail -> head+relation)
        related_loss = self._compute_bidirectional_loss(
            outputs.related_logits, outputs.related_labels
        )

        # Compute HR loss (head+relation -> tail and tail -> head+relation)
        hr_loss = self._compute_bidirectional_loss(
            outputs.hr_logits, outputs.hr_labels
        )

        total_loss = 0.2 * related_loss + hr_loss

        return {
            'total_loss': total_loss,
            'related_loss': related_loss,
            'hr_loss': hr_loss,
            'hr_logits': outputs.hr_logits,
            'hr_labels': outputs.hr_labels
        }

    def _compute_bidirectional_loss(self, logits, labels):
        """Compute loss in both directions for relation prediction."""
        assert logits.size(0) == self.args.batch_size

        loss = self.criterion(logits, labels)
        loss += self.criterion(logits[:, :self.args.batch_size].t(), labels)

        return loss

    def _update_meters(self, meters, loss_components):
        """Update tracking meters with current batch results."""
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
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )
            self.optimizer.step()

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        """Run evaluation and handle checkpointing."""
        metric_dict = self.eval_epoch(epoch)
        is_best = self._check_best_metric(metric_dict)

        if is_best:
            self.best_metric = metric_dict

        self._save_checkpoint(epoch, step, is_best)

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

        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)

        delete_old_checkpoints(
            path_pattern=f'{self.args.model_dir}/checkpoint_*.mdl',
            keep=self.args.max_to_keep
        )

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        """Evaluate the model on validation set."""
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

            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(
                output_dict=outputs, batch_dict=batch_dict
            )
            outputs = ModelOutput(**outputs)

            loss = self.criterion(outputs.hr_logits, outputs.hr_labels)
            acc1, acc3 = accuracy(outputs.hr_logits, outputs.hr_labels, topk=(1, 3))

            batch_size = self.args.batch_size
            meters['losses'].update(loss.item(), batch_size)
            meters['top1'].update(acc1.item(), batch_size)
            meters['top3'].update(acc3.item(), batch_size)

        metric_dict = self._format_metrics(meters)
        logger.info(f'Epoch {epoch}, valid metric: {json.dumps(metric_dict)}')

        return metric_dict

    def _format_metrics(self, meters):
        """Format meter values into a metric dictionary."""
        return {
            'Acc@1': round(meters['top1'].avg, 3),
            'Acc@3': round(meters['top3'].avg, 3),
            'loss': round(meters['losses'].avg, 3)
        }
