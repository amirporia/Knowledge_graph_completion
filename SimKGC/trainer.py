import glob
import json
import torch
import shutil

import torch.nn as nn
import torch.utils.data

from typing import Dict, List, Optional
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from doc import Dataset, collate, Example, load_data
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger

# NOTE: importing evaluate here (rather than only in a standalone eval script) is what
# lets the trainer reuse the *exact* filtered-ranking MRR computation used at test time
# for early stopping / best-checkpoint selection, instead of a proxy in-batch metric.
from evaluate import compute_metrics, entity_dict


class EarlyStopping:
    """Stops training when validation MRR hasn't improved for `patience` full evals."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        improved = self.best is None or metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class _LiveModelAdapter:
    """Minimal shim exposing the same `predict_by_examples` / `predict_by_entities`
    interface as predict.BertPredictor, but pointed at the model that is *currently
    training* (no checkpoint round-trip through disk needed to compute MRR each eval).
    """

    def __init__(self, model: nn.Module, task: str, batch_size: int, use_cuda: bool):
        self.model = model
        self.task = task
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.task),
            num_workers=2,
            batch_size=self.batch_size,
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list = [], []
        for batch_dict in data_loader:
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            tail_tensor_list.append(outputs['tail_vector'])

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = [Example(head_id='', relation='', tail_id=ex.entity_id) for ex in entity_exs]
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.task),
            num_workers=2,
            batch_size=max(self.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        for batch_dict in data_loader:
            batch_dict['only_ent_embedding'] = True
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)


class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        # AMP scaler now lives on the trainer for the whole run (previously it
        # was only created inside train_loop()) so it can be saved to / restored
        # from a checkpoint -- needed for --resume to reproduce the exact
        # training state, not just the model weights.
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.start_epoch = 0

        # define loss function (criterion) and optimizer.
        # BUGFIX: this was previously `nn.CrossEntropyLoss().cuda()` -- an
        # *unconditional* CUDA call that crashes with
        # "AssertionError: Torch not compiled with CUDA enabled" / "No CUDA
        # GPUs are available" on any CPU-only machine, even though
        # `_setup_training()` right above it already picks CPU as a fallback.
        # `.to(self.device)` matches that fallback.
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

        # --- MRR-based early stopping / best-model selection ---------------------
        self.early_stopping = EarlyStopping(patience=getattr(args, 'early_stop_patience', 5))
        self.full_eval_every_n_epoch = getattr(args, 'full_eval_every_n_epoch', 1)
        self.mrr_eval_batch_size = getattr(args, 'mrr_eval_batch_size', 256)
        # --------------------------------------------------------------------------

        self._maybe_resume()

    def _maybe_resume(self):
        """Restore model/optimizer/scheduler/AMP-scaler state, the best-metric seen
        so far, and the early-stopping counter from `--resume-path` (default
        <model-dir>/model_last.mdl), and continue training at the following epoch.
        A no-op unless `--resume` was passed.
        """
        if not self.args.resume:
            return

        checkpoint = torch.load(self.args.resume_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.args.use_amp and self.scaler is not None and checkpoint.get('scaler') is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.best_metric = checkpoint.get('best_metric')
        early_stopping_state = checkpoint.get('early_stopping') or {}
        self.early_stopping.best = early_stopping_state.get('best')
        self.early_stopping.counter = early_stopping_state.get('counter', 0)

        self.start_epoch = checkpoint.get('epoch', -1) + 1
        logger.info(
            'Resumed from {} (checkpoint epoch {}, resuming at epoch {})'.format(
                self.args.resume_path, checkpoint.get('epoch'), self.start_epoch))

    def train_loop(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            stop = self._run_epoch_end_eval(epoch)
            if stop:
                logger.info('Early stopping: no MRR improvement for {} full evals, '
                            'best MRR={:.4f}. Stopping at epoch {}.'
                            .format(self.args.early_stop_patience, self.early_stopping.best, epoch))
                break

    @torch.no_grad()
    def _compute_full_mrr(self) -> Dict[str, float]:
        """Filtered forward+backward MRR/Hits@k on args.valid_path, computed with the
        exact same routine (evaluate.compute_metrics) used for final test-set reporting.
        This — not the in-batch classification accuracy below — is what best-checkpoint
        selection and early stopping are keyed on, per the "MRR for best model
        selection" requirement.
        """
        was_training = self.model.training
        self.model.eval()
        was_is_test = self.args.is_test
        self.args.is_test = True

        adapter = _LiveModelAdapter(get_model_obj(self.model), task=self.args.task,
                                    batch_size=self.args.batch_size,
                                    use_cuda=torch.cuda.is_available())
        entity_tensor = adapter.predict_by_entities(entity_dict.entity_exs)
        if torch.cuda.is_available():
            entity_tensor = entity_tensor.cuda()

        direction_metrics = []
        for eval_forward in (True, False):
            examples = load_data(self.args.valid_path,
                                 add_forward_triplet=eval_forward,
                                 add_backward_triplet=not eval_forward)
            hr_tensor, _ = adapter.predict_by_examples(examples)
            hr_tensor = hr_tensor.to(entity_tensor.device)
            target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]

            _, _, metrics, _ = compute_metrics(
                hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                target=target, examples=examples, batch_size=self.mrr_eval_batch_size)
            direction_metrics.append(metrics)

        avg_metrics = {k: round((direction_metrics[0][k] + direction_metrics[1][k]) / 2, 4)
                      for k in direction_metrics[0]}

        self.args.is_test = was_is_test
        if was_training:
            self.model.train()
        return avg_metrics

    def _run_epoch_end_eval(self, epoch: int) -> bool:
        """Runs at the end of every epoch. Always logs the cheap in-batch metric;
        every `full_eval_every_n_epoch` epochs (and always on the last epoch) also runs
        the expensive full-corpus MRR eval used for early stopping / checkpoint
        selection. Returns True iff training should stop.
        """
        light_metric_dict = self.eval_epoch(epoch)

        run_full_eval = (
            (epoch + 1) % self.full_eval_every_n_epoch == 0
            or epoch == self.args.epochs - 1
        ) and self.args.valid_path

        is_best = False
        mrr_metrics = None
        if run_full_eval:
            mrr_metrics = self._compute_full_mrr()
            is_best = self.early_stopping.step(mrr_metrics['mrr'])
            self.best_metric = mrr_metrics if is_best else self.best_metric
            logger.info('Epoch {} full MRR metrics: {} (best so far: {:.4f})'.format(
                epoch, json.dumps(mrr_metrics), self.early_stopping.best))

        filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        model_state = self.model.state_dict()
        # Full state -> checkpoint_epoch{N}.mdl and model_last.mdl: everything
        # needed to resume training exactly (--resume).
        full_state = {
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if (self.args.use_amp and self.scaler is not None) else None,
            'best_metric': self.best_metric,
            'early_stopping': {
                'best': self.early_stopping.best,
                'counter': self.early_stopping.counter,
            },
            'light_metrics': light_metric_dict,
            'mrr_metrics': mrr_metrics,
        }
        # Light state -> model_best.mdl only: all predict.py/evaluate.py ever read.
        eval_state = {
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': model_state,
        }
        save_checkpoint(full_state, is_best=is_best, filename=filename, eval_state=eval_state)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

        return run_full_eval and self.early_stopping.should_stop

    def _save_periodic_checkpoint(self, epoch, step):
        """Mid-epoch checkpoint (from --eval-every-n-step). Records `epoch - 1`
        (the last fully-completed epoch) as this checkpoint's resume point,
        since the current epoch hasn't finished -- so --resume re-runs the
        interrupted epoch in full rather than silently skipping it.
        """
        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        save_checkpoint({
            'epoch': epoch - 1,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if (self.args.use_amp and self.scaler is not None) else None,
            'best_metric': self.best_metric,
            'early_stopping': {
                'best': self.early_stopping.best,
                'counter': self.early_stopping.counter,
            },
        }, is_best=False, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        """Cheap in-batch classification metric (Acc@1/Acc@3/loss), kept only for
        logging/diagnostics — no longer used to pick the best checkpoint (see
        `_compute_full_mrr` / `_run_epoch_end_eval` above).
        """
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            assert logits.size(0) == batch_size
            # head + relation -> tail
            loss = self.criterion(logits, labels)
            # tail -> head + relation
            loss += self.criterion(logits[:, :batch_size].t(), labels)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t, 1)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
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
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
            if self.args.eval_every_n_step > 0 and (i + 1) % self.args.eval_every_n_step == 0:
                self._save_periodic_checkpoint(epoch=epoch, step=i + 1)
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            # BUGFIX: was `self.args.scheduler`, which doesn't exist -- the
            # field is `self.args.lr_scheduler` (see config.py). Unreachable
            # today since config.py already asserts lr_scheduler is 'linear'
            # or 'cosine' before Trainer is constructed, but would have raised
            # an unrelated AttributeError instead of this assertion message if
            # that guard were ever removed or this method called directly.
            assert False, 'Unknown lr scheduler: {}'.format(self.args.lr_scheduler)
