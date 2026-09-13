import json
import torch

import torch.nn as nn
import torch.utils.data

from typing import Dict, List, Optional
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc import Dataset, collate, Example, load_data
from star_data import StarDataset, collate_star
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from models import build_model
from dict_hub import build_tokenizer
from logger_config import logger

# NOTE: evaluate.py / predict.py / doc.py are unchanged from SimKGC — StarBertModel's
# forward() still returns 'hr_vector' (u) / 'tail_vector' (v), so the shared filtered-
# MRR evaluator works without modification. See models.py's module docstring for why
# ranking uses s^d rather than the paper's s^c.
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
    """Minimal shim exposing predict.BertPredictor's interface but pointed at the
    model currently training, so full-corpus MRR can be computed without a checkpoint
    round-trip through disk.
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
            num_workers=2, batch_size=max(self.batch_size, 512),
            collate_fn=collate, shuffle=False)

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
            num_workers=2, batch_size=max(self.batch_size, 1024),
            collate_fn=collate, shuffle=False)

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

        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none').cuda() \
            if torch.cuda.is_available() else nn.BCEWithLogitsLoss(reduction='none')

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        # Section 3.3.1: K explicit negatives per positive triple (not in-batch
        # contrastive), resampled fresh every epoch by StarDataset.__getitem__.
        train_dataset = StarDataset(path=args.train_path, num_negatives=args.num_negatives)
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_star,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        # --- MRR-based early stopping / best-model selection ---------------------
        self.early_stopping = EarlyStopping(patience=getattr(args, 'early_stop_patience', 5))
        self.full_eval_every_n_epoch = getattr(args, 'full_eval_every_n_epoch', 1)
        self.mrr_eval_batch_size = getattr(args, 'mrr_eval_batch_size', 256)
        # ---------------------------------------------------------------------------

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            stop = self._run_epoch_end_eval(epoch)
            if stop:
                logger.info('Early stopping: no MRR improvement for {} full evals, '
                            'best MRR={:.4f}. Stopping at epoch {}.'
                            .format(self.args.early_stop_patience, self.early_stopping.best, epoch))
                break

    @torch.no_grad()
    def _compute_full_mrr(self) -> Dict[str, float]:
        """Filtered forward+backward MRR/Hits@k on args.valid_path via evaluate.py's
        own compute_metrics — the same routine used for final test-set reporting.
        Drives early stopping and which checkpoint becomes model_best.mdl.
        """
        was_training = self.model.training
        self.model.eval()
        # Temporarily flag "test mode" so doc.py's collate() skips building unused
        # negative-sampling masks for this pure inference/embedding pass (mirrors
        # what predict.py's BertPredictor does when loading a checkpoint for eval).
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
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
            'mrr_metrics': mrr_metrics,
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

        return run_full_eval and self.early_stopping.should_stop

    def _save_periodic_checkpoint(self, epoch, step):
        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=False, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        cls_losses = AverageMeter('L^c', ':.4')
        struct_losses = AverageMeter('L^d', ':.4')
        pos_rank1 = AverageMeter('PosRank1', ':6.3f')  # fraction where s^d(pos) beats every sampled negative
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, cls_losses, struct_losses, pos_rank1],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()
            batch_size, num_negatives = batch_dict['batch_size'], batch_dict['num_negatives']
            group = 1 + num_negatives

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)

            u = outputs['hr_vector'].view(batch_size, group, -1)
            v = outputs['tail_vector'].view(batch_size, group, -1)

            # L^c (Eq. 8-9, 12): classification objective over [u; u*v; u-v; v]
            logits_c = get_model_obj(self.model).interaction_logits(
                u.reshape(batch_size * group, -1), v.reshape(batch_size * group, -1)
            ).view(batch_size, group)
            labels_c = torch.zeros_like(logits_c)
            labels_c[:, 0] = 1.0
            # Eq. 12's 1/(1+|N(tp)|) weighting per triple group, then averaged over the batch
            loss_c = self.bce_criterion(logits_c, labels_c).mean(dim=1).mean()

            # L^d (Eq. 11, 13): spatial structure / margin hinge objective
            s_d = -torch.norm(u - v, p=2, dim=-1)                        # (B, group)
            pos_s_d = s_d[:, :1]
            neg_s_d = s_d[:, 1:]
            hinge = torch.clamp(self.args.margin - pos_s_d + neg_s_d, min=0.0)
            loss_d = hinge.mean()

            # L = L^c + gamma * L^d (Eq. 14)
            loss = loss_c + self.args.structure_loss_weight * loss_d

            with torch.no_grad():
                acc = (pos_s_d.squeeze(-1) > neg_s_d.max(dim=1).values).float().mean()

            losses.update(loss.item(), batch_size)
            cls_losses.update(loss_c.item(), batch_size)
            struct_losses.update(loss_d.item(), batch_size)
            pos_rank1.update(acc.item(), batch_size)

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
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._save_periodic_checkpoint(epoch=epoch, step=i + 1)
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

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
            assert False, 'Unknown lr scheduler: {}'.format(self.args.lr_scheduler)
