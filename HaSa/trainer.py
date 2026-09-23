import json
import random
import torch

import torch.nn as nn
import torch.utils.data

from typing import Dict, List, Optional
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc import Dataset, collate, Example, load_data, to_indices_and_mask
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from models import build_model
from dict_hub import build_tokenizer, get_link_graph, get_entity_dict, get_tokenizer
from logger_config import logger

# NOTE: evaluate.py / predict.py / doc.py (aside from the head/relation/tail split
# added to Example.vectorize()/collate(), see doc.py) are unchanged from SimKGC.
# HaSaBertModel.forward() still returns 'hr_vector' (e_hr) / 'tail_vector' (e_t), so
# the shared filtered-MRR evaluator (dot-product ranking, matching the paper's own
# exp(e_hr^T e_t) scoring) works without modification.
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

        # AMP scaler now lives on the trainer for the whole run (previously it
        # was only created inside train_loop()) so it can be saved to / restored
        # from a checkpoint -- needed for --resume to reproduce the exact
        # training state, not just the model weights.
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.start_epoch = 0

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.task)
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

        # HaSa always needs the link graph for false-negative correction (Eq. 9),
        # regardless of --use-link-graph (which only controls whether *text* context
        # from neighbours is appended to entity descriptions -- a separate concern).
        self.link_graph = get_link_graph()
        # Use the full training EntityDict here (not evaluate.py's, which may be a
        # smaller inductive-filtered copy for the wiki5m_ind task) since structural
        # negative candidates are drawn from the training graph.
        self.entity_dict = get_entity_dict()

        self.tau = args.tau
        self.num_false_neg_samples = args.num_false_neg_samples

        # --- MRR-based early stopping / best-model selection ---------------------
        self.early_stopping = EarlyStopping(patience=getattr(args, 'early_stop_patience', 5))
        self.full_eval_every_n_epoch = getattr(args, 'full_eval_every_n_epoch', 1)
        self.mrr_eval_batch_size = getattr(args, 'mrr_eval_batch_size', 256)
        # ---------------------------------------------------------------------------

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
            self.train_epoch(epoch)
            stop = self._run_epoch_end_eval(epoch)
            if stop:
                logger.info('Early stopping: no MRR improvement for {} full evals, '
                            'best MRR={:.4f}. Stopping at epoch {}.'
                            .format(self.args.early_stop_patience, self.early_stopping.best, epoch))
                break

    @torch.no_grad()
    def _compute_full_mrr(self) -> Dict[str, float]:
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
        # Saved mid-epoch (--eval-every-n-step), so the current epoch hasn't
        # finished yet -- record `epoch - 1` (the last fully-completed epoch) as
        # this checkpoint's resume point, so --resume re-runs the interrupted
        # epoch in full rather than silently skipping it.
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

    # ------------------------------------------------------------------------------
    # HaSa loss (Algorithm 1 / Eq. 6-9)
    # ------------------------------------------------------------------------------

    def _sample_false_negative_candidates(self, batch_exs: List[Example]) -> Optional[dict]:
        """alpha(t|e_hr) (Eq. 9): uniform over the head entity's <=2-hop link-graph
        neighbourhood. Candidates are de-duplicated and encoded once per batch; each
        row then samples `num_false_neg_samples` positions from its own neighbourhood
        (with replacement if the neighbourhood is smaller than that).
        """
        M = self.num_false_neg_samples
        rows = [
            list(self.link_graph.get_n_hop_entity_indices(
                ex.head_id, entity_dict=self.entity_dict, n_hop=2))
            for ex in batch_exs
        ]

        unique_idx = sorted({idx for row in rows for idx in row})
        if not unique_idx:
            return None
        idx_to_pos = {idx: pos for pos, idx in enumerate(unique_idx)}

        candidates = [Example(head_id='', relation='',
                              tail_id=self.entity_dict.get_entity_by_idx(idx).entity_id)
                     for idx in unique_idx]
        vectorized = [ex.vectorize() for ex in candidates]
        tail_token_ids, tail_mask = to_indices_and_mask(
            [torch.LongTensor(v['tail_token_ids']) for v in vectorized],
            pad_token_id=get_tokenizer().pad_token_id)
        tail_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(v['tail_token_type_ids']) for v in vectorized], need_mask=False)

        row_samples = []
        for row in rows:
            if not row:
                row_samples.append([])
                continue
            picks = random.sample(row, M) if len(row) >= M else random.choices(row, k=M)
            row_samples.append([idx_to_pos[p] for p in picks])

        return {
            'tail_token_ids': tail_token_ids,
            'tail_mask': tail_mask,
            'tail_token_type_ids': tail_token_type_ids,
            'row_samples': row_samples,
        }

    def _hasa_loss(self, e_hr: torch.Tensor, e_t: torch.Tensor, batch_exs: List[Example]):
        """L_HaSa(h,r,t) = -log( Pos / (Pos + NegHasa) ), Algorithm 1. The paper's
        pseudocode writes L_HaSa(h,r,t) = Pos/(Pos+NegHasa) directly as "the loss";
        taken literally that's a quantity you'd *maximize*, not minimize, which
        contradicts minimizing a loss — we use -log(...) instead, consistent with
        Eq. 6's actual InfoNCE-style formula (of which Algorithm 1 is presented as
        pseudocode for the same quantity).
        """
        inv_t = get_model_obj(self.model).log_inv_t.exp()
        B = e_hr.size(0)
        device = e_hr.device

        # Clamp before exp(): a plain (non log-sum-exp) translation of Algorithm 1's
        # Pos/Neg/FalseNeg quantities, so guard against inv_t drifting up during
        # training and blowing up exp() -- cheap insurance, doesn't change the loss
        # in the normal operating range.
        sim = torch.clamp(e_hr.mm(e_t.t()) * inv_t, min=-30.0, max=30.0)  # (B, B)
        pos = torch.exp(sim.diagonal())                                  # (B,)

        off_diag = ~torch.eye(B, dtype=torch.bool, device=device)
        neg_exp = torch.exp(sim) * off_diag
        neg_mean = neg_exp.sum(dim=1) / max(B - 1, 1)                    # Eq. 5, "Neg"
        K = max(B - 1, 1)

        false_neg_mean = torch.zeros(B, device=device)
        candidates = self._sample_false_negative_candidates(batch_exs)
        if candidates is not None:
            if torch.cuda.is_available():
                candidates = move_to_cuda(candidates)
            cand_vec = get_model_obj(self.model).encode_text(
                candidates['tail_token_ids'], candidates['tail_mask'], candidates['tail_token_type_ids'])
            cand_sim = torch.clamp(e_hr.mm(cand_vec.t()) * inv_t, min=-30.0, max=30.0)
            cand_exp = torch.exp(cand_sim)                                # (B, num_unique_candidates)
            for i, positions in enumerate(candidates['row_samples']):
                if positions:
                    idx = torch.tensor(positions, device=device, dtype=torch.long)
                    false_neg_mean[i] = cand_exp[i].index_select(0, idx).mean()

        # NegHasa = K * ( Neg/(1-tau) - tau*FalseNeg ) (Algorithm 1). Clamped to stay
        # positive -- a numerical safety net not spelled out in the paper, needed
        # because the correction term can in principle overshoot for small batches.
        neg_hasa = K * (neg_mean / (1.0 - self.tau) - self.tau * false_neg_mean)
        neg_hasa = torch.clamp(neg_hasa, min=1e-9)

        loss = -torch.log(pos / (pos + neg_hasa) + 1e-12)
        return loss.mean(), pos, neg_mean, false_neg_mean

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        pos_meter = AverageMeter('Pos', ':.4')
        neg_meter = AverageMeter('Neg', ':.4')
        fneg_meter = AverageMeter('FalseNeg', ':.4')
        inv_t_meter = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, pos_meter, neg_meter, fneg_meter, inv_t_meter],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()
            batch_exs = batch_dict['batch_data']
            batch_size = len(batch_exs)

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)

            e_hr, e_t = outputs['hr_vector'], outputs['tail_vector']
            loss, pos, neg, false_neg = self._hasa_loss(e_hr, e_t, batch_exs)

            losses.update(loss.item(), batch_size)
            pos_meter.update(pos.mean().item(), batch_size)
            neg_meter.update(neg.mean().item(), batch_size)
            fneg_meter.update(false_neg.mean().item(), batch_size)
            inv_t_meter.update(get_model_obj(self.model).log_inv_t.detach().exp().item(), 1)

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
            assert False, 'Unknown lr scheduler: {}'.format(self.args.lr_scheduler)
