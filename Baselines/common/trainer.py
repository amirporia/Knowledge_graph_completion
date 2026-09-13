import json
import os
import time
from typing import Dict, Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from .data import (KGIndex, NegSamplingDataset, collate_neg_sampling,
                    build_message_passing_graph)
from .metrics import evaluate_mrr
from .utils import AverageMeter, EarlyStopping, save_checkpoint, delete_old_checkpoints, move_to_device

import logging
logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """Optimized, model-agnostic training loop shared by all 7 embedding baselines.

    Optimizations:
      - AMP (mixed precision) when args.use_amp and CUDA is available.
      - RGCN's graph encoder runs once per epoch (not once per batch); see
        `model.requires_graph_encode`.
      - Filtered negative sampling avoids wasted/incorrect gradient signal from
        accidental false negatives.
      - Early stopping + best-checkpoint selection by validation MRR (not just loss),
        matching the "best model = highest MRR" criterion requested for every baseline.
    """

    def __init__(self, model, kg_index: KGIndex, train_triples: torch.LongTensor,
                 valid_triples: torch.LongTensor, true_tail_filter, args):
        self.args = args
        self.kg_index = kg_index
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.true_tail_filter = true_tail_filter

        self.dataset = NegSamplingDataset(
            train_triples, kg_index.num_entities, args.neg_size, true_tail_filter)
        self.loader = DataLoader(
            self.dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=collate_neg_sampling,
            pin_memory=torch.cuda.is_available(), drop_last=True)

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.early_stopping = EarlyStopping(patience=args.patience, mode='max')
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())

        self._graph = None
        if getattr(self.model, 'requires_graph_encode', False):
            edge_index, edge_type, edge_norm = build_message_passing_graph(
                train_triples, kg_index.num_entities, kg_index.num_relations)
            self._graph = (edge_index.to(self.device), edge_type.to(self.device),
                           edge_norm.to(self.device))

    def _maybe_encode_graph(self) -> None:
        if self._graph is not None:
            self.model.encode_graph(*self._graph)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self._maybe_encode_graph()  # cache fresh entity embeddings for this epoch (RGCN only)

        loss_meter = AverageMeter('Loss', ':.4f')
        for i, batch in enumerate(self.loader):
            batch = move_to_device(batch, self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                # RGCN's graph-derived embeddings are a function of model parameters,
                # so autograd still flows correctly even though encode_graph() was
                # called once above (it isn't wrapped in no_grad).
                loss = self.model(batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(loss.item(), batch['h'].size(0))

            if i % self.args.print_freq == 0:
                logger.info(f'Epoch [{epoch}] step [{i}/{len(self.loader)}] '
                            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})')
        return loss_meter.avg

    @torch.no_grad()
    def evaluate(self, triples: torch.LongTensor, desc: str = 'valid') -> Dict[str, float]:
        self.model.eval()
        self._maybe_encode_graph()
        return evaluate_mrr(self.model, triples, self.true_tail_filter,
                            self.kg_index.num_entities, batch_size=self.args.eval_batch_size,
                            device=self.device, desc=desc)

    def fit(self) -> Dict[str, float]:
        best_metrics = {}
        for epoch in range(self.args.epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)

            if (epoch + 1) % self.args.eval_every == 0 or epoch == self.args.epochs - 1:
                metrics = self.evaluate(self.valid_triples, desc=f'valid@{epoch}')
                is_best = self.early_stopping.step(metrics['mrr'])
                logger.info(f'Epoch {epoch} | train_loss={train_loss:.4f} | '
                            f'valid={json.dumps(metrics)} | best_mrr={self.early_stopping.best:.4f} | '
                            f'{round(time.time() - t0, 1)}s')

                if is_best:
                    best_metrics = metrics
                state = {
                    'epoch': epoch,
                    'args': vars(self.args),
                    'state_dict': self.model.state_dict(),
                    'valid_metrics': metrics,
                }
                save_checkpoint(state, is_best=is_best, model_dir=self.args.model_dir)
                delete_old_checkpoints(os.path.join(self.args.model_dir, 'checkpoint_*.mdl'),
                                       keep=self.args.max_to_keep)

                if self.early_stopping.should_stop:
                    logger.info(f'Early stopping at epoch {epoch} '
                                f'(no MRR improvement for {self.args.patience} evals).')
                    break
            else:
                logger.info(f'Epoch {epoch} | train_loss={train_loss:.4f} | '
                            f'{round(time.time() - t0, 1)}s')

        return best_metrics
