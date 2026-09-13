from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGEModel(nn.Module):
    """Common training/eval contract for every triple-based baseline.

    Design choice (documented so it's easy to challenge): rather than giving every
    baseline its own bespoke loss (margin ranking for TransE/RotatE, 1-N BCE for
    DistMult/ComplEx/ConvE, ...), all 7 models are trained with RotatE's
    *self-adversarial negative sampling loss* against the same negative-sampling
    dataset. This is a single, well-optimized, vectorized loss that is known to work
    well across both distance-based and semantic-matching scoring functions (it's how
    the RotatE paper itself trains its TransE/DistMult/ComplEx baselines), and it keeps
    common/trainer.py, common/metrics.py and common/data.py completely model-agnostic.
    Subclasses only need to implement `score()` and `score_all()`.

    Score convention: **higher score = more plausible triple** for every subclass.
    `distance_based=True` subclasses (TransE, RotatE) define score = -distance and use
    `margin` as RotatE's gamma; similarity-based subclasses leave margin unused.
    """

    distance_based: bool = False

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.args = args
        self.margin = getattr(args, 'margin', 0.0)
        self.adv_temperature = getattr(args, 'adv_temperature', 1.0)
        self.eval_chunk_size = getattr(args, 'eval_entity_chunk', 20000)

    # ---- to be implemented by subclasses -----------------------------------------
    def score(self, h_idx: torch.Tensor, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """(B,) plausibility score for a batch of (h, r, t) index triples."""
        raise NotImplementedError

    def score_all(self, h_idx: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
        """(B, num_entities) plausibility score of (h, r, ?) against every entity.
        Default implementation chunks over entities calling `score()`; subclasses with
        a bilinear/multilinear form override this with a single fast matmul.
        """
        B = h_idx.size(0)
        out = torch.empty(B, self.num_entities, device=h_idx.device)
        for start in range(0, self.num_entities, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, self.num_entities)
            cand = torch.arange(start, end, device=h_idx.device)
            n_cand = cand.size(0)
            hh = h_idx.repeat_interleave(n_cand)
            rr = r_idx.repeat_interleave(n_cand)
            tt = cand.repeat(B)
            s = self.score(hh, rr, tt).view(B, n_cand)
            out[:, start:end] = s
        return out

    # ---- shared loss ----------------------------------------------------------
    def _margin_shift(self, score: torch.Tensor) -> torch.Tensor:
        return self.margin + score if self.distance_based else score

    def forward(self, batch: dict) -> torch.Tensor:
        pos_score = self.score(batch['h'], batch['r'], batch['t'])                    # (B,)
        neg_score = self.score(
            batch['neg_h'].reshape(-1), batch['neg_r'].reshape(-1), batch['neg_t'].reshape(-1)
        ).view(batch['h'].size(0), -1)                                                # (B, N)

        with torch.no_grad():
            neg_weight = F.softmax(self._margin_shift(neg_score) * self.adv_temperature, dim=-1)

        pos_loss = -F.logsigmoid(self._margin_shift(pos_score)).mean()
        neg_loss = -(neg_weight * F.logsigmoid(-self._margin_shift(neg_score))).sum(dim=-1).mean()
        return (pos_loss + neg_loss) / 2.0


def xavier_embedding(num: int, dim: int) -> nn.Parameter:
    w = torch.empty(num, dim)
    nn.init.xavier_uniform_(w)
    return nn.Parameter(w)
