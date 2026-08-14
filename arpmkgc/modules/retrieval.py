"""
Query-conditioned anchor selection (S_A, Sec 4.4), diversity regularization
(L_div, Sec 4.4.1), and the optional discrete Gumbel-Sigmoid keep/drop gate
(Sec 4.13.1 / ablation A11).

Config knobs (see `ARPMConfig`):
  anchor_selection_mode: 'learned' (default) | 'random' | 'uniform'
      'learned' -> S_A(q, a_i, r, d_i) as defined in 4.4.
      'random'  -> ablation A1: query-independent noise replaces s_i, so
                   alpha_i comes from a softmax over random scores rather
                   than a query-conditioned scorer.
      'uniform' -> alpha_i = 1 / N_A for all valid anchors (used e.g. by the
                   faithful-baseline preset's simpler anchor-averaging
                   mechanism, see ablations.py).
  anchor_gating_mode: 'none' (default, dense) | 'gumbel_sigmoid' (A11)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .gumbel import gumbel_sigmoid_st, straight_through
from ..candidates import GLOBAL_HOP
from ..config import ARPMConfig


class AnchorRelevanceScorer(nn.Module):
    """S_A(q, a_i, r, d_i): a small MLP over [q, a_i, r_emb, hop_emb]."""

    def __init__(self, hidden_dim: int, num_relations: int, max_hop: int):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        # +1 slot for GLOBAL_HOP (0); local hops occupy 1..max_hop.
        self.hop_embedding = nn.Embedding(max_hop + 1, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q: torch.Tensor, anchors: torch.Tensor, relation_ids: torch.Tensor,
                hop_ids: torch.Tensor) -> torch.Tensor:
        """q: [B, d], anchors: [B, NA, d], relation_ids: [B], hop_ids: [B, NA]
        -> scores s_i: [B, NA]"""
        b, na, d = anchors.shape
        q_exp = q.unsqueeze(1).expand(b, na, d)
        r_emb = self.relation_embedding(relation_ids).unsqueeze(1).expand(b, na, d)
        hop_emb = self.hop_embedding(hop_ids.clamp(min=0, max=self.hop_embedding.num_embeddings - 1))
        feats = torch.cat([q_exp, anchors, r_emb, hop_emb], dim=-1)
        return self.scorer(feats).squeeze(-1)


class AnchorSelector(nn.Module):
    """Produces the weighted anchor set W_q = {(a_i, alpha_i)} per Sec 4.4,
    plus the diversity regularizer L_div and (optionally) the discrete
    keep/drop gate of Sec 4.13.1.
    """

    def __init__(self, config: ARPMConfig, hidden_dim: int, num_relations: int):
        super().__init__()
        self.config = config
        self.scorer = AnchorRelevanceScorer(hidden_dim, num_relations, config.max_hop)

    def forward(self, q: torch.Tensor, anchors: torch.Tensor, anchor_mask: torch.Tensor,
                relation_ids: torch.Tensor, hop_ids: torch.Tensor,
                gumbel_tau: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (alpha, raw_scores, gate) all shaped [B, NA]. `gate` is
        all-ones when `anchor_gating_mode == 'none'`.
        """
        cfg = self.config
        mode = cfg.anchor_selection_mode

        if mode == "learned":
            raw_scores = self.scorer(q, anchors, relation_ids, hop_ids)
        elif mode == "random":
            raw_scores = torch.rand_like(anchor_mask.float()) * 2 - 1  # query-independent noise
        elif mode == "uniform":
            raw_scores = torch.zeros_like(anchor_mask.float())
        else:
            raise ValueError(f"Unknown anchor_selection_mode: {mode}")

        masked_scores = raw_scores.masked_fill(~anchor_mask, float("-inf"))
        alpha = torch.softmax(masked_scores / max(cfg.retrieval_temperature, 1e-6), dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # rows with zero valid anchors -> all-zero

        gate = torch.ones_like(alpha)
        if cfg.anchor_gating_mode == "gumbel_sigmoid":
            tau = gumbel_tau if gumbel_tau is not None else (cfg.gumbel_sel_temperature or 1.0)
            pi_logit = raw_scores  # sigma(s_i) reused as the keep probability (Sec 4.13.1)
            c_soft, c_hard = gumbel_sigmoid_st(pi_logit, tau, mask=anchor_mask)
            gate = straight_through(c_hard, c_soft) if self.training else c_hard

        effective_alpha = alpha * gate
        return effective_alpha, raw_scores, gate

    @staticmethod
    def diversity_loss(alpha: torch.Tensor, anchors: torch.Tensor,
                        anchor_mask: torch.Tensor) -> torch.Tensor:
        """L_div = (1/Z) * sum_{i != j} alpha_i alpha_j sim(a_i, a_j),
        Z = N_A (N_A - 1), averaged over the batch (Sec 4.4.1)."""
        b, na, _ = anchors.shape
        norm_anchors = torch.nn.functional.normalize(anchors, dim=-1)
        sim = torch.bmm(norm_anchors, norm_anchors.transpose(1, 2))  # [B, NA, NA]

        outer_alpha = alpha.unsqueeze(2) * alpha.unsqueeze(1)  # [B, NA, NA]
        off_diag = ~torch.eye(na, dtype=torch.bool, device=anchors.device).unsqueeze(0)
        pair_mask = anchor_mask.unsqueeze(2) & anchor_mask.unsqueeze(1) & off_diag

        weighted = outer_alpha * sim * pair_mask.float()
        per_example_sum = weighted.sum(dim=(1, 2))

        n_valid = anchor_mask.float().sum(dim=1)
        z = (n_valid * (n_valid - 1)).clamp(min=1.0)
        per_example_div = per_example_sum / z

        has_pairs = (n_valid >= 2).float()
        if has_pairs.sum() == 0:
            return torch.zeros((), device=anchors.device)
        return (per_example_div * has_pairs).sum() / has_pairs.sum()

    @staticmethod
    def retrieval_supervision_loss(raw_scores: torch.Tensor, anchor_mask: torch.Tensor,
                                    anchor_tail_ids, gold_tail_ids) -> torch.Tensor:
        """Optional L_rel (Sec 4.4.1): weak supervision when explicit anchor
        relevance labels are unavailable is approximated by treating anchors
        whose tail equals the query's own gold tail as positives (they are
        literally corroborating evidence for the same answer) and all other
        *local* candidates within the same (h, r) group as negatives. This
        heuristic is used only when `use_retrieval_supervision=True`; the
        proposal explicitly allows training with L_final alone otherwise.
        """
        labels = torch.zeros_like(raw_scores)
        for b in range(len(gold_tail_ids)):
            for a in range(len(anchor_tail_ids[b])):
                if anchor_tail_ids[b][a] == gold_tail_ids[b]:
                    labels[b, a] = 1.0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            raw_scores, labels, reduction="none"
        )
        loss = loss * anchor_mask.float()
        denom = anchor_mask.float().sum().clamp(min=1.0)
        return loss.sum() / denom
