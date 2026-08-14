"""
Adaptive structural memory (Sec 4.6) and the optional discrete hop-selection
extension (Sec 4.13.2 / ablations A5, A12).

Per Sec 4.6, structural memory reuses the *already weighted* local anchor
embeddings from the retrieval stage (Sec 4.4) -- no separate entity encoder
or structural attention scorer is introduced here; H_l is literally the
alpha-weighted mean of hop-l local anchors (`aggregate_hop_memories`).

`hop_selection_mode`:
    'soft'            (default) beta_l = softmax_l(z_l)                  (Sec 4.6)
    'uniform'          A5:        beta_l = 1 / |valid hops|
    'gumbel_softmax'   A12:       Gumbel-top-k_hop straight-through pick   (Sec 4.13.2)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .gumbel import gumbel_softmax_top_k_st, straight_through
from ..config import ARPMConfig


def aggregate_hop_memories(anchors: torch.Tensor, alpha: torch.Tensor, anchor_hop: torch.Tensor,
                            anchor_is_local: torch.Tensor, max_hop: int,
                            eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds m^(l) for l = 1..L (Sec 4.6's aggregation formula).

    Returns:
        m_hop: [B, L, d]      hop-specific structural representations
        hop_has_data: [B, L]  bool, True where at least one local anchor
                               with alpha > 0 was assigned to that hop
    """

    hop_ids = anchor_hop.clamp(min=0, max=max_hop)  # GLOBAL_HOP anchors map to index 0 (dropped below)
    onehot = torch.nn.functional.one_hot(hop_ids, num_classes=max_hop + 1).float()  # [B, NA, L+1]
    onehot = onehot[:, :, 1:]  # drop the GLOBAL_HOP column -> [B, NA, L]
    onehot = onehot * anchor_is_local.unsqueeze(-1).float()

    weights = alpha.unsqueeze(-1) * onehot  # [B, NA, L]
    numerator = torch.einsum("bnl,bnd->bld", weights, anchors)  # [B, L, d]
    denom_raw = weights.sum(dim=1)  # [B, L]
    m_hop = numerator / (denom_raw.unsqueeze(-1) + eps)

    hop_has_data = denom_raw > 0
    return m_hop, hop_has_data


class HopScorer(nn.Module):
    """G_hop(query_repr, r, l): scores each hop distance (Sec 4.6)."""

    def __init__(self, hidden_dim: int, num_relations: int, max_hop: int):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        self.hop_embedding = nn.Embedding(max_hop, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.max_hop = max_hop

    def forward(self, query_repr: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        b, d = query_repr.shape
        L = self.max_hop
        q_exp = query_repr.unsqueeze(1).expand(b, L, d)
        r_emb = self.relation_embedding(relation_ids).unsqueeze(1).expand(b, L, d)
        hop_emb = self.hop_embedding(torch.arange(L, device=query_repr.device)).unsqueeze(0).expand(b, L, d)
        feats = torch.cat([q_exp, r_emb, hop_emb], dim=-1)
        return self.scorer(feats).squeeze(-1)  # [B, L] == z_l (index 0 <-> hop 1, ...)


class StructuralMemory(nn.Module):
    """Wraps hop aggregation + hop scoring + weighting into m_struct (Sec 4.6)."""

    def __init__(self, config: ARPMConfig, hidden_dim: int, num_relations: int):
        super().__init__()
        self.config = config
        self.hop_scorer = HopScorer(hidden_dim, num_relations, config.max_hop)

    def forward(self, anchors: torch.Tensor, alpha: torch.Tensor, anchor_hop: torch.Tensor,
                anchor_is_local: torch.Tensor, relation_ids: torch.Tensor, query_repr: torch.Tensor,
                gumbel_tau: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (m_struct [B, d], beta [B, L], hop_has_data [B, L])."""
        cfg = self.config
        m_hop, hop_has_data = aggregate_hop_memories(
            anchors, alpha, anchor_hop, anchor_is_local, cfg.max_hop, cfg.structural_eps,
        )

        z = self.hop_scorer(query_repr, relation_ids)  # [B, L]
        z = z.masked_fill(~hop_has_data, float("-inf"))

        # A query with zero local anchors at every hop: fall back to a
        # uniform-over-all-hops score so downstream ops stay well-defined;
        # m_struct will still be (eps-stabilized) ~0 since m_hop is 0 there.
        no_data_rows = ~hop_has_data.any(dim=-1)
        if no_data_rows.any():
            z = torch.where(no_data_rows.unsqueeze(-1), torch.zeros_like(z), z)
            hop_has_data = hop_has_data | no_data_rows.unsqueeze(-1)

        mode = cfg.hop_selection_mode
        if mode == "soft":
            beta = torch.softmax(z, dim=-1)
        elif mode == "uniform":
            valid = hop_has_data.float()
            beta = valid / valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        elif mode == "gumbel_softmax":
            tau = gumbel_tau if gumbel_tau is not None else (cfg.gumbel_hop_temperature or 1.0)
            soft, hard = gumbel_softmax_top_k_st(z, tau, cfg.top_k_hop, mask=hop_has_data)
            beta = straight_through(hard, soft) if self.training else hard
        else:
            raise ValueError(f"Unknown hop_selection_mode: {mode}")

        beta = torch.nan_to_num(beta, nan=0.0)
        m_struct = torch.einsum("bl,bld->bd", beta, m_hop)
        return m_struct, beta, hop_has_data
