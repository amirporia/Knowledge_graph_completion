"""
Multi-prototype semantic memory (Sec 4.5) and query-prototype interaction
(Sec 4.7), with the optional adaptive prototype-count extension
(Sec 4.13.3 / ablation A13).

Pipeline within this module:
    ProtoGen:                  W_q, q          -> P_q = {p_1, ..., p_Kmax}
    SlotGate (optional, 4.13.3): q, r           -> per-slot keep gate omega_k
    QueryPrototypeInteraction: q, P_q, omega    -> q_p, m_p, gamma_k

Design note: per Sec 4.13.3, gating does not change how many prototypes are
*generated* (ProtoGen always fills K_max slots) -- it masks which slots
contribute to the query summary m_p (Sec 4.7) and, downstream, to S_p(t)
(Sec 4.10). This keeps a single fixed-shape computation graph regardless of
the realized K_q, as the proposal's "Optimization" paragraph requires.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .gumbel import gumbel_sigmoid_st, straight_through
from ..config import ARPMConfig


class PrototypeGenerator(nn.Module):
    """ProtoGen: query-conditioned attention that maps the weighted anchor
    set W_q to K (or K_max) prototype vectors (Sec 4.5)."""

    def __init__(self, hidden_dim: int, num_slots: int, weight_by_alpha: bool = True):
        super().__init__()
        self.num_slots = num_slots
        self.weight_by_alpha = weight_by_alpha
        self.slot_embedding = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, q: torch.Tensor, anchors: torch.Tensor, alpha: torch.Tensor,
                anchor_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """q: [B, d], anchors: [B, NA, d], alpha: [B, NA], anchor_mask: [B, NA]
        Returns (prototypes [B, K, d], attention rho [B, K, NA])."""
        b, na, d = anchors.shape
        k = self.num_slots

        slot_query = self.query_proj(q).unsqueeze(1) + self.slot_embedding.unsqueeze(0)  # [B, K, d]
        keys = self.key_proj(anchors)  # [B, NA, d]

        u = torch.bmm(slot_query, keys.transpose(1, 2)) / self.scale  # [B, K, NA] == u_ik

        mask = anchor_mask.unsqueeze(1).expand(b, k, na)
        u = u.masked_fill(~mask, float("-inf"))

        if self.weight_by_alpha:
            # rho_ik = alpha_i * exp(u_ik) / sum_j alpha_j * exp(u_jk)  (Sec 4.5)
            exp_u = torch.exp(u - u.amax(dim=-1, keepdim=True).clamp(min=-1e4))
            exp_u = exp_u.masked_fill(~mask, 0.0)
            weighted = exp_u * alpha.unsqueeze(1)
            denom = weighted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            rho = weighted / denom
        else:
            rho = torch.softmax(u, dim=-1)
            rho = torch.nan_to_num(rho, nan=0.0)

        prototypes = torch.bmm(rho, anchors)  # [B, K, d]
        return prototypes, rho


class SlotGate(nn.Module):
    """G_K(q, r, k): per-slot activation logit for the discrete prototype
    activation extension (Sec 4.13.3)."""

    def __init__(self, hidden_dim: int, num_relations: int, num_slots: int):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        self.slot_embedding = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_slots = num_slots

    def forward(self, q: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        b, d = q.shape
        k = self.num_slots
        q_exp = q.unsqueeze(1).expand(b, k, d)
        r_emb = self.relation_embedding(relation_ids).unsqueeze(1).expand(b, k, d)
        slot_emb = self.slot_embedding.unsqueeze(0).expand(b, k, d)
        feats = torch.cat([q_exp, r_emb, slot_emb], dim=-1)
        return self.scorer(feats).squeeze(-1)  # [B, K] == zeta_k


class QueryPrototypeInteraction(nn.Module):
    """Sec 4.7: q_p = q + CrossAttention(q, P_q); gamma_k = softmax_k(G_gamma(q, p_k));
    m_p = sum_k gamma_k p_k (restricted to active slots when slot gating is on)."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.summary_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q: torch.Tensor, prototypes: torch.Tensor,
                slot_mask: Optional[torch.Tensor] = None,
                slot_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q: [B, d], prototypes: [B, K, d].
        slot_mask: [B, K] bool, True = active slot (all-True if gating is off).
        slot_weight: [B, K] optional multiplicative weight (the ST gate value)
                     applied to gamma_k before renormalization, so gradients
                     flow through inactive-but-recently-active slots.
        Returns (q_p [B, d], m_p [B, d], gamma [B, K])."""
        b, k, d = prototypes.shape
        if slot_mask is None:
            slot_mask = torch.ones(b, k, dtype=torch.bool, device=prototypes.device)

        attn_mask = ~slot_mask  # True = ignore, per nn.MultiheadAttention convention
        # guard against a fully-masked row (would make softmax attention undefined)
        all_masked = attn_mask.all(dim=-1)
        safe_attn_mask = attn_mask.clone()
        safe_attn_mask[all_masked] = False

        context, _ = self.cross_attn(
            query=q.unsqueeze(1), key=prototypes, value=prototypes,
            key_padding_mask=safe_attn_mask,
        )
        context = context.squeeze(1)
        context = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(context), context)
        q_p = q + context

        q_exp = q.unsqueeze(1).expand(b, k, d)
        gamma_logits = self.summary_scorer(torch.cat([q_exp, prototypes], dim=-1)).squeeze(-1)
        gamma_logits = gamma_logits.masked_fill(~slot_mask, float("-inf"))
        gamma = torch.softmax(gamma_logits, dim=-1)
        gamma = torch.nan_to_num(gamma, nan=0.0)

        if slot_weight is not None:
            gamma = gamma * slot_weight
            denom = gamma.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            gamma = gamma / denom

        m_p = torch.bmm(gamma.unsqueeze(1), prototypes).squeeze(1)
        return q_p, m_p, gamma


def resolve_slot_gate(config: ARPMConfig, slot_gate_module: Optional[SlotGate], q: torch.Tensor,
                       relation_ids: torch.Tensor, gumbel_tau: Optional[float],
                       training: bool) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Returns (slot_mask, slot_weight, zeta) or (None, None, None) when the
    fixed-K core model is in use (Sec 4.5.1, no 4.13.3 extension)."""
    if config.prototype_gating_mode != "gumbel_sigmoid_slots":
        return None, None, None

    zeta = slot_gate_module(q, relation_ids)  # [B, K_max]
    tau = gumbel_tau if gumbel_tau is not None else (config.gumbel_proto_temperature or 1.0)
    omega_soft, omega_hard = gumbel_sigmoid_st(zeta, tau)
    omega_st = straight_through(omega_hard, omega_soft) if training else omega_hard

    slot_mask = omega_hard.bool()
    # never fully collapse to zero active prototypes: fall back to the single
    # highest-logit slot if the gate happens to drop every slot for a query.
    empty_rows = ~slot_mask.any(dim=-1)
    if empty_rows.any():
        fallback = torch.zeros_like(slot_mask)
        top1 = zeta.argmax(dim=-1)
        fallback[torch.arange(zeta.size(0)), top1] = True
        slot_mask = torch.where(empty_rows.unsqueeze(-1), fallback, slot_mask)
        omega_st = torch.where(empty_rows.unsqueeze(-1), fallback.float(), omega_st)

    return slot_mask, omega_st, zeta
