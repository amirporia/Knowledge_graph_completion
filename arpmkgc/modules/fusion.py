"""
G_mem (Sec 4.8, semantic <-> structural memory fusion), G_q (Sec 4.9,
query <-> memory fusion), and G_lambda (Sec 4.10, adaptive memory gate).

The proposal is explicit that these are three conceptually distinct
modules (4.8: "Gmem =/= Gq at the conceptual level"); they are kept as
separate `nn.Module`s here rather than folded into one block, even though
they share a similar gated-residual shape.

When only one of {prototype memory, structural memory} is enabled (Sec 5 /
ablations A9, A10), `model.py` bypasses `MemoryFusion.forward` entirely and
feeds the sole active memory straight to `QueryMemoryFusion` -- gating
between a real vector and an artificial zero vector would not correspond to
anything in the proposal's formulas.
"""

import torch
import torch.nn as nn

from ..config import ARPMConfig


class MemoryFusion(nn.Module):
    """G_mem(m_p, m_struct) = m   (Sec 4.8)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, m_p: torch.Tensor, m_struct: torch.Tensor) -> torch.Tensor:
        z_m = torch.sigmoid(self.gate(torch.cat([m_p, m_struct], dim=-1)))
        return z_m * m_p + (1 - z_m) * m_struct


class QueryMemoryFusion(nn.Module):
    """G_q(q_p, m) = q_e   (Sec 4.9): q_e = LN(q_p + z_q * W_mem m)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.memory_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, q_p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        z_q = torch.sigmoid(self.gate(torch.cat([q_p, m], dim=-1)))
        return self.layer_norm(q_p + z_q * self.memory_proj(m))


class MemoryGate(nn.Module):
    """G_lambda(q, r): the query-dependent memory-trust gate (Sec 4.10).
    lambda_mem = sigma(G_lambda(q, r)), 0 <= lambda_mem <= 1.

    When `use_adaptive_memory_gate=False` (ablation A8), the model bypasses
    this module and uses `config.fixed_memory_gate` directly instead.
    """

    def __init__(self, hidden_dim: int, num_relations: int):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        r_emb = self.relation_embedding(relation_ids)
        logit = self.scorer(torch.cat([q, r_emb], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit)


def resolve_memory_gate(config: ARPMConfig, gate_module: MemoryGate, q: torch.Tensor,
                         relation_ids: torch.Tensor) -> torch.Tensor:
    if config.use_adaptive_memory_gate:
        return gate_module(q, relation_ids)
    return torch.full((q.size(0),), config.fixed_memory_gate, device=q.device, dtype=q.dtype)
