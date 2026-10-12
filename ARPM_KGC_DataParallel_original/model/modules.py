"""Neural building blocks for ARPM-KGC v2.

  - ProtoGen       -> K query-conditioned prototypes from the weighted anchor set
  - HopScorer      -> z_l = G_hop(q, l)
  - diversity_loss / prototype_diversity

The learned memory gate (G_lambda) was removed: the weights of the three score components
(S_hrta, S_p, S_struct) are now plain constants chosen on the validation set
(see evaluation/evaluate.py::resolve_coefs).
"""
import torch
import torch.nn as nn


class ProtoGen(nn.Module):
    """u_ik = a_i^T W_k q / T ; rho_ik = alpha_i exp(u_ik) / sum_j alpha_j exp(u_jk) ; p_k = sum_i rho_ik a_i

    W_k is initialised to identity + small noise, so every slot starts as a (slightly different)
    cosine-attention instead of a random projection of q."""

    def __init__(self, hidden_size: int, num_prototypes: int, temperature: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        w = torch.eye(hidden_size).unsqueeze(0).repeat(num_prototypes, 1, 1)
        w = w + 0.01 * torch.randn_like(w)
        self.bilinear = nn.Parameter(w)

    def forward(self, cand_emb, query, alpha, valid_mask):
        """cand_emb (B,N,d) zero on invalid slots; query (B,d); alpha (B,N); valid_mask (B,N)
        -> prototypes (B,K,d)"""
        wq = torch.einsum('kde,be->bkd', self.bilinear, query)
        u = torch.einsum('bnd,bkd->bnk', cand_emb, wq) / self.temperature
        u = u.masked_fill(~valid_mask.unsqueeze(-1), -1e4)
        u = u - u.amax(dim=1, keepdim=True)

        weighted = alpha.unsqueeze(-1) * torch.exp(u) * valid_mask.unsqueeze(-1)
        denom = weighted.sum(dim=1, keepdim=True).clamp(min=1e-12)
        rho = weighted / denom
        return torch.einsum('bnk,bnd->bkd', rho, cand_emb)


class HopScorer(nn.Module):
    def __init__(self, hidden_size: int, num_hops: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_hops)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.linear(query)


def diversity_loss(cand_emb, alpha, valid_mask):
    """L_div = (1/Z) sum_{i!=j} alpha_i alpha_j cos(a_i,a_j). Returns (B,) (NOT reduced) so that
    nn.DataParallel can gather it."""
    sim = torch.einsum('bnd,bmd->bnm', cand_emb, cand_emb)
    outer = alpha.unsqueeze(2) * alpha.unsqueeze(1)
    n = alpha.size(1)
    eye = torch.eye(n, device=alpha.device, dtype=torch.bool).unsqueeze(0)
    outer = outer.masked_fill(eye, 0.0)
    weighted_sum = (outer * sim).sum(dim=(1, 2))
    n_valid = valid_mask.sum(dim=-1).float()
    z = (n_valid * (n_valid - 1)).clamp(min=1.0)
    return weighted_sum / z


def prototype_diversity(prototypes: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine between the K (L2-normalised or zero) prototypes -> (B,).
    Minimising it forces the K slots to specialise instead of collapsing."""
    k = prototypes.size(1)
    if k < 2:
        return prototypes.new_zeros(prototypes.size(0))
    sim = torch.einsum('bkd,bmd->bkm', prototypes, prototypes)
    eye = torch.eye(k, device=prototypes.device, dtype=torch.bool).unsqueeze(0)
    return sim.masked_fill(eye, 0.0).sum(dim=(1, 2)) / (k * (k - 1))
