"""Neural building blocks for ARPM-KGC.

  - ProtoGen       -> (Multi-Prototype Semantic Memory), P_q = ProtoGen(W_q, q)
  - HopScorer      -> (Adaptive Structural Memory), z_l = G_hop(q, l)
  - MemoryGate     -> (Adaptive Memory-Aware Tail Prediction), [lambda_p,lambda_s]=G_lambda(q)
  - PrototypeActivationScorer -> zeta_k = G_K(q, r, k)  (optional, A13)
  - diversity_loss -> L_div
  - gumbel_sigmoid_gate      -> (A11, anchor keep/drop gate c_i)
  - gumbel_softmax_topk      -> (A12, hop selection beta_l)
  - gumbel_sigmoid_slot_gate -> (A13, prototype slot gate omega_k)
"""
from typing import Optional

import torch
import torch.nn as nn

_EPS = 1e-10
_NEG_INF = -1e4  # finite sentinel, so an all-masked row still


# produces a finite (if meaningless) softmax instead of NaN;
# callers that need "no valid options" to mean exactly zero
# downstream (e.g. hop selection when a query has no local
# anchor at any hop) handle that explicitly at a higher level.


class ProtoGen(nn.Module):
    """maps the weighted anchor set W_q = {(a_i, alpha_i)} to K semantic
    prototypes via query-conditioned attention.

    u_ik = a_i^T W_k q / sqrt(d)                         (per-prototype bilinear score)
    rho_ik = alpha_i * exp(u_ik) / sum_j alpha_j * exp(u_jk)
    p_k = sum_i rho_ik * a_i
    """

    def __init__(self, hidden_size: int, num_prototypes: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.scale = hidden_size ** 0.5

        # W_k, one d x d bilinear matrix per prototype slot.
        self.bilinear = nn.Parameter(torch.empty(num_prototypes, hidden_size, hidden_size))
        for k in range(num_prototypes):
            nn.init.xavier_uniform_(self.bilinear[k])

    def forward(self, cand_emb: torch.Tensor, query: torch.Tensor,
                alpha: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cand_emb: (B, N, d) anchor embeddings (already zeroed on invalid slots)
            query: (B, d) query embedding q
            alpha: (B, N) query-conditioned anchor weights (already zero on invalid slots)
            valid_mask: (B, N) bool, True where a real candidate anchor exists

        Returns:
            prototypes: (B, K, d)
        """
        # u_ik = a_i^T W_k q / sqrt(d)
        u = torch.einsum('bnd,kde,be->bnk', cand_emb, self.bilinear, query) / self.scale
        u = u.masked_fill(~valid_mask.unsqueeze(-1), -1e4)

        u_max = u.amax(dim=1, keepdim=True)
        exp_u = torch.exp(u - u_max) * valid_mask.unsqueeze(-1)

        weighted = alpha.unsqueeze(-1) * exp_u  # (B, N, K)
        denom = weighted.sum(dim=1, keepdim=True).clamp(min=1e-12)  # (B, 1, K)
        rho = weighted / denom  # (B, N, K)

        prototypes = torch.einsum('bnk,bnd->bkd', rho, cand_emb)
        return prototypes


class HopScorer(nn.Module):
    """z_l = G_hop(q, l), implemented as a single linear layer whose
    l-th output channel is z_l -- i.e. G_hop(q, l) = W[l, :] . q + b[l].
    Constructed with num_hops+1 channels (l=0 is the same-head/same-relation
    local category, graph distance 0; l=1..num_hops are increasing graph
    distances)"""

    def __init__(self, hidden_size: int, num_hops: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_hops)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.linear(query)  # (B, L)


class MemoryGate(nn.Module):
    """[lambda_p, lambda_s] = G_lambda(q), independent per-source gates
    in [0, 1] via a linear layer + sigmoid."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(query))  # (B, 2) -> [:,0]=lambda_p, [:,1]=lambda_s


class PrototypeActivationScorer(nn.Module):
    """(A13, optional): zeta_k = G_K(q, r, k), implemented the same way
    as HopScorer -- one linear output channel per prototype slot."""

    def __init__(self, hidden_size: int, num_prototypes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_prototypes)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.linear(query)  # (B, K)


def diversity_loss(cand_emb: torch.Tensor, alpha: torch.Tensor,
                   valid_mask: torch.Tensor) -> torch.Tensor:
    """L_div = (1/Z) * sum_{i!=j} alpha_i alpha_j sim(a_i, a_j),
    Z = N_A * (N_A - 1), averaged over the batch.

    cand_emb is expected to already be L2-normalized (as produced by the shared
    E_0 encoder), so a_i . a_j IS cosine similarity.
    """
    sim = torch.einsum('bnd,bmd->bnm', cand_emb, cand_emb)  # (B, N, N)
    outer = alpha.unsqueeze(2) * alpha.unsqueeze(1)  # (B, N, N)

    n = alpha.size(1)
    eye = torch.eye(n, device=alpha.device, dtype=torch.bool).unsqueeze(0)
    outer = outer.masked_fill(eye, 0.0)

    weighted_sum = (outer * sim).sum(dim=(1, 2))  # (B,)

    n_valid = valid_mask.sum(dim=-1).float()
    z = (n_valid * (n_valid - 1)).clamp(min=1.0)

    return (weighted_sum / z).mean()


# ---------------------------------------------------------------------------
# Optional discrete (Gumbel) extensions -- ablations A11-A13.
# All three use the Gumbel-Softmax / straight-through (ST) estimator: hard in
# the forward pass, soft (differentiable) in the backward pass.
# ---------------------------------------------------------------------------

def gumbel_sigmoid_gate(relevance_logit: torch.Tensor, tau: float, training: bool) -> torch.Tensor:
    """(A11): binary keep/drop gate c_i for anchor retrieval.

    pi_i = sigma(s_i) reuses the raw relevance score s_i (NOT divided by tau_r).
    c_i(tau_sel) = sigma( ((log pi_i + g_i^1) - (log(1-pi_i) + g_i^0)) / tau_sel )
    c_i^hard = 1[ log pi_i + g_i^1 > log(1-pi_i) + g_i^0 ]
    c_i^ST = c_i^hard + c_i(tau_sel) - sg[c_i(tau_sel)]

    `relevance_logit` may contain -inf entries (masked/invalid candidates);
    these naturally map to pi=0 -> hard gate 0, matching candidate_valid_mask.
    """
    pi = torch.sigmoid(relevance_logit)

    if training:
        u1 = torch.rand_like(pi).clamp(_EPS, 1 - _EPS)
        u0 = torch.rand_like(pi).clamp(_EPS, 1 - _EPS)
        g1 = -torch.log(-torch.log(u1))
        g0 = -torch.log(-torch.log(u0))
    else:
        g1 = torch.zeros_like(pi)
        g0 = torch.zeros_like(pi)

    log_pi = torch.log(pi.clamp(min=_EPS))
    log_1mpi = torch.log((1 - pi).clamp(min=_EPS))

    a = log_pi + g1
    b = log_1mpi + g0
    soft = torch.sigmoid((a - b) / tau)
    hard = (a > b).float()

    return hard + soft - soft.detach()


def gumbel_softmax_topk(logits: torch.Tensor, tau: float, k: int,
                        training: bool, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(A12): hop selection. k=1 is the exact boxed ST Gumbel-Softmax
    (beta^hard = one-hot(argmax_l(z_l+g_l))); k>1 is the "Gumbel-top-k" variant
    that lets more than one hop distance stay active.
    """
    if valid_mask is not None:
        logits = logits.masked_fill(~valid_mask, _NEG_INF)

    if training:
        u = torch.rand_like(logits).clamp(_EPS, 1 - _EPS)
        g = -torch.log(-torch.log(u))
        perturbed = logits + g
    else:
        perturbed = logits

    soft = torch.softmax(perturbed / tau, dim=-1)

    topk = min(k, logits.size(-1))
    _, top_idx = perturbed.topk(topk, dim=-1)
    hard = torch.zeros_like(logits).scatter_(-1, top_idx, 1.0)

    if valid_mask is not None:
        # If fewer than `k` slots are valid (e.g. only 1 hop has any local
        # anchor but k_hop=2), top-k is forced to also pick a masked slot to
        # fill out k -- strip any such invalid pick rather than letting it
        # receive real (if small) weight; this may leave fewer than k active
        # hops for that query, which is the correct, conservative outcome.
        hard = hard * valid_mask.to(hard.dtype)

    return hard + soft - soft.detach()


def gumbel_sigmoid_slot_gate(zeta: torch.Tensor, tau: float, training: bool) -> torch.Tensor:
    """(A13): per-slot prototype activation gate omega_k.

    omega_k(tau_proto) = sigma( ((zeta_k+g_k^1) - (-zeta_k+g_k^0)) / tau_proto )
    omega_k^hard = 1[zeta_k > 0]           (deterministic threshold, as specified)
    omega_k^ST = omega_k^hard + omega_k(tau_proto) - sg[omega_k(tau_proto)]
    """
    if training:
        u1 = torch.rand_like(zeta).clamp(_EPS, 1 - _EPS)
        u0 = torch.rand_like(zeta).clamp(_EPS, 1 - _EPS)
        g1 = -torch.log(-torch.log(u1))
        g0 = -torch.log(-torch.log(u0))
    else:
        g1 = torch.zeros_like(zeta)
        g0 = torch.zeros_like(zeta)

    soft = torch.sigmoid(((zeta + g1) - (-zeta + g0)) / tau)
    hard = (zeta > 0).float()

    return hard + soft - soft.detach()
