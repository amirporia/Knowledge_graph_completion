"""
Shared Gumbel-Softmax / Gumbel-Sigmoid straight-through (ST) machinery
(Proposal Sec 4.13), used by all three optional discrete extensions:

  4.13.1  discrete anchor retrieval        (Gumbel-Sigmoid keep/drop gate)
  4.13.2  discrete hop selection            (Gumbel-Softmax over hops)
  4.13.3  discrete prototype slot gating    (Gumbel-Sigmoid per-slot gate)

All three reuse the identical reparameterization: draw Gumbel noise, form
a temperature-tau relaxed sample, and route the forward pass through a hard
(one-hot / binary) sample while letting gradients flow through the soft
relaxation (`hard + soft - sg[soft]`).
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

_EPS = 1e-10


def sample_gumbel(shape, device, dtype=torch.float32) -> torch.Tensor:
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + _EPS) + _EPS)


def gumbel_softmax_st(logits: torch.Tensor, tau: float, mask: Optional[torch.Tensor] = None,
                       dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Straight-through Gumbel-Softmax over `dim`.

    Args:
        logits: unnormalized scores z_c (Sec 4.13's zeta_c).
        tau: temperature; smaller -> closer to one-hot.
        mask: optional bool tensor, same shape as logits, True = valid
              category (used e.g. to exclude padded hops/anchors). Masked
              positions get -inf logits before sampling.
        dim: category dimension.

    Returns:
        (y_soft, y_hard) both broadcastable to `logits.shape`; the caller
        typically uses `y_st = y_hard + y_soft - y_soft.detach()` at the
        call site if it needs the straight-through tensor itself, or uses
        `y_hard` for the forward value and `y_soft` only for logging.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))

    gumbel_noise = sample_gumbel(logits.shape, logits.device, logits.dtype)
    y_soft = F.softmax((logits + gumbel_noise) / tau, dim=dim)

    index = y_soft.argmax(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    if mask is not None:
        y_hard = y_hard * mask.float()

    return y_soft, y_hard


def gumbel_softmax_top_k_st(logits: torch.Tensor, tau: float, k: int,
                             mask: Optional[torch.Tensor] = None,
                             dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-top-k variant of `gumbel_softmax_st` (Sec 4.13.2's k_hop
    extension): perturb logits with Gumbel noise once, then take the top-k
    positions as a k-hot hard sample (Straight-through: the perturbed
    softmax over ALL categories is used as the soft relaxation).
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))

    gumbel_noise = sample_gumbel(logits.shape, logits.device, logits.dtype)
    perturbed = logits + gumbel_noise
    y_soft = F.softmax(perturbed / tau, dim=dim)

    k_eff = min(k, logits.shape[dim])
    topk_idx = perturbed.topk(k_eff, dim=dim).indices
    y_hard = torch.zeros_like(logits).scatter_(dim, topk_idx, 1.0)
    if mask is not None:
        y_hard = y_hard * mask.float()

    return y_soft, y_hard


def gumbel_sigmoid_st(logit: torch.Tensor, tau: float,
                       mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Straight-through Gumbel-Sigmoid (two-class Gumbel-Softmax special
    case) for independent binary keep/drop gates -- shared by 4.13.1 (anchor
    gating) and 4.13.3 (prototype slot gating).

    `logit` plays the role of `s_i` (4.13.1) or `zeta_k` (4.13.3): a single
    scalar per gate whose sigmoid is the "keep" probability pi.

    Returns (c_soft, c_hard), both in [0, 1] / {0, 1}, same shape as `logit`.
    """
    g1 = sample_gumbel(logit.shape, logit.device, logit.dtype)
    g0 = sample_gumbel(logit.shape, logit.device, logit.dtype)

    log_pi = F.logsigmoid(logit)
    log_1m_pi = F.logsigmoid(-logit)

    keep_score = (log_pi + g1) / tau
    drop_score = (log_1m_pi + g0) / tau
    c_soft = torch.sigmoid(keep_score - drop_score)
    c_hard = (keep_score > drop_score).float()

    if mask is not None:
        c_soft = c_soft * mask.float()
        c_hard = c_hard * mask.float()

    return c_soft, c_hard


def straight_through(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """y_ST = y_hard + y_soft - sg[y_soft] (Sec 4.13's ST estimator)."""
    return hard + soft - soft.detach()


def anneal_temperature(epoch: int, init_temp: float, min_temp: float, rate: float) -> float:
    """Exponential decay schedule shared across tau_sel, tau_hop, tau_proto
    unless a mechanism-specific override is set (Sec 4.13, "Shared
    implementation note"). Called once per epoch by the Trainer."""
    return max(min_temp, init_temp * (rate ** epoch))
