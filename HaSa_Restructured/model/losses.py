from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = {
    'hasa': 'hasa',                    # debiased + hardness-aware (false-negative correction)
    'hasa_hard_bias': 'hard_bias',     # hardness-aware only
    'hasa_wohard_bias': 'wohard',      # plain InfoNCE over in-batch + extra negatives
}


class HaSaLoss(nn.Module):
    """The three HaSa objectives of the original `loss.py`, vectorised and written in a
    numerically stable way (everything is divided by the positive score, so exp() only sees
    score differences). Mathematically identical to the original formulas:

        hasa      : -log( e^{s+} / ( e^{s+} + N/(1-tau) * relu( sum_j p_j e^{s_j} - e^{s+} - tau * sum_k q_k e^{f_k} ) ) )
        hard_bias : -log( e^{s+} / ( e^{s+} + N * ( sum_j p_j e^{s_j} - e^{s+} ) ) )
        wohard    : -log( e^{s+} / sum_j e^{s_j} )

    s_j are raw dot products (no normalisation, no temperature) of the (h, r) vector with the candidate
    tails [in-batch tails | in-batch heads | mined hard negatives]; s+ has `margin` subtracted;
    p are hardness weights from the sampler (p[i, i] == 1), q the false-negative weights.
    N is the embedding dim in the original code (`_, num_neg = v_h_t.shape`), see --neg-count-mode.
    """

    def __init__(self, method: str, tau: float, margin: float, plus: bool, neg_count_mode: str):
        super().__init__()
        self.variant = VARIANTS[method]
        self.tau = tau
        self.margin = margin
        self.plus = plus
        self.neg_count_mode = neg_count_mode

    def forward(self,
                s: torch.Tensor,
                v_h: torch.Tensor,
                v_t: torch.Tensor,
                v_hard: torch.Tensor,
                v_false: Optional[torch.Tensor],
                prob: Optional[torch.Tensor],
                f_prob: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        s, v_h, v_t, v_hard = s.float(), v_h.float(), v_t.float(), v_hard.float()
        a, dim = v_h.shape
        arange = torch.arange(a, device=s.device)

        candidates = torch.cat([v_t, v_h, v_hard], dim=0)
        scores = s @ candidates.t()                                  # a x (2a + a*hard)

        # positives get a margin (original: p_score -= diag(1))
        margin = torch.zeros_like(scores)
        margin[arange, arange] = self.margin
        scores = scores - margin

        num_neg = dim if self.neg_count_mode == 'dim' else candidates.size(0)
        pos = scores[arange, arange]
        rel = torch.exp(scores - pos.unsqueeze(1))                   # e^{s_j - s+}; rel[i, i] == 1

        if self.variant == 'wohard':
            neg = rel.sum(1) - 1.0
            loss = torch.log1p(neg.clamp(min=0)).mean()
        elif self.variant == 'hard_bias':
            neg = (rel * prob).sum(1) - 1.0
            loss = torch.log1p(num_neg * neg.clamp(min=0)).mean()
        else:
            k = v_false.size(0) // a
            f_scores = torch.einsum('ad,akd->ak', s, v_false.float().view(a, k, dim))
            false_term = self.tau * (torch.exp(f_scores - pos.unsqueeze(1)) * f_prob).sum(1)
            neg = ((rel * prob).sum(1) - 1.0 - false_term).clamp(min=0)
            loss = torch.log1p(num_neg / (1.0 - self.tau) * neg).mean()

        if self.plus:
            loss = loss + F.cross_entropy(scores[:, :a].t(), arange)

        return loss, scores
