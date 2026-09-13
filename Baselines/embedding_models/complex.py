import torch

from .base_model import KGEModel, xavier_embedding


class ComplEx(KGEModel):
    """Trouillon & Bouchard 2016 — Complex Embeddings for Simple Link Prediction.
    Entities/relations live in C^d (stored as two real halves of size d each).
    score(h, r, t) = Re(<h, r, conj(t)>)
                   = sum(h_re*r_re*t_re + h_re*r_im*t_im + h_im*r_re*t_im - h_im*r_im*t_re)

    Verified against the paper's own Eq. 11 (its notation: r=relation w_r, h=subject
    e_s, t=object e_o): expanding Re(w_r * e_s * conj(e_o)) term-by-term gives
    r_re*h_re*t_re + r_re*h_im*t_im + r_im*h_re*t_im - r_im*h_im*t_re, which looks
    different from the formula above at a glance (the cross terms pair "re" and "im"
    components from opposite operands), but is algebraically identical: e.g.
    h_re*r_im*t_im (this file's term) and r_re*h_im*t_im (paper's term) are different
    individual products, but this file's two cross terms sum to exactly the same
    total as the paper's two cross terms, since scalar multiplication is commutative
    (h_re*r_im + h_im*r_re == r_im*h_re + r_re*h_im). Checked both symbolically and
    numerically. score_all()'s q_re/q_im construction is the corresponding
    matmul-friendly factoring of the same expression.

    One thing worth setting explicitly if replicating the paper closely: it trains
    with L2 regularization (Eq. 12, lambda||Theta||^2 over both real and imaginary
    parts) via plain logistic loss over positive+corrupted triples (no
    self-adversarial weighting, unlike this repo's shared loss — see base_model.py).
    The L2 term maps directly to this repo's --weight-decay (0.0 by default; the
    paper's grid search found up to +0.05 MRR from tuning it away from 0 on
    FB15K/WN18), not something separate you need to add by hand.
    """

    distance_based = False

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__(num_entities, num_relations, args)
        dim = args.embedding_dim  # this is the *complex* dim; real storage is 2*dim
        self.dim = dim
        self.ent_emb = xavier_embedding(num_entities, 2 * dim)
        self.rel_emb = xavier_embedding(num_relations, 2 * dim)

    @staticmethod
    def _split(x, dim):
        return x[..., :dim], x[..., dim:]

    def score(self, h_idx, r_idx, t_idx):
        h_re, h_im = self._split(self.ent_emb[h_idx], self.dim)
        r_re, r_im = self._split(self.rel_emb[r_idx], self.dim)
        t_re, t_im = self._split(self.ent_emb[t_idx], self.dim)
        return torch.sum(
            h_re * r_re * t_re + h_re * r_im * t_im +
            h_im * r_re * t_im - h_im * r_im * t_re, dim=-1
        )

    def score_all(self, h_idx, r_idx):
        h_re, h_im = self._split(self.ent_emb[h_idx], self.dim)
        r_re, r_im = self._split(self.rel_emb[r_idx], self.dim)
        # query components such that score_all = q_re @ E_re^T + q_im @ E_im^T
        q_re = h_re * r_re - h_im * r_im
        q_im = h_re * r_im + h_im * r_re
        e_re, e_im = self._split(self.ent_emb, self.dim)
        return q_re.mm(e_re.t()) + q_im.mm(e_im.t())
