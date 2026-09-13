import math

import torch
import torch.nn as nn

from .base_model import KGEModel


class RotatE(KGEModel):
    """Sun et al. 2019 — RotatE: Knowledge Graph Embedding by Relational Rotation in
    Complex Space. Entities in C^d, relations are pure phases (unit-modulus rotations).
    score(h, r, t) = -|| h ∘ r - t ||   where r = e^{i*phase}, ∘ is elementwise complex mult.

    Verified directly against the paper: the L1-over-complex-modulus distance
    (footnote 2: "we use L1-norm for all distance-based models"), the embedding_range
    = (gamma+2)/dim initialization scheme and phase-from-unconstrained-weight trick,
    and the self-adversarial loss (Eq. 5-6) all match exactly -- Eq. 6's loss, worked
    through algebraically, is exactly base_model.py's shared self-adversarial loss
    with distance_based=True and margin=gamma, so this model's training loop is a
    literal implementation of the paper's Eq. 6, not just "inspired by" it.
    """

    distance_based = True

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__(num_entities, num_relations, args)
        dim = args.embedding_dim
        self.dim = dim
        gamma = getattr(args, 'margin', 12.0)
        self.margin = gamma
        # RotatE-style bounded init: entities uniform in [-range, range], relation
        # phases uniform in [-pi, pi] via a learned "phase weight" scaled by range/pi.
        self.embedding_range = (gamma + 2.0) / dim
        self.ent_emb = nn.Parameter(torch.empty(num_entities, 2 * dim).uniform_(
            -self.embedding_range, self.embedding_range))
        self.rel_phase = nn.Parameter(torch.empty(num_relations, dim).uniform_(
            -self.embedding_range, self.embedding_range))

    def _rotation(self, r_idx):
        phase = self.rel_phase[r_idx] / (self.embedding_range / math.pi)
        return torch.cos(phase), torch.sin(phase)

    def score(self, h_idx, r_idx, t_idx):
        h_re, h_im = self.ent_emb[h_idx, :self.dim], self.ent_emb[h_idx, self.dim:]
        t_re, t_im = self.ent_emb[t_idx, :self.dim], self.ent_emb[t_idx, self.dim:]
        r_re, r_im = self._rotation(r_idx)

        re_diff = h_re * r_re - h_im * r_im - t_re
        im_diff = h_re * r_im + h_im * r_re - t_im
        dist = torch.stack([re_diff, im_diff], dim=0).norm(dim=0).sum(dim=-1)
        return -dist

    def score_all(self, h_idx, r_idx):
        """Chunked broadcast implementation. Unlike TransE's L1 distance (which cdist
        computes exactly via a single fused call), RotatE's official distance is a
        *sum of per-dimension complex moduli* — not a plain Lp norm over the
        concatenated real vector — so it can't be expressed as a single cdist call.
        We instead chunk over candidate entities and broadcast, matching `score()`
        exactly; `args.eval_entity_chunk` controls the memory/speed trade-off.
        """
        h_re, h_im = self.ent_emb[h_idx, :self.dim], self.ent_emb[h_idx, self.dim:]
        r_re, r_im = self._rotation(r_idx)
        q_re = h_re * r_re - h_im * r_im          # (B, d)
        q_im = h_re * r_im + h_im * r_re          # (B, d)

        B = q_re.size(0)
        out = torch.empty(B, self.num_entities, device=q_re.device)
        for start in range(0, self.num_entities, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, self.num_entities)
            e_re = self.ent_emb[start:end, :self.dim]           # (C, d)
            e_im = self.ent_emb[start:end, self.dim:]

            re_diff = q_re.unsqueeze(1) - e_re.unsqueeze(0)     # (B, C, d)
            im_diff = q_im.unsqueeze(1) - e_im.unsqueeze(0)
            modulus = torch.sqrt(re_diff ** 2 + im_diff ** 2 + 1e-12)
            out[:, start:end] = -modulus.sum(dim=-1)
        return out
