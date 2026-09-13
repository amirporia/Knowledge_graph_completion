import torch
import torch.nn as nn

from .base_model import KGEModel, xavier_embedding


class TransE(KGEModel):
    """Bordes et al. 2013 — Translating Embeddings for Modeling Multi-relational Data.
    score(h, r, t) = -|| h + r - t ||_p  (higher = more plausible)
    """

    distance_based = True

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__(num_entities, num_relations, args)
        dim = args.embedding_dim
        self.p_norm = getattr(args, 'p_norm', 1)
        self.ent_emb = xavier_embedding(num_entities, dim)
        self.rel_emb = xavier_embedding(num_relations, dim)

    def score(self, h_idx, r_idx, t_idx):
        h = self.ent_emb[h_idx]
        r = self.rel_emb[r_idx]
        t = self.ent_emb[t_idx]
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def score_all(self, h_idx, r_idx):
        hr = self.ent_emb[h_idx] + self.rel_emb[r_idx]          # (B, d)
        # cdist computes pairwise p-norm distances without materializing (B, N, d)
        dist = torch.cdist(hr.unsqueeze(0), self.ent_emb.unsqueeze(0), p=self.p_norm).squeeze(0)
        return -dist
