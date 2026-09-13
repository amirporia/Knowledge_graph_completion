import torch

from .base_model import KGEModel, xavier_embedding


class DistMult(KGEModel):
    """Yang et al. 2015 — Embedding Entities and Relations for Learning and Inference
    in Knowledge Bases. score(h, r, t) = sum(h * r * t).
    """

    distance_based = False

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__(num_entities, num_relations, args)
        dim = args.embedding_dim
        self.ent_emb = xavier_embedding(num_entities, dim)
        self.rel_emb = xavier_embedding(num_relations, dim)

    def score(self, h_idx, r_idx, t_idx):
        h = self.ent_emb[h_idx]
        r = self.rel_emb[r_idx]
        t = self.ent_emb[t_idx]
        return torch.sum(h * r * t, dim=-1)

    def score_all(self, h_idx, r_idx):
        query = self.ent_emb[h_idx] * self.rel_emb[r_idx]       # (B, d)
        return query.mm(self.ent_emb.t())                       # (B, num_entities)
