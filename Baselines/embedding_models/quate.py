import torch
import torch.nn.functional as F

from .base_model import KGEModel, xavier_embedding


def _hamilton_product(a, b):
    """Hamilton product of two batches of quaternions, each given as a tuple of
    (real, i, j, k) components of shape (..., d)."""
    a_r, a_i, a_j, a_k = a
    b_r, b_i, b_j, b_k = b
    r = a_r * b_r - a_i * b_i - a_j * b_j - a_k * b_k
    i = a_r * b_i + a_i * b_r + a_j * b_k - a_k * b_j
    j = a_r * b_j - a_i * b_k + a_j * b_r + a_k * b_i
    k = a_r * b_k + a_i * b_j - a_j * b_i + a_k * b_r
    return r, i, j, k


class QuatE(KGEModel):
    """Quaternion-embedding KGE (Zhang et al. 2019; used as the base representation in
    Li et al. 2023's quantum/quaternion-interaction completion method). Entities and
    relations are quaternions in H^d (4 real components each). Relations are
    normalized to unit quaternions so h ⊗ r_normalized acts as a rotation of h; the
    score is the quaternion inner product of the rotated head with the tail.
    """

    distance_based = False

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__(num_entities, num_relations, args)
        dim = args.embedding_dim
        self.dim = dim
        self.ent_emb = xavier_embedding(num_entities, 4 * dim)
        self.rel_emb = xavier_embedding(num_relations, 4 * dim)

    def _components(self, emb):
        d = self.dim
        return emb[..., :d], emb[..., d:2 * d], emb[..., 2 * d:3 * d], emb[..., 3 * d:]

    def _normalize_relation(self, r_idx):
        r_r, r_i, r_j, r_k = self._components(self.rel_emb[r_idx])
        norm = torch.sqrt(r_r ** 2 + r_i ** 2 + r_j ** 2 + r_k ** 2 + 1e-9)
        return r_r / norm, r_i / norm, r_j / norm, r_k / norm

    def _rotated_head(self, h_idx, r_idx):
        h_quat = self._components(self.ent_emb[h_idx])
        r_quat = self._normalize_relation(r_idx)
        return _hamilton_product(h_quat, r_quat)  # tuple of 4 tensors (B, dim)

    def score(self, h_idx, r_idx, t_idx):
        rh_r, rh_i, rh_j, rh_k = self._rotated_head(h_idx, r_idx)
        t_r, t_i, t_j, t_k = self._components(self.ent_emb[t_idx])
        return torch.sum(rh_r * t_r + rh_i * t_i + rh_j * t_j + rh_k * t_k, dim=-1)

    def score_all(self, h_idx, r_idx):
        rh_r, rh_i, rh_j, rh_k = self._rotated_head(h_idx, r_idx)
        query = torch.cat([rh_r, rh_i, rh_j, rh_k], dim=-1)      # (B, 4d)
        return query.mm(self.ent_emb.t())                        # (B, num_entities)
