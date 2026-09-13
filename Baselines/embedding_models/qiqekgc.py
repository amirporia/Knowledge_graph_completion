"""QIQE-KGC — Knowledge graph completion method based on quantum embedding and
quaternion interaction enhancement (Li, Zhang, Jin, Gao, Zhu, Liang & Ma, 2023,
Information Sciences 648). https://doi.org/10.1016/j.ins.2023.119548

This is the paper actually cited for the "QuatE" baseline in the original request
("Li, L. et al. 2023b. Knowledge graph completion method based on quantum embedding
and quaternion interaction enhancement") -- it is *not* the original Zhang et al.
2019 QuatE (that's `quate.py`; it's also one of QIQE-KGC's own baselines, so it's
kept as a separate, correct, standalone model). Use `--model qiqekgc` for the
baseline matching your citation list; `--model quate` remains available for plain
QuatE if that's what you actually want for that row.

QIQE-KGC combines two modules whose scores/losses are weighted together (Eq. 18-19):

score(h,r,t) = alpha * score_quantum(h,r,t) + beta * score_quaternion(h,r,t)   (Eq. 19)

**Quaternion module** (Section 3.2, Eq. 9-16) -- implemented in full. Q_h, Q_t are
entity quaternions; W_r, W_{r,h}, W_{r,t} are per-relation quaternions:
    Q_{r,h} = Q_h (x) normalize(W_{r,h})                                    (Eq. 14)
    Q_{r,t} = Q_t (x) normalize(W_{r,t})                                    (Eq. 15)
    score_quaternion = (Q_{r,h} (x) normalize(W_r)) . Q_{r,t}               (Eq. 16)
where (x) is the Hamilton product and . is the quaternion inner product. Unlike
plain QuatE (which only ever rotates the head), QIQE-KGC additionally rotates the
tail by its own relation-specific quaternion -- the paper's proposed fix for
entities of very different "attribute types" (e.g. a person head entity and a club
tail entity) sharing one relation.

**Quantum embedding module** (Section 3.1, Eq. 1) -- score implemented in full;
regularization losses now implemented against the confirmed source formulas (see
below), with one piece still genuinely out of scope.
    E_r = concat(v_h, v_t) in R^(2d)      (entity-pair representation of a triple)
    score_quantum = -|| (1 - E_r) . E_r ||^2                                (Eq. 1)
Note on the minus sign: the paper states the score should be *higher* when a triple
better satisfies the quantum-logic orthogonality condition, but Eq. 1 as written is
a norm that *decreases* toward that condition. Negating it is the only reading
consistent with Eq. 19 combining it additively with score_quaternion (higher =
better) for ranking; flag if you intended a different reading.

**Regularization losses (ELoss + MLoss), corrected against Garg et al. 2019
(E2R, NeurIPS) -- QIQE-KGC's own cited source for this module.** An earlier version
of this file guessed at Eq. 23-32 from the extracted PDF text and got one term
wrong and dropped MLoss entirely as unreconstructable; having E2R's original
equations (its Eq. 1-4) resolves this:
  - L_entity (Eq. 23 = E2R Eq. 1): unchanged, ||z_h||^2 + ||z_t||^2, pushing each
    entity's "imaginary" half toward zero.
  - L_bound (Eq. 24 = E2R Eq. 2): unchanged.
  - L_gamma (Eq. 25 = E2R Eq. 3): **corrected**. This is a regularizer keeping the
    relation's indicator vector close to *binary* (0/1) -- E2R's actual formula is
    ||gamma ⊙ (1-gamma)||^2 (minimized exactly when every entry is 0 or 1), not the
    quaternion-conjugate-style "sum of 4th powers" guessed earlier.
  - MLoss (Eq. 32 = E2R Eq. 4, its *binary-relation* case): **now implemented**,
    not dropped. All four of its terms are meaningful in E2R itself (which allows
    entity-pair embeddings to be learned as free parameters, only *softly* tied
    back to the individual entity embeddings via this loss). This repo instead
    builds E_r deterministically by concatenating the entity embeddings (simpler,
    and consistent with how every other baseline here scores triples), which makes
    two of the four terms provably redundant with L_entity above (not ambiguous —
    algebraically identical given this construction) rather than something to guess
    at; they're included anyway since the paper's design reinforces that
    regularization from two loss terms rather than one.
  - LLoss (Eq. 27-31, logic inclusion/conjunction/disjunction/negation between
    *different relations*) is the one piece still not implemented: it requires a
    T-box (a table of known logical relationships between relations, e.g.
    hypernym(Dentist, Doctor)) that E2R's own experiments only had available for an
    ontology benchmark (LUBM) with an explicit T-box -- standard KGC triple datasets
    (FB15k-237, WN18RR, ...) don't ship with one, and there's no dataset-agnostic way
    to mine it. Still out of scope; if you have T-box annotations for your dataset
    (or want to skip this and use LUBM-style data instead) I can add it.
  Practical note: as before, since these are regularizers rather than a
  discriminative pos-vs-negative training signal, score_quantum is trained through
  this repo's shared self-adversarial negative-sampling loss (base_model.py) with
  ELoss+MLoss added as a small auxiliary term (`args.qiqe_reg_weight`) -- not the
  paper's literal ELoss+LLoss+MLoss-only training regime.
"""

import torch
import torch.nn as nn

from .base_model import KGEModel, xavier_embedding
from .quate import _hamilton_product


class QIQEKGC(KGEModel):
    distance_based = False

    def __init__(self, num_entities: int, num_relations: int, args):
        super().__init__(num_entities, num_relations, args)
        d = args.embedding_dim
        self.d = d
        self.alpha = getattr(args, 'qiqe_alpha', 0.5)
        self.beta = getattr(args, 'qiqe_beta', 0.5)
        self.reg_weight = getattr(args, 'qiqe_reg_weight', 0.01)

        # --- quaternion module (Section 3.2) ---
        self.q_ent = xavier_embedding(num_entities, 4 * d)          # Q_h / Q_t
        self.q_rel_w = xavier_embedding(num_relations, 4 * d)       # W_r  (Eq. 10)
        self.q_rel_wh = xavier_embedding(num_relations, 4 * d)      # W_{r,h} (Eq. 12)
        self.q_rel_wt = xavier_embedding(num_relations, 4 * d)      # W_{r,t} (Eq. 13)

        # --- task-oriented quantum embedding module (Section 3.1) ---
        self.quantum_ent = xavier_embedding(num_entities, 2 * d)          # e_i = (v_i; z_i)
        self.quantum_rel_gamma = xavier_embedding(num_relations, 2 * d)   # gamma_r

    # ---- quaternion module --------------------------------------------------
    def _q_components(self, emb):
        d = self.d
        return emb[..., :d], emb[..., d:2 * d], emb[..., 2 * d:3 * d], emb[..., 3 * d:]

    def _q_normalize_components(self, emb):
        r, i, j, k = self._q_components(emb)
        norm = torch.sqrt(r ** 2 + i ** 2 + j ** 2 + k ** 2 + 1e-9)
        return r / norm, i / norm, j / norm, k / norm

    def _quaternion_score(self, h_idx, r_idx, t_idx):
        q_h = self._q_components(self.q_ent[h_idx])
        q_t = self._q_components(self.q_ent[t_idx])
        w_rh = self._q_normalize_components(self.q_rel_wh[r_idx])
        w_rt = self._q_normalize_components(self.q_rel_wt[r_idx])
        w_r = self._q_normalize_components(self.q_rel_w[r_idx])

        q_rh = _hamilton_product(q_h, w_rh)          # Q_{r,h}, Eq. 14
        q_rt = _hamilton_product(q_t, w_rt)          # Q_{r,t}, Eq. 15
        q_rh_rot = _hamilton_product(q_rh, w_r)      # (Q_{r,h} (x) W_r), Eq. 16

        rh_r, rh_i, rh_j, rh_k = q_rh_rot
        rt_r, rt_i, rt_j, rt_k = q_rt
        return torch.sum(rh_r * rt_r + rh_i * rt_i + rh_j * rt_j + rh_k * rt_k, dim=-1)

    # ---- quantum embedding module -------------------------------------------
    def _quantum_score(self, h_idx, r_idx, t_idx):
        d = self.d
        v_h = self.quantum_ent[h_idx, :d]
        v_t = self.quantum_ent[t_idx, :d]
        e_r = torch.cat([v_h, v_t], dim=-1)                          # E_r (Eq. entity-pair rep)
        f_quantum = torch.sum(((1.0 - e_r) * e_r) ** 2, dim=-1)      # Eq. 1
        return -f_quantum                                            # negated, see module docstring

    # ---- combined QIQE-KGC score (Eq. 19) ------------------------------------
    def score(self, h_idx, r_idx, t_idx):
        return (self.alpha * self._quantum_score(h_idx, r_idx, t_idx) +
                self.beta * self._quaternion_score(h_idx, r_idx, t_idx))

    def score_all(self, h_idx, r_idx):
        B = h_idx.size(0)
        device = h_idx.device

        # --- quaternion part: rotate the query once; W_{r,t} is per-relation so
        # candidates need a fresh rotation per distinct relation in the batch (eval
        # batches typically span few unique relations; this loops over them). ---
        q_h = self._q_components(self.q_ent[h_idx])
        w_rh = self._q_normalize_components(self.q_rel_wh[r_idx])
        w_r = self._q_normalize_components(self.q_rel_w[r_idx])
        q_rh = _hamilton_product(q_h, w_rh)
        query = _hamilton_product(q_rh, w_r)                          # (4 x (B, d))

        ent_components = self._q_components(self.q_ent)               # all entities, (4 x (N, d))
        quaternion_scores = torch.empty(B, self.num_entities, device=device)
        for rel in torch.unique(r_idx).tolist():
            rows = (r_idx == rel).nonzero(as_tuple=True)[0]
            w_rt = self._q_normalize_components(
                self.q_rel_wt[torch.tensor([rel], device=device)])
            cand = _hamilton_product(
                ent_components, tuple(c.expand(self.num_entities, -1) for c in w_rt))
            cand_stack = torch.stack(cand, dim=0)                     # (4, N, d)
            q_rows = torch.stack([c[rows] for c in query], dim=0)     # (4, n_rows, d)
            quaternion_scores[rows] = torch.einsum('cnd,cmd->nm', q_rows, cand_stack)

        # --- quantum part: chunked over candidate entities (exact, not approximated) ---
        d = self.d
        v_h = self.quantum_ent[h_idx, :d]                             # (B, d)
        v_all_t = self.quantum_ent[:, :d]                             # (N, d)
        quantum_scores = torch.empty(B, self.num_entities, device=device)
        for start in range(0, self.num_entities, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, self.num_entities)
            v_t_chunk = v_all_t[start:end]
            e_r = torch.cat([
                v_h.unsqueeze(1).expand(-1, end - start, -1),
                v_t_chunk.unsqueeze(0).expand(B, -1, -1),
            ], dim=-1)
            f_quantum = torch.sum(((1.0 - e_r) * e_r) ** 2, dim=-1)
            quantum_scores[:, start:end] = -f_quantum

        return self.alpha * quantum_scores + self.beta * quaternion_scores

    # ---- Eq. 23-26 & 32 regularization, corrected against E2R (Garg et al. 2019) ---
    def regularization_loss(self, h_idx: torch.Tensor, r_idx: torch.Tensor,
                            t_idx: torch.Tensor) -> torch.Tensor:
        d = self.d
        e_h = self.quantum_ent[h_idx]                     # (B, 2d)
        e_t = self.quantum_ent[t_idx]
        z_h, z_t = e_h[:, d:], e_t[:, d:]
        l_entity = (z_h ** 2).sum(dim=-1) + (z_t ** 2).sum(dim=-1)                   # Eq. 23 = E2R Eq. 1

        gamma = self.quantum_rel_gamma[r_idx]                                        # (B, 2d)
        sum_first_half = gamma[:, :d].sum(dim=-1)
        sum_second_half = gamma[:, d:].sum(dim=-1)
        l_bound = (torch.clamp(sum_second_half - 1.0, max=0.0) ** 2 +
                  torch.clamp(sum_first_half - 1.0, max=0.0) ** 2)                   # Eq. 24 = E2R Eq. 2
        l_gamma = ((gamma * (1.0 - gamma)) ** 2).sum(dim=-1)                          # Eq. 25 = E2R Eq. 3

        e_loss = l_entity + l_bound + l_gamma                                        # Eq. 26

        # Eq. 32 = E2R Eq. 4 (binary-relation case): membership loss tying gamma_r to
        # the entity-pair representation E_r, and E_r back to the individual entities.
        v_h, v_t = e_h[:, :d], e_t[:, :d]
        e_r = torch.cat([v_h, v_t], dim=-1)                                          # (B, 2d)
        swap_gamma = torch.cat([gamma[:, d:], gamma[:, :d]], dim=-1)                  # (0 I; I 0) gamma

        m_term1 = ((gamma * e_r) ** 2).sum(dim=-1)
        m_term2 = (swap_gamma * e_r).sum(dim=-1) ** 2
        m_term3 = (z_h ** 2).sum(dim=-1)   # == ||(1_d;0_d) . E_r - E_h||^2 given E_r=(v_h,v_t), E_h=(v_h,z_h)
        m_term4 = (z_t ** 2).sum(dim=-1)   # == ||(0_d;1_d) . E_r - E_t||^2, same reasoning
        m_loss = m_term1 + m_term2 + m_term3 + m_term4

        return (e_loss + m_loss).mean()

    def forward(self, batch: dict) -> torch.Tensor:
        base_loss = super().forward(batch)
        # Eq. 23-26 nominally sums over G union G' (positives and negatives); applying
        # it to positives only is a reasonable simplification since every entity that
        # appears as a negative also appears as a positive elsewhere in training.
        reg = self.regularization_loss(batch['h'], batch['r'], batch['t'])
        return base_loss + self.reg_weight * reg
