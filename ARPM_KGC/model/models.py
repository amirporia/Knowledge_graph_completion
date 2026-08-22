"""ARPM-KGC model.

Pipeline per query (h, r, ?):
  1. q = E_0(h, r)                                   [self._encode]
  2. a_i = E_0(h_i, r, t_i) for every candidate       [self._encode, flattened+batched]
  3. alpha_i = softmax_i(cos(q,a_i) / tau_r)               -> RQ1
  4. L_div
  5. P_q = ProtoGen(W_q, q) = {p_1..p_K}                   -> RQ2
  6. m_struct = sum_l beta_l * m^(l)                       -> RQ3
  7. [lambda_p, lambda_s] = G_lambda(q)                    -> RQ4
  8. S(t|h,r) = S_q(t) + lambda_p S_p(t) + lambda_s S_struct(t)

Steps 3-7 are computed once per batch, fully vectorized over a padded
(batch, max_candidates) candidate grid (see utils/doc.py::collate)
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .modules import (
    ProtoGen, HopScorer, MemoryGate, PrototypeActivationScorer,
    diversity_loss, gumbel_sigmoid_gate, gumbel_softmax_topk, gumbel_sigmoid_slot_gate,
)


def build_model(args) -> nn.Module:
    """Factory function to create the model."""
    return ARPMModel(args)


class ARPMModel(nn.Module):
    """Adaptive Relation-Aware Prototype Memory model for KGC."""

    NEGATIVE_INF = -1e4

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        d = self.config.hidden_size

        # ---- Shared dual encoder ----
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)  # E_0
        self._drop_unused_pooler(self.hr_bert)
        from copy import deepcopy
        self.tail_bert = deepcopy(self.hr_bert)  # E_1

        # ---- ARPM-KGC memory modules ----
        # `args.num_hops` (N) is the max graph distance considered beyond the
        # same-head/same-relation category: local anchors span hop-0
        # (same head & relation), hop-1 (graph distance 1), ..., hop-N (graph
        # distance N) -- N+1 categories in total (utils/candidate_pool.py),
        # so structural memory needs N+1 slots.
        self.num_hops = args.num_hops
        self.num_hop_slots = args.num_hops + 1
        self.num_prototypes = args.num_prototypes

        self.proto_gen = ProtoGen(hidden_size=d, num_prototypes=args.num_prototypes)
        self.hop_scorer = HopScorer(hidden_size=d, num_hops=self.num_hop_slots)
        self.memory_gate = MemoryGate(hidden_size=d)

        self.tau_r = args.retrieval_temperature
        self.tau_p = args.proto_temperature
        self.eps_struct = args.eps_struct

        # Shared InfoNCE temperature (training-time logit scaling only) and additive margin
        self.log_inv_t = nn.Parameter(
            torch.tensor(1.0 / args.t).log(),
            requires_grad=args.finetune_t
        )
        self.add_margin = args.additive_margin

        # ---- Optional discrete (Gumbel) extensions, ablations A11-A13 ----
        self.use_gumbel_anchor = args.use_gumbel_anchor
        self.gumbel_tau_sel = args.gumbel_tau_sel
        self.use_gumbel_hop = args.use_gumbel_hop
        self.gumbel_tau_hop = args.gumbel_tau_hop
        self.gumbel_topk_hop = args.gumbel_topk_hop
        self.use_gumbel_proto = args.use_gumbel_proto
        self.gumbel_tau_proto = args.gumbel_tau_proto
        if self.use_gumbel_proto:
            self.proto_activation = PrototypeActivationScorer(hidden_size=d, num_prototypes=args.num_prototypes)

        # ---- Ablation overrides ----
        self.random_anchor_selection = args.random_anchor_selection  # A1
        self.uniform_hop_weighting = args.uniform_hop_weighting  # A5
        self.fixed_lambda_p = args.fixed_lambda_p  # A8/A9
        self.fixed_lambda_s = args.fixed_lambda_s  # A8/A10

    @staticmethod
    def _drop_unused_pooler(encoder: nn.Module) -> None:
        if getattr(encoder, 'pooler', None) is not None:
            encoder.pooler = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, encoder: nn.Module, token_ids: torch.Tensor,
                mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        outputs = encoder(
            input_ids=token_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        return _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
            self,
            tail_token_ids: torch.Tensor,
            tail_mask: torch.Tensor,
            tail_token_type_ids: torch.Tensor,
            h_triple_token_ids: Optional[torch.Tensor] = None,
            h_triple_mask: Optional[torch.Tensor] = None,
            h_triple_token_type_ids: Optional[torch.Tensor] = None,
            head_token_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            head_token_type_ids: Optional[torch.Tensor] = None,
            candidate_token_ids: Optional[torch.Tensor] = None,
            candidate_mask_tok: Optional[torch.Tensor] = None,
            candidate_token_type_ids: Optional[torch.Tensor] = None,
            candidate_valid_mask: Optional[torch.Tensor] = None,
            candidate_hop_id: Optional[torch.Tensor] = None,
            candidate_is_local: Optional[torch.Tensor] = None,
            max_candidates: Optional[int] = None,
            only_ent_embedding: bool = False,
            **kwargs
    ) -> Dict:
        """A single unified forward pass, used for both training and evaluation."""
        if only_ent_embedding:
            return self._predict_ent_embedding(tail_token_ids, tail_mask, tail_token_type_ids)

        q = self._encode(self.hr_bert, h_triple_token_ids, h_triple_mask, h_triple_token_type_ids)
        tail_vector = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        head_vector = self._encode(self.tail_bert, head_token_ids, head_mask, head_token_type_ids)

        memory_out = self._build_memory(
            q, candidate_token_ids, candidate_mask_tok, candidate_token_type_ids,
            candidate_valid_mask, candidate_hop_id, candidate_is_local, max_candidates
        )

        gates = self.memory_gate(q)  # (B, 2)
        lambda_p, lambda_s = gates[:, 0], gates[:, 1]

        # A8/A9/A10: override the learned gate with a fixed constant, if requested.
        if self.fixed_lambda_p is not None:
            lambda_p = torch.full_like(lambda_p, self.fixed_lambda_p)
        if self.fixed_lambda_s is not None:
            lambda_s = torch.full_like(lambda_s, self.fixed_lambda_s)

        # Failure mode 2: no local anchor at any hop
        # for this query -> no structural evidence exists, so lambda_s is
        # hard-set to 0 regardless of the learned gate OR a --fixed-lambda-s
        # ablation override above -- "no evidence" is a structural fact about
        # the query, not a policy choice.
        has_local_anchor = memory_out['has_local_anchor']
        lambda_s = torch.where(has_local_anchor, lambda_s, torch.zeros_like(lambda_s))

        output = {
            'q': q,
            'tail_vector': tail_vector,
            'head_vector': head_vector,
            'prototypes': memory_out['prototypes'],
            'm_struct': memory_out['m_struct'],
            'div_loss': memory_out['div_loss'],
            'lambda_p': lambda_p,
            'lambda_s': lambda_s,
        }

        if self.use_gumbel_proto:
            zeta = self.proto_activation(q)  # (B, K)
            output['slot_gate'] = gumbel_sigmoid_slot_gate(zeta, self.gumbel_tau_proto, self.training)

        return output

    def _build_memory(self, q, cand_ids, cand_mask_tok, cand_type_ids,
                      valid_mask, hop_id, is_local, max_candidates) -> Dict:
        batch_size = q.size(0)
        d = q.size(1)

        cand_emb_flat = self._encode(self.hr_bert, cand_ids, cand_mask_tok, cand_type_ids)
        cand_emb = cand_emb_flat.view(batch_size, max_candidates, d)
        cand_emb = cand_emb * valid_mask.unsqueeze(-1)  # zero-out padded/invalid slots

        # ---- Query-Conditioned Anchor Selection (RQ1) ----
        s = torch.einsum('bnd,bd->bn', cand_emb, q)  # cosine similarity (both L2-normalized)
        s_masked = s.masked_fill(~valid_mask, float('-inf'))

        if self.random_anchor_selection:
            # A1: uniform weights over valid candidates (query-blind), instead of
            # the learned softmax(cos(q,a_i)/tau_r) -- isolates the contribution
            # of query-conditioned selection itself (RQ1).
            n_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(cand_emb.dtype)
            alpha = valid_mask.to(cand_emb.dtype) / n_valid
        else:
            alpha = torch.softmax(s_masked / self.tau_r, dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0) * valid_mask  # rows with 0 valid candidates -> 0

        if self.use_gumbel_anchor:
            keep_gate = gumbel_sigmoid_gate(s_masked, self.gumbel_tau_sel, self.training)  # (B, N)
            alpha = alpha * keep_gate  # W_q^sparse = {(a_i, alpha_i * c_i^ST) : c_i^ST = 1}

        # ---- Diversity regularization ----
        div = diversity_loss(cand_emb, alpha, valid_mask)

        # ---- Multi-Prototype Semantic Memory (RQ2) ----
        prototypes = self.proto_gen(cand_emb, q, alpha, valid_mask)

        # ---- Adaptive Structural Memory (RQ3) ----
        m_struct, has_local_anchor = self._structural_memory(cand_emb, alpha, hop_id, is_local, valid_mask, q)

        return {'prototypes': prototypes, 'm_struct': m_struct, 'div_loss': div,
                'has_local_anchor': has_local_anchor}

    def _structural_memory(self, cand_emb, alpha, hop_id, is_local, valid_mask, q):
        """Adaptive Structural Memory, with two cheap, parameter-free
        safeguards against degenerate candidate pools:

        1. Some hops empty, not all: the hop-selection softmax (or its Gumbel/
           uniform-ablation counterparts) is masked so a hop with zero local
           anchors gets EXACTLY zero weight, rather than relying on G_hop to
           learn that on its own.
        2. No local anchor at ANY hop for this query: m_struct has nothing to
           be built from, so it is hard-set to 0 and the caller (`forward`)
           hard-overrides lambda_s <- 0 for that query, rather than trusting
           the learned (or ablation-fixed) gate to discover this rare case
           itself.

        Returns (m_struct, has_local_anchor) where has_local_anchor is a
        (B,) bool used by `forward` for the lambda_s override.
        """
        batch_size, n, d = cand_emb.shape
        L = self.num_hop_slots  # num_hops + 1 (hop-0 through hop-num_hops inclusive)

        local_valid = valid_mask & is_local  # (B, N)

        # hop_id is 0-indexed for local anchors: slot 0 = same head & relation
        # as the query (graph distance 0, no traversal), slot 1 = graph
        # distance 1 (direct/1-edge-away neighbors), ..., slot L-1 (=
        # self.num_hops) = graph distance self.num_hops (farthest configured);
        # -1 for global anchors (no known structural distance) and for
        # padding (see utils/candidate_pool.py, utils/doc.py). Clamping here
        # is only to keep the scatter index non-negative -- any candidate
        # that isn't a genuine local anchor is zeroed out immediately after
        # via `local_valid`, regardless of which slot its clamped hop lands in.
        clamped_hop = hop_id.clamp(min=0, max=L - 1)
        hop_onehot = torch.zeros(batch_size, n, L, device=cand_emb.device, dtype=cand_emb.dtype)
        hop_onehot.scatter_(2, clamped_hop.unsqueeze(-1), 1.0)
        hop_onehot = hop_onehot * local_valid.unsqueeze(-1).to(cand_emb.dtype)

        alpha_hop = alpha.unsqueeze(-1) * hop_onehot  # (B, N, L)
        numer = torch.einsum('bnl,bnd->bld', alpha_hop, cand_emb)  # (B, L, d)
        hop_anchor_count = hop_onehot.sum(dim=1)  # (B, L): real local anchors per hop slot
        denom = alpha_hop.sum(dim=1).unsqueeze(-1) + self.eps_struct  # (B, L, 1)
        m_hop = numer / denom  # (B, L, d)

        # ---- Failure mode 1: mask empty hops out of the hop-selection weighting ----
        hop_valid_mask = hop_anchor_count > 0  # (B, L)

        z = self.hop_scorer(q)  # (B, L)
        if self.uniform_hop_weighting:
            # A5: fixed uniform beta_l, ignoring G_hop(q,l) entirely -- but still
            # only over hops that actually have a local anchor; an empty hop
            # must get zero weight under any weighting *policy*, ablated or not.
            n_valid_hops = hop_valid_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(z.dtype)
            beta = hop_valid_mask.to(z.dtype) / n_valid_hops
        elif self.use_gumbel_hop:
            beta = gumbel_softmax_topk(z, self.gumbel_tau_hop, self.gumbel_topk_hop,
                                       self.training, valid_mask=hop_valid_mask)
        else:
            z_masked = z.masked_fill(~hop_valid_mask, self.NEGATIVE_INF)
            beta = torch.softmax(z_masked, dim=-1)

        m_struct = torch.einsum('bl,bld->bd', beta, m_hop)

        # ---- Failure mode 2: no local anchor at any hop -> hard-zero m_struct ----
        has_local_anchor = hop_valid_mask.any(dim=-1)  # (B,)
        m_struct = torch.where(
            has_local_anchor.unsqueeze(-1), m_struct, torch.zeros_like(m_struct)
        )

        return m_struct, has_local_anchor

    @torch.no_grad()
    def _predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids) -> Dict:
        ent_vectors = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}

    def score_query(self, q: torch.Tensor, entity_matrix: torch.Tensor) -> torch.Tensor:
        """S_q(t) = sim(q, e_t)."""
        return q.mm(entity_matrix.t())

    def score_prototypes(self, prototypes: torch.Tensor, entity_matrix: torch.Tensor,
                         slot_gate: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
        """S_p(t) = tau_p * log sum_k exp(sim(p_k, e_t) / tau_p).

        If `slot_gate` (B, K) is given (A13), inactive slots are
        masked out of the sum instead of contributing at full weight:
        S_p^ST(t) = tau_p * log( sum_k omega_k^ST * exp(sim(p_k,e_t)/tau_p) + eps ).
        """
        sim = torch.einsum('bkd,ed->bke', prototypes, entity_matrix) / self.tau_p  # (B, K, Ne)

        if slot_gate is not None:
            weighted_exp = slot_gate.unsqueeze(-1) * torch.exp(sim)
            return self.tau_p * torch.log(weighted_exp.sum(dim=1) + eps)

        return self.tau_p * torch.logsumexp(sim, dim=1)

    def score_struct(self, m_struct: torch.Tensor, entity_matrix: torch.Tensor) -> torch.Tensor:
        """S_struct(t) = sim(m_struct, e_t)."""
        return m_struct.mm(entity_matrix.t())

    def combined_score(self, S_q: torch.Tensor, S_p: torch.Tensor, S_s: torch.Tensor,
                       lambda_p: torch.Tensor, lambda_s: torch.Tensor) -> torch.Tensor:
        """S(t|h,r) = S_q(t) + lambda_p * S_p(t) + lambda_s * S_struct(t)."""
        return S_q + lambda_p.unsqueeze(-1) * S_p + lambda_s.unsqueeze(-1) * S_s


def _pool_output(
        pooling: str,
        cls_output: torch.Tensor,
        mask: torch.Tensor,
        last_hidden_state: torch.Tensor
) -> torch.Tensor:
    """Pool the output hidden states according to the specified pooling strategy"""
    if pooling == 'cls':
        output_vector = cls_output

    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state_masked = last_hidden_state.clone()
        last_hidden_state_masked[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state_masked, 1)[0]

    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask

    else:
        raise ValueError(f'Unknown pooling mode: {pooling}')

    return nn.functional.normalize(output_vector, dim=1)
