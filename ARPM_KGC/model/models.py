"""ARPM-KGC model.

Implements ARPM_KGC_Proposal.tex Sec. 3 end to end on top of the same dual-BERT
backbone as Baseline/model/models.py:

  E_0 = hr_bert  (query encoder AND anchor encoder, Sec. 3.2 notation)
  E_1 = tail_bert (entity encoder)

Pipeline per query (h, r, ?):
  1. q = E_0(h, r)                                   [self._encode]
  2. a_i = E_0(h_i, r, t_i) for every candidate       [self._encode, flattened+batched]
  3. alpha_i = softmax_i(cos(q,a_i) / tau_r)          [Sec. 3.3, QCAS]      -> RQ1
  4. L_div                                            [Sec. 3.3.1]
  5. P_q = ProtoGen(W_q, q) = {p_1..p_K}              [Sec. 3.4, MPSM]      -> RQ2
  6. m_struct = sum_l beta_l * m^(l)                  [Sec. 3.5, ASM]       -> RQ3
  7. [lambda_p, lambda_s] = G_lambda(q)                [Sec. 3.6, AMATP]     -> RQ4
  8. S(t|h,r) = S_q(t) + lambda_p S_p(t) + lambda_s S_struct(t)

Steps 3-7 are computed once per batch, fully vectorized over a padded
(batch, max_candidates) candidate grid (see utils/doc.py::collate), rather than
Baseline's Python for-loop over a short related-triplet list -- this is both the
"clean, efficient" implementation requested and a hard requirement here since the
candidate pool is much larger (local + global, budget M) than Baseline's <=3
related triplets.
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

        # ---- Shared dual encoder (Sec. 3.2 notation: E_0, E_1) ----
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)   # E_0
        self._drop_unused_pooler(self.hr_bert)
        from copy import deepcopy
        self.tail_bert = deepcopy(self.hr_bert)                           # E_1

        # ---- ARPM-KGC memory modules ----
        self.num_hops = args.num_hops
        self.num_prototypes = args.num_prototypes

        self.proto_gen = ProtoGen(hidden_size=d, num_prototypes=args.num_prototypes)
        self.hop_scorer = HopScorer(hidden_size=d, num_hops=args.num_hops)
        self.memory_gate = MemoryGate(hidden_size=d)

        self.tau_r = args.retrieval_temperature
        self.tau_p = args.proto_temperature
        self.eps_struct = args.eps_struct

        # ---- Shared InfoNCE temperature (training-time logit scaling only, see
        # model/trainer.py) and additive margin, reused from Baseline for a like-
        # for-like optimization setup. ----
        self.log_inv_t = nn.Parameter(
            torch.tensor(1.0 / args.t).log(),
            requires_grad=args.finetune_t
        )
        self.add_margin = args.additive_margin

        # ---- Optional discrete (Gumbel) extensions, Sec. 3.8 / ablations A11-A13 ----
        self.use_gumbel_anchor = args.use_gumbel_anchor
        self.gumbel_tau_sel = args.gumbel_tau_sel
        self.use_gumbel_hop = args.use_gumbel_hop
        self.gumbel_tau_hop = args.gumbel_tau_hop
        self.gumbel_topk_hop = args.gumbel_topk_hop
        self.use_gumbel_proto = args.use_gumbel_proto
        self.gumbel_tau_proto = args.gumbel_tau_proto
        if self.use_gumbel_proto:
            self.proto_activation = PrototypeActivationScorer(hidden_size=d, num_prototypes=args.num_prototypes)

        # ---- Ablation overrides (Table "Planned ARPM-KGC ablation study") ----
        self.random_anchor_selection = args.random_anchor_selection   # A1
        self.uniform_hop_weighting = args.uniform_hop_weighting       # A5
        self.fixed_lambda_p = args.fixed_lambda_p                     # A8/A9
        self.fixed_lambda_s = args.fixed_lambda_s                     # A8/A10

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
        """A single unified forward pass, used for both training and evaluation
        (unlike Baseline, ARPM-KGC's memory pipeline is needed at eval time too,
        so there is no separate `test_forward` code path -- see module docstring)."""
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

        # ---- Sec. 3.3 Query-Conditioned Anchor Selection (RQ1) ----
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

        # ---- Sec. 3.3.1 Diversity regularization ----
        div = diversity_loss(cand_emb, alpha, valid_mask)

        # ---- Sec. 3.4 Multi-Prototype Semantic Memory (RQ2) ----
        prototypes = self.proto_gen(cand_emb, q, alpha, valid_mask)

        # ---- Sec. 3.5 Adaptive Structural Memory (RQ3) ----
        m_struct = self._structural_memory(cand_emb, alpha, hop_id, is_local, valid_mask, q)

        return {'prototypes': prototypes, 'm_struct': m_struct, 'div_loss': div}

    def _structural_memory(self, cand_emb, alpha, hop_id, is_local, valid_mask, q) -> torch.Tensor:
        batch_size, n, d = cand_emb.shape
        L = self.num_hops

        local_valid = valid_mask & is_local  # (B, N)
        clamped_hop = (hop_id - 1).clamp(min=0, max=L - 1)  # hop_id in [1..L] -> index [0..L-1]
        hop_onehot = torch.zeros(batch_size, n, L, device=cand_emb.device, dtype=cand_emb.dtype)
        hop_onehot.scatter_(2, clamped_hop.unsqueeze(-1), 1.0)
        hop_onehot = hop_onehot * local_valid.unsqueeze(-1).to(cand_emb.dtype)

        alpha_hop = alpha.unsqueeze(-1) * hop_onehot            # (B, N, L)
        numer = torch.einsum('bnl,bnd->bld', alpha_hop, cand_emb)   # (B, L, d)
        denom = alpha_hop.sum(dim=1).unsqueeze(-1) + self.eps_struct  # (B, L, 1)
        m_hop = numer / denom                                    # (B, L, d)

        z = self.hop_scorer(q)  # (B, L)
        if self.uniform_hop_weighting:
            # A5: fixed beta_l = 1/L, ignoring G_hop(q,l) entirely -- isolates the
            # contribution of *adaptive* hop weighting itself (RQ3).
            beta = torch.full_like(z, 1.0 / z.size(-1))
        elif self.use_gumbel_hop:
            beta = gumbel_softmax_topk(z, self.gumbel_tau_hop, self.gumbel_topk_hop, self.training)
        else:
            beta = torch.softmax(z, dim=-1)

        m_struct = torch.einsum('bl,bld->bd', beta, m_hop)
        return m_struct

    @torch.no_grad()
    def _predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids) -> Dict:
        ent_vectors = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}

    # ------------------------------------------------------------------
    # Scoring (Sec. 3.6, AMATP) -- shared by in-batch training loss
    # (model/trainer.py) and full-entity-set evaluation (evaluation/evaluate.py).
    # These implement the RAW, paper-exact S_q / S_p / S_struct / S. Any extra
    # temperature scaling used to sharpen the training CE loss is applied by the
    # caller on top of these, never inside them, so ranking-based eval metrics
    # and the interpretability analysis of Sec. 5 always see the literal formulas.
    # ------------------------------------------------------------------

    def score_query(self, q: torch.Tensor, entity_matrix: torch.Tensor) -> torch.Tensor:
        """S_q(t) = sim(q, e_t)."""
        return q.mm(entity_matrix.t())

    def score_prototypes(self, prototypes: torch.Tensor, entity_matrix: torch.Tensor,
                          slot_gate: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
        """S_p(t) = tau_p * log sum_k exp(sim(p_k, e_t) / tau_p).

        If `slot_gate` (B, K) is given (Sec. 3.8.3, A13), inactive slots are
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
    """Pool the output hidden states according to the specified pooling strategy
    (identical to Baseline/model/models.py, reused verbatim for parity)."""
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
