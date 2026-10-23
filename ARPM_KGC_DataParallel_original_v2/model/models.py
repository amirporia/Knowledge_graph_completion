"""ARPM-KGC v3: q_hrta removed, S_p takes its role.

Final score (per query (h, r, ?)):

    S(t) = cos(q, e_t) + c_p * S_p(t) + c_s * S_struct(t)

c_p, c_s are CONSTANTS (no learned gate), tuned on the validation set (`--tune-coefs`) or given with
`--coef-p/--coef-s`. With c_p = c_s = 0 the model is a plain dual-encoder (SimKGC-style) baseline.

What changed w.r.t. v2:
  * The RAA-KGC anchor-enhanced query (q_hrta, S_hrta, L_hrta) is gone; no (h, r, t_i) triples are encoded.
  * S_p is now a PRIMARY score, not a weak side signal. To let it play that role it is trained with
    gradients into the encoders:
        - `memory_grad_query` (default on): q is NOT detached for the semantic branch, so L_proto
          trains E_0 through the anchor attention (attn_proj) and the prototype bilinear forms.
        - `memory_grad_tail`  (trainer, default on): L_proto sees the live tail_vector, so E_1 is trained too.
        - `memory_grad_cand`  (default off, costly): candidate embeddings E_1(t_i) also carry gradient.
  * S_struct is unchanged in form AND in gradient policy: it always sees q.detach(), cand_emb.detach()
    and the detached tail vector (the encoders are never trained through it).
  * Prototypes / m_struct are L2-normalised, so S_p and S_struct are cosine-scale scores.
"""
import math
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .modules import ProtoGen, HopScorer, diversity_loss, prototype_diversity


def build_model(args) -> nn.Module:
    return ARPMModel(args)


class ARPMModel(nn.Module):
    NEGATIVE_INF = -1e4
    MEM_LOG_INV_T_RANGE = (0.0, math.log(100.0))  # inverse temperature kept in [1, 100]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        d = self.config.hidden_size
        self.hidden_size = d

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)  # E_0
        self._drop_unused_pooler(self.hr_bert)
        self.tail_bert = deepcopy(self.hr_bert)  # E_1
        self.hr_bert.gradient_checkpointing_enable()
        self.tail_bert.gradient_checkpointing_enable()

        self.num_hops = args.num_hops
        self.num_hop_slots = args.num_hops + 1
        self.num_prototypes = args.num_prototypes
        self.anchor_budget = args.anchor_budget

        # query-conditioned anchor attention: logits_i = a_i . (W q) / tau_r, W initialised to I
        self.attn_proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.attn_proj.weight.copy_(torch.eye(d))

        self.proto_gen = ProtoGen(d, args.num_prototypes, temperature=args.proto_attn_temperature)
        self.hop_scorer = HopScorer(d, self.num_hop_slots)

        self.tau_r = args.retrieval_temperature
        self.tau_p = args.proto_temperature
        self.eps_struct = args.eps_struct

        self.log_inv_t = nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        # Separate inverse temperatures for the memory losses (tail-to-tail cosines are more compressed
        # than query-to-tail cosines and need their own sharpness).
        self.log_inv_t_proto = nn.Parameter(torch.tensor(1.0 / args.t_mem).log(),
                                            requires_grad=args.finetune_t_mem)
        self.log_inv_t_struct = nn.Parameter(torch.tensor(1.0 / args.t_mem).log(),
                                             requires_grad=args.finetune_t_mem)
        self.add_margin = args.additive_margin

        self.random_anchor_selection = args.random_anchor_selection
        self.uniform_hop_weighting = args.uniform_hop_weighting
        self.memory_grad_query = args.memory_grad_query
        self.memory_grad_cand = args.memory_grad_cand

    def memory_inv_temps(self):
        """(inv_t_proto, inv_t_struct), clamped so a learnable temperature cannot run away."""
        lo, hi = self.MEM_LOG_INV_T_RANGE
        return (self.log_inv_t_proto.clamp(lo, hi).exp(),
                self.log_inv_t_struct.clamp(lo, hi).exp())

    @staticmethod
    def _drop_unused_pooler(encoder: nn.Module) -> None:
        if getattr(encoder, 'pooler', None) is not None:
            encoder.pooler = None

    # ------------------------------------------------------------------ encoding
    def _encode(self, encoder, token_ids, mask, token_type_ids) -> torch.Tensor:
        outputs = encoder(input_ids=token_ids, attention_mask=mask,
                          token_type_ids=token_type_ids, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        return _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

    def _encode_slots(self, encoder, ids, mask, tt, valid) -> torch.Tensor:
        """Encode only the VALID slots of a (B, N, L) batch; invalid slots are zero vectors.
        Returns (B, N, d) float32."""
        batch_size, n, length = ids.shape
        out = torch.zeros(batch_size * n, self.hidden_size, device=ids.device)
        flat_valid = valid.reshape(-1)
        if bool(flat_valid.any()):
            sel = flat_valid.nonzero(as_tuple=False).squeeze(1)
            m = mask.reshape(-1, length)[sel]
            max_len = max(int(m.sum(dim=1).max()), 1)  # drop batch padding
            vec = self._encode(
                encoder,
                ids.reshape(-1, length)[sel][:, :max_len],
                m[:, :max_len],
                tt.reshape(-1, length)[sel][:, :max_len],
            )
            out = out.index_copy(0, sel, vec.float())
        return out.view(batch_size, n, self.hidden_size)

    # ------------------------------------------------------------------ forward
    def forward(
            self,
            tail_token_ids, tail_mask, tail_token_type_ids,
            h_triple_token_ids=None, h_triple_mask=None, h_triple_token_type_ids=None,
            head_token_ids=None, head_mask=None, head_token_type_ids=None,
            cand_tail_token_ids=None, cand_tail_mask=None, cand_tail_token_type_ids=None,
            candidate_valid_mask=None, candidate_hop_id=None, candidate_is_local=None,
            only_ent_embedding: bool = False,
            **kwargs
    ) -> Dict:
        if only_ent_embedding:
            return self._predict_ent_embedding(tail_token_ids, tail_mask, tail_token_type_ids)

        q = self._encode(self.hr_bert, h_triple_token_ids, h_triple_mask, h_triple_token_type_ids)
        tail_vector = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        head_vector = self._encode(self.tail_bert, head_token_ids, head_mask, head_token_type_ids)

        # memory anchors E_1(t_i)
        if self.memory_grad_cand and self.training:
            cand_emb = self._encode_slots(self.tail_bert, cand_tail_token_ids, cand_tail_mask,
                                          cand_tail_token_type_ids, candidate_valid_mask)
        else:
            # no gradient, deterministic (eval mode -> no dropout / no checkpointing)
            was_training = self.tail_bert.training
            self.tail_bert.eval()
            with torch.no_grad():
                cand_emb = self._encode_slots(self.tail_bert, cand_tail_token_ids, cand_tail_mask,
                                              cand_tail_token_type_ids, candidate_valid_mask)
            self.tail_bert.train(was_training)

        q_f = q.float()
        q_sem = q_f if self.memory_grad_query else q_f.detach()  # semantic branch (S_p)
        q_str = q_f.detach()                                      # structural branch (S_struct): always detached

        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            mem = self._memory(q_sem, q_str, cand_emb.float(), candidate_valid_mask,
                               candidate_hop_id, candidate_is_local)

        return {
            'q': q,
            'tail_vector': tail_vector,
            'head_vector': head_vector,
            'prototypes': mem['prototypes'],
            'm_struct': mem['m_struct'],
            'div_loss': mem['div_loss'],
            'proto_div': mem['proto_div'],
        }

    # ------------------------------------------------------------------ memory
    def _anchor_weights(self, qm, cand_emb, valid) -> torch.Tensor:
        if self.random_anchor_selection:  # A1
            return valid.float() / valid.sum(-1, keepdim=True).clamp(min=1).float()
        logits = torch.einsum('bnd,bd->bn', cand_emb, self.attn_proj(qm)) / self.tau_r
        alpha = torch.softmax(logits.masked_fill(~valid, self.NEGATIVE_INF), dim=-1) * valid
        return alpha / alpha.sum(-1, keepdim=True).clamp(min=1e-12)  # all-invalid row -> zeros

    def _memory(self, q_sem, q_str, cand_emb, valid, hop_id, is_local) -> Dict:
        # ---- semantic branch: S_p (may carry gradient into the encoders) ----
        alpha_p = self._anchor_weights(q_sem, cand_emb, valid)
        div = diversity_loss(cand_emb, alpha_p, valid)
        prototypes = F.normalize(self.proto_gen(cand_emb, q_sem, alpha_p, valid), dim=-1)  # zero stays zero
        proto_div = prototype_diversity(prototypes)

        # ---- structural branch: S_struct (unchanged format, fully detached from the encoders) ----
        cand_d = cand_emb.detach()
        alpha_s = self._anchor_weights(q_str, cand_d, valid)
        m_struct, _ = self._structural_memory(cand_d, alpha_s, hop_id, is_local, valid, q_str)
        m_struct = F.normalize(m_struct, dim=-1)

        return {'prototypes': prototypes, 'm_struct': m_struct, 'div_loss': div, 'proto_div': proto_div}

    def _structural_memory(self, cand_emb, alpha, hop_id, is_local, valid_mask, q):
        """m_struct = sum_l beta_l m^(l). Empty hops get exactly zero weight; a query with no local
        anchor gets m_struct = 0."""
        batch_size, n, _ = cand_emb.shape
        L = self.num_hop_slots

        local_valid = valid_mask & is_local
        clamped_hop = hop_id.clamp(min=0, max=L - 1)
        hop_onehot = torch.zeros(batch_size, n, L, device=cand_emb.device, dtype=cand_emb.dtype)
        hop_onehot.scatter_(2, clamped_hop.unsqueeze(-1), 1.0)
        hop_onehot = hop_onehot * local_valid.unsqueeze(-1).to(cand_emb.dtype)

        alpha_hop = alpha.unsqueeze(-1) * hop_onehot
        numer = torch.einsum('bnl,bnd->bld', alpha_hop, cand_emb)
        hop_anchor_count = hop_onehot.sum(dim=1)
        denom = alpha_hop.sum(dim=1).unsqueeze(-1) + self.eps_struct
        m_hop = numer / denom

        hop_valid_mask = hop_anchor_count > 0
        z = self.hop_scorer(q)
        if self.uniform_hop_weighting:  # A5
            n_valid_hops = hop_valid_mask.sum(-1, keepdim=True).clamp(min=1).to(z.dtype)
            beta = hop_valid_mask.to(z.dtype) / n_valid_hops
        else:
            beta = torch.softmax(z.masked_fill(~hop_valid_mask, self.NEGATIVE_INF), dim=-1)
            beta = beta * hop_valid_mask

        m_struct = torch.einsum('bl,bld->bd', beta, m_hop)
        has_local_anchor = hop_valid_mask.any(dim=-1)
        m_struct = torch.where(has_local_anchor.unsqueeze(-1), m_struct, torch.zeros_like(m_struct))
        return m_struct, has_local_anchor

    @torch.no_grad()
    def _predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids) -> Dict:
        ent_vectors = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}

    # ------------------------------------------------------------------ scoring
    def score_query(self, q: torch.Tensor, entity_matrix: torch.Tensor) -> torch.Tensor:
        return q.mm(entity_matrix.t())

    def score_prototypes(self, prototypes: torch.Tensor, entity_matrix: torch.Tensor) -> torch.Tensor:
        """S_p(t) = tau_p * (logsumexp_k(cos(p_k, e_t)/tau_p) - log K): a soft-max over prototypes on
        the cosine scale (zero prototypes give exactly 0)."""
        sim = torch.einsum('bkd,ed->bke', prototypes, entity_matrix).float() / self.tau_p
        return self.tau_p * (torch.logsumexp(sim, dim=1) - math.log(prototypes.size(1)))

    def score_struct(self, m_struct: torch.Tensor, entity_matrix: torch.Tensor) -> torch.Tensor:
        return m_struct.mm(entity_matrix.t())

    @staticmethod
    def combined_score(S_q: torch.Tensor, S_p: torch.Tensor, S_s: torch.Tensor,
                       coef_p: float, coef_s: float) -> torch.Tensor:
        """S = S_q + c_p S_p + c_s S_struct with constant coefficients."""
        return S_q + coef_p * S_p + coef_s * S_s


def _pool_output(pooling, cls_output, mask, last_hidden_state) -> torch.Tensor:
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