"""ARPM-KGC v2.

Final score (per query (h, r, ?)):

    S(t) = cos(q, e_t) + cos(q_hrta, e_t)  +  lambda_p * S_p(t)  +  lambda_s * S_struct(t)
           [------ RAA-KGC baseline ------]   [----- gated memory residual -----]

Key differences from v1:
  1. The baseline's anchor-enhanced query q_hrta (mean of E_0(h, r, t_i) over hop-0 anchors) is
     part of the model, so with lambda = 0 the model IS the baseline.
  2. Memory anchors are represented by their tail embeddings E_1(t_i) from tail_bert, computed
     under no_grad (eval mode). The anchors therefore live in the same space as the entities they
     are scored against and no longer send gradients into hr_bert / tail_bert.
  3. The memory branch sees q.detach(); its auxiliary losses see tail_vector.detach(). The encoders
     are trained only by L_query and L_hrta (exactly as in the baseline).
  4. Prototypes / m_struct are L2-normalised, so S_p and S_struct are real cosine-scale scores.
  5. The gate also sees statistics of the retrieved pool and starts at lambda ~ 0.12.
"""
import math
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .modules import (
    ProtoGen, HopScorer, MemoryGate, NUM_GATE_FEATS,
    diversity_loss, prototype_diversity,
)


def build_model(args) -> nn.Module:
    return ARPMModel(args)


class ARPMModel(nn.Module):
    NEGATIVE_INF = -1e4

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
        self.use_raa = args.anchor_num > 0
        self.anchor_budget = args.anchor_budget

        # query-conditioned anchor attention: logits_i = a_i . (W q) / tau_r, W initialised to I
        self.attn_proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.attn_proj.weight.copy_(torch.eye(d))

        self.proto_gen = ProtoGen(d, args.num_prototypes, temperature=args.proto_attn_temperature)
        self.hop_scorer = HopScorer(d, self.num_hop_slots)
        self.memory_gate = MemoryGate(d, NUM_GATE_FEATS, init_bias=args.gate_init_bias)

        self.tau_r = args.retrieval_temperature
        self.tau_p = args.proto_temperature
        self.eps_struct = args.eps_struct

        self.log_inv_t = nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin

        self.random_anchor_selection = args.random_anchor_selection
        self.uniform_hop_weighting = args.uniform_hop_weighting
        self.fixed_lambda_p = args.fixed_lambda_p
        self.fixed_lambda_s = args.fixed_lambda_s

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
            anchor_token_ids=None, anchor_mask=None, anchor_token_type_ids=None, anchor_valid=None,
            only_ent_embedding: bool = False,
            **kwargs
    ) -> Dict:
        if only_ent_embedding:
            return self._predict_ent_embedding(tail_token_ids, tail_mask, tail_token_type_ids)

        q = self._encode(self.hr_bert, h_triple_token_ids, h_triple_mask, h_triple_token_type_ids)
        tail_vector = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        head_vector = self._encode(self.tail_bert, head_token_ids, head_mask, head_token_type_ids)

        # RAA-KGC: e^avg_hrta = mean_i E_0(h, r, t_i) over hop-0 anchors (falls back to q)
        q_hrta = q
        if self.use_raa and anchor_token_ids is not None:
            a_vec = self._encode_slots(self.hr_bert, anchor_token_ids, anchor_mask,
                                       anchor_token_type_ids, anchor_valid)
            cnt = anchor_valid.sum(dim=1, keepdim=True)
            mean = a_vec.sum(dim=1) / cnt.clamp(min=1).to(a_vec.dtype)
            q_hrta = torch.where(cnt > 0, mean.to(q.dtype), q)

        # memory anchors: E_1(t_i), no gradient, deterministic (eval mode -> no dropout / no checkpointing)
        was_training = self.tail_bert.training
        self.tail_bert.eval()
        with torch.no_grad():
            cand_emb = self._encode_slots(self.tail_bert, cand_tail_token_ids, cand_tail_mask,
                                          cand_tail_token_type_ids, candidate_valid_mask)
        self.tail_bert.train(was_training)

        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            mem = self._memory(q.detach().float(), cand_emb.float(), candidate_valid_mask,
                               candidate_hop_id, candidate_is_local)

        lambda_p, lambda_s = mem['gates'][:, 0], mem['gates'][:, 1]
        if self.fixed_lambda_p is not None:
            lambda_p = torch.full_like(lambda_p, self.fixed_lambda_p)
        if self.fixed_lambda_s is not None:
            lambda_s = torch.full_like(lambda_s, self.fixed_lambda_s)
        # no local anchor at any hop -> no structural evidence -> lambda_s = 0 (structural fact)
        lambda_s = torch.where(mem['has_local_anchor'], lambda_s, torch.zeros_like(lambda_s))

        return {
            'q': q,
            'q_hrta': q_hrta,
            'tail_vector': tail_vector,
            'head_vector': head_vector,
            'prototypes': mem['prototypes'],
            'm_struct': mem['m_struct'],
            'div_loss': mem['div_loss'],
            'proto_div': mem['proto_div'],
            'lambda_p': lambda_p,
            'lambda_s': lambda_s,
        }

    # ------------------------------------------------------------------ memory
    def _memory(self, qm, cand_emb, valid, hop_id, is_local) -> Dict:
        n = cand_emb.size(1)

        logits = torch.einsum('bnd,bd->bn', cand_emb, self.attn_proj(qm)) / self.tau_r
        if self.random_anchor_selection:  # A1
            alpha = valid.float() / valid.sum(-1, keepdim=True).clamp(min=1).float()
        else:
            alpha = torch.softmax(logits.masked_fill(~valid, self.NEGATIVE_INF), dim=-1) * valid
            alpha = alpha / alpha.sum(-1, keepdim=True).clamp(min=1e-12)  # all-invalid row -> zeros

        div = diversity_loss(cand_emb, alpha, valid)

        prototypes = F.normalize(self.proto_gen(cand_emb, qm, alpha, valid), dim=-1)  # zero stays zero
        proto_div = prototype_diversity(prototypes)

        m_struct, has_local_anchor = self._structural_memory(cand_emb, alpha, hop_id, is_local, valid, qm)
        m_struct = F.normalize(m_struct, dim=-1)

        with torch.no_grad():
            n_valid = valid.sum(1).float()
            n_local = (valid & is_local).sum(1).float()
            budget = float(max(self.anchor_budget, 1))
            ent = -(alpha * torch.log(alpha + 1e-12)).sum(1) / math.log(max(n, 2))
            has_hop0 = ((hop_id == 0) & valid & is_local).any(1).float()
            feats = torch.stack([n_valid / budget, n_local / budget, alpha.max(1).values, ent, has_hop0], 1)
        gates = self.memory_gate(qm, feats)

        return {'prototypes': prototypes, 'm_struct': m_struct, 'div_loss': div, 'proto_div': proto_div,
                'has_local_anchor': has_local_anchor, 'gates': gates}

    def _structural_memory(self, cand_emb, alpha, hop_id, is_local, valid_mask, q):
        """m_struct = sum_l beta_l m^(l). Empty hops get exactly zero weight; a query with no local
        anchor gets m_struct = 0 (and lambda_s = 0 in forward)."""
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
    def combined_score(S_base, S_p, S_s, lambda_p, lambda_s) -> torch.Tensor:
        """S = S_base + lambda_p S_p + lambda_s S_struct, S_base = S_q (+ S_hrta)."""
        return S_base + lambda_p.unsqueeze(-1) * S_p + lambda_s.unsqueeze(-1) * S_s


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
