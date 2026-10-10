import math
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .modules import (
    ProtoGen, HopScorer, MemoryGate, NUM_GATE_FEATS,
    diversity_loss, prototype_diversity,
)
from ..utils.triplet_mask import construct_mask


def build_model(args) -> nn.Module:
    """Factory function to create the model."""
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    """Container for model output tensors."""
    related_logits: Optional[torch.Tensor]
    related_labels: Optional[torch.Tensor]
    hr_labels: torch.Tensor
    hr_logits: torch.Tensor


class CustomBertModel(nn.Module, ABC):
    """RAA-KGC dual-encoder (g1 = hr_bert, g2 = tail_bert), optionally extended with the ARPM
    memory branch (`args.use_memory`). With use_memory=False no memory parameter exists and the
    forward / loss / score are exactly the baseline's.

    ARPM score:  S(t) = cos(q,e_t) + cos(q_hrta,e_t) + lambda_p S_p(t) + lambda_s S_struct(t)
                        [-------- RAA-KGC baseline (Eq. 9) --------]  [gated memory residual]
    """

    NEGATIVE_INF = -1e4

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.hidden_size = self.config.hidden_size

        # Inverse temperature parameter for scaling logits (tau is learnable in the paper)
        self.log_inv_t = nn.Parameter(
            torch.tensor(1.0 / args.t).log(),
            requires_grad=args.finetune_t
        )

        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch

        # Initialize pre-batch negative samples
        self._init_pre_batch_vectors()

        # Dual encoder architecture (same init, no parameter sharing)
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self._drop_unused_pooler(self.hr_bert)
        self.tail_bert = deepcopy(self.hr_bert)
        self.hr_bert.gradient_checkpointing_enable()
        self.tail_bert.gradient_checkpointing_enable()

        # ---- ARPM extension -------------------------------------------------------------
        self.use_memory = bool(getattr(args, 'use_memory', False))
        if self.use_memory:
            self._init_memory(args)

    # ------------------------------------------------------------------ ARPM init
    def _init_memory(self, args) -> None:
        d = self.hidden_size
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
        self.memory_gate = MemoryGate(d, NUM_GATE_FEATS, init_bias=args.gate_init_bias)

        self.tau_r = args.retrieval_temperature
        self.tau_p = args.proto_temperature
        self.eps_struct = args.eps_struct

        self.random_anchor_selection = args.random_anchor_selection
        self.uniform_hop_weighting = args.uniform_hop_weighting
        self.fixed_lambda_p = args.fixed_lambda_p
        self.fixed_lambda_s = args.fixed_lambda_s

    @staticmethod
    def _drop_unused_pooler(encoder: nn.Module) -> None:
        """Remove the encoder's pooler head, if it has one."""
        if getattr(encoder, 'pooler', None) is not None:
            encoder.pooler = None

    def _init_pre_batch_vectors(self):
        """Initialize pre-batch vectors for negative sampling."""
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)

        self.register_buffer(
            "pre_batch_vectors",
            nn.functional.normalize(random_vector, dim=1),
            persistent=False
        )

        self.offset = 0
        self.pre_batch_exs = [None] * num_pre_batch_vectors

    def _encode(self, encoder: nn.Module, token_ids: torch.Tensor,
                mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """Encode input tokens using the specified encoder."""
        outputs = encoder(
            input_ids=token_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

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

    def forward(
            self,
            h_triple_token_ids: torch.Tensor,
            h_triple_mask: torch.Tensor,
            h_triple_token_type_ids: torch.Tensor,
            tail_token_ids: torch.Tensor,
            tail_mask: torch.Tensor,
            tail_token_type_ids: torch.Tensor,
            head_token_ids: torch.Tensor,
            head_mask: torch.Tensor,
            head_token_type_ids: torch.Tensor,
            test_forward: bool,
            anchor_token_ids: Optional[torch.Tensor] = None,
            anchor_mask: Optional[torch.Tensor] = None,
            anchor_token_type_ids: Optional[torch.Tensor] = None,
            anchor_valid: Optional[torch.Tensor] = None,
            cand_tail_token_ids: Optional[torch.Tensor] = None,
            cand_tail_mask: Optional[torch.Tensor] = None,
            cand_tail_token_type_ids: Optional[torch.Tensor] = None,
            candidate_valid_mask: Optional[torch.Tensor] = None,
            candidate_hop_id: Optional[torch.Tensor] = None,
            candidate_is_local: Optional[torch.Tensor] = None,
            only_ent_embedding: bool = False,
            **kwargs
    ) -> Dict:
        """Forward pass.

        Args:
            test_forward: True -> entity-embedding / plain inference path (no anchors)
            only_ent_embedding: True -> only compute candidate entity embeddings
            anchor_*: (B, K, L) padded RAA anchor queries; anchor_valid (B, K) marks real anchors
            cand_*, candidate_*: (B, N, ...) ARPM memory candidates (only if use_memory)
        """
        if test_forward:
            return self._forward_test(
                tail_token_ids, tail_mask, tail_token_type_ids,
                h_triple_token_ids, h_triple_mask, h_triple_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding
            )
        return self._forward_train(
            h_triple_token_ids, h_triple_mask, h_triple_token_type_ids,
            tail_token_ids, tail_mask, tail_token_type_ids,
            head_token_ids, head_mask, head_token_type_ids,
            anchor_token_ids, anchor_mask, anchor_token_type_ids, anchor_valid,
            cand_tail_token_ids, cand_tail_mask, cand_tail_token_type_ids,
            candidate_valid_mask, candidate_hop_id, candidate_is_local,
        )

    def _forward_test(
            self,
            tail_token_ids, tail_mask, tail_token_type_ids,
            h_triple_token_ids, h_triple_mask, h_triple_token_type_ids,
            head_token_ids, head_mask, head_token_type_ids,
            only_ent_embedding: bool
    ) -> Dict:
        """Forward pass for inference/testing without anchors."""
        if only_ent_embedding:
            return self._predict_ent_embedding(
                tail_token_ids, tail_mask, tail_token_type_ids
            )

        hr_vector = self._encode(
            self.hr_bert, h_triple_token_ids, h_triple_mask, h_triple_token_type_ids
        )
        tail_vector = self._encode(
            self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids
        )
        head_vector = self._encode(
            self.tail_bert, head_token_ids, head_mask, head_token_type_ids
        )

        return {
            'hr_vector': hr_vector,
            'tail_vector': tail_vector,
            'head_vector': head_vector
        }

    def _forward_train(
            self,
            h_triple_token_ids, h_triple_mask, h_triple_token_type_ids,
            tail_token_ids, tail_mask, tail_token_type_ids,
            head_token_ids, head_mask, head_token_type_ids,
            anchor_token_ids, anchor_mask, anchor_token_type_ids, anchor_valid,
            cand_tail_token_ids=None, cand_tail_mask=None, cand_tail_token_type_ids=None,
            candidate_valid_mask=None, candidate_hop_id=None, candidate_is_local=None,
    ) -> Dict:
        """Baseline: also produces the anchor-enhanced query embedding e^avg_hrta.
        ARPM (use_memory): additionally returns prototypes, m_struct, gates and auxiliary terms."""
        tail_vector = self._encode(
            self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids
        )
        hr_vector = self._encode(
            self.hr_bert, h_triple_token_ids, h_triple_mask, h_triple_token_type_ids
        )
        head_vector = self._encode(
            self.tail_bert, head_token_ids, head_mask, head_token_type_ids
        )

        anchor_hr_vector = self._encode_anchor_queries(
            hr_vector, anchor_token_ids, anchor_mask, anchor_token_type_ids, anchor_valid
        )

        out = {
            'related_hr_vector': anchor_hr_vector,  # e^avg_hrta
            'hr_vector': hr_vector,  # e_hr
            'tail_vector': tail_vector,  # e_t
            'head_vector': head_vector,
        }

        if self.use_memory and cand_tail_token_ids is not None:
            out.update(self._memory_forward(
                hr_vector, cand_tail_token_ids, cand_tail_mask, cand_tail_token_type_ids,
                candidate_valid_mask, candidate_hop_id, candidate_is_local
            ))
        return out

    def _encode_anchor_queries(
            self,
            hr_vector: torch.Tensor,
            token_ids: Optional[torch.Tensor],
            mask: Optional[torch.Tensor],
            token_type_ids: Optional[torch.Tensor],
            valid: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Eq.(5): e^avg_hrta = average over the k anchor-enhanced query embeddings of each example.
        Plain mean (no re-normalisation). Examples without any anchor fall back to e_hr."""
        if token_ids is None or valid is None or not valid.any():
            return hr_vector

        num_queries, k, seq_len = token_ids.shape
        flat_valid = valid.reshape(-1)

        anchor_vectors = self._encode(
            self.hr_bert,
            token_ids.reshape(-1, seq_len)[flat_valid],
            mask.reshape(-1, seq_len)[flat_valid],
            token_type_ids.reshape(-1, seq_len)[flat_valid],
        )
        group_ids = torch.arange(num_queries, device=hr_vector.device).repeat_interleave(k)[flat_valid]

        summed = torch.zeros(
            num_queries, anchor_vectors.size(1), dtype=anchor_vectors.dtype, device=anchor_vectors.device
        ).index_add(0, group_ids, anchor_vectors)
        counts = torch.bincount(group_ids, minlength=num_queries)

        mean = summed / counts.clamp(min=1).unsqueeze(1).to(summed.dtype)
        avg = mean.to(hr_vector.dtype)

        return torch.where((counts > 0).unsqueeze(1), avg, hr_vector)

    # ================================================================== ARPM memory branch
    def _memory_forward(self, hr_vector, cand_ids, cand_mask, cand_tt, valid, hop_id, is_local) -> Dict:
        # memory anchors: tail-encoder embeddings E_1(t_i), no gradient, deterministic (eval mode)
        was_training = self.tail_bert.training
        self.tail_bert.eval()
        with torch.no_grad():
            cand_emb = self._encode_slots(self.tail_bert, cand_ids, cand_mask, cand_tt, valid)
        self.tail_bert.train(was_training)

        # the memory branch sees q.detach(): it never sends gradient into hr_bert / tail_bert
        with torch.amp.autocast(device_type=hr_vector.device.type, enabled=False):
            mem = self._memory(hr_vector.detach().float(), cand_emb.float(), valid, hop_id, is_local)

        lambda_p, lambda_s = mem['gates'][:, 0], mem['gates'][:, 1]
        if self.fixed_lambda_p is not None:
            lambda_p = torch.full_like(lambda_p, self.fixed_lambda_p)
        if self.fixed_lambda_s is not None:
            lambda_s = torch.full_like(lambda_s, self.fixed_lambda_s)
        # no local anchor at any hop -> no structural evidence -> lambda_s = 0
        lambda_s = torch.where(mem['has_local_anchor'], lambda_s, torch.zeros_like(lambda_s))

        return {
            'prototypes': mem['prototypes'],
            'm_struct': mem['m_struct'],
            'div_loss': mem['div_loss'],
            'proto_div': mem['proto_div'],
            'lambda_p': lambda_p,
            'lambda_s': lambda_s,
        }

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
        anchor gets m_struct = 0 (and lambda_s = 0)."""
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

    # ---- memory scoring (raw cosine-scale scores, used both in training losses and ranking) ----
    def score_prototypes(self, prototypes: torch.Tensor, entity_matrix: torch.Tensor) -> torch.Tensor:
        """S_p(t) = tau_p * (logsumexp_k(cos(p_k, e_t)/tau_p) - log K); zero prototypes give exactly 0."""
        sim = torch.einsum('bkd,ed->bke', prototypes, entity_matrix).float() / self.tau_p
        return self.tau_p * (torch.logsumexp(sim, dim=1) - math.log(prototypes.size(1)))

    def score_struct(self, m_struct: torch.Tensor, entity_matrix: torch.Tensor) -> torch.Tensor:
        return m_struct.mm(entity_matrix.t())

    @staticmethod
    def combined_score(S_base, S_p, S_s, lambda_p, lambda_s) -> torch.Tensor:
        """S = S_base + lambda_p S_p + lambda_s S_struct, S_base = S_q + S_hrta."""
        return S_base + lambda_p.unsqueeze(-1) * S_p + lambda_s.unsqueeze(-1) * S_s

    # ================================================================== baseline logits
    def compute_logits(self, output_dict: Dict, batch_dict: Dict) -> Dict:
        """Compute logits for training/evaluation."""
        if 'related_hr_vector' in output_dict:
            return self._compute_related_logits(output_dict, batch_dict)
        return self._compute_standard_logits(output_dict, batch_dict)

    def _compute_related_logits(self, output_dict: Dict, batch_dict: Dict) -> Dict:
        """Logits for L_hrta (Eq.7) and L_hr (Eq.8)."""
        anchor_vector = output_dict['related_hr_vector']
        tail_vector = output_dict['tail_vector']

        # --- L_hrta: positive = (e^avg_hrta_i, e_t_i); negatives = IBN ---------
        related_labels = torch.arange(anchor_vector.size(0), device=anchor_vector.device)
        related_logits = self._compute_similarity_logits(
            anchor_vector, tail_vector,
            batch_dict.get('related_triplet_mask')
        )

        # --- L_hr: positive = (e_hr_i, e_t_i); negatives = IBN + SN ---------------------
        hr_vector = output_dict['hr_vector']
        hr_labels = torch.arange(hr_vector.size(0), device=hr_vector.device)
        hr_logits = self._compute_similarity_logits(
            hr_vector, tail_vector,
            batch_dict.get('triplet_mask')
        )

        if self.args.use_self_negative and self.training:
            hr_logits = self._add_self_negative_logits(
                hr_logits, hr_vector, output_dict['head_vector'],
                batch_dict['self_negative_mask']
            )

        return {
            'related_logits': related_logits,
            'related_labels': related_labels,
            'hr_labels': hr_labels,
            'hr_logits': hr_logits
        }

    def _compute_standard_logits(self, output_dict: Dict, batch_dict: Dict) -> Dict:
        """Standard (SimKGC) logits without anchors."""
        hr_vector = output_dict['hr_vector']
        tail_vector = output_dict['tail_vector']

        hr_labels = torch.arange(hr_vector.size(0), device=hr_vector.device)
        hr_logits = self._compute_similarity_logits(
            hr_vector, tail_vector,
            batch_dict.get('triplet_mask')
        )

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(
                hr_vector, tail_vector, batch_dict
            )
            hr_logits = torch.cat([hr_logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            hr_logits = self._add_self_negative_logits(
                hr_logits, hr_vector, output_dict['head_vector'],
                batch_dict['self_negative_mask']
            )

        return {
            'related_logits': None,
            'related_labels': None,
            'hr_labels': hr_labels,
            'hr_logits': hr_logits
        }

    def _compute_similarity_logits(
            self,
            query_vectors: torch.Tensor,
            key_vectors: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cosine logits with additive margin gamma on positives, scaled by 1/tau."""
        logits = query_vectors.mm(key_vectors.t())

        if self.training:
            logits = logits - torch.diag_embed(
                torch.full((logits.size(0),), self.add_margin, device=logits.device)
            )

        logits *= self.log_inv_t.exp()

        if mask is not None:
            logits.masked_fill_(~mask, self.NEGATIVE_INF)

        return logits

    def _add_self_negative_logits(
            self,
            logits: torch.Tensor,
            hr_vector: torch.Tensor,
            head_vector: torch.Tensor,
            self_negative_mask: torch.Tensor
    ) -> torch.Tensor:
        """Add self-negative (SN) logits to the existing logits."""
        self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
        self_neg_logits.masked_fill_(~self_negative_mask, self.NEGATIVE_INF)

        return torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

    def _compute_pre_batch_logits(
            self,
            hr_vector: torch.Tensor,
            tail_vector: torch.Tensor,
            batch_dict: Dict
    ) -> torch.Tensor:
        """Compute logits against pre-batch negative samples (not used by RAA-KGC)."""
        batch_exs = batch_dict['batch_data']

        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight

        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, self.NEGATIVE_INF)

        start_idx = self.offset
        end_idx = self.offset + self.batch_size

        self.pre_batch_vectors[start_idx:end_idx] = tail_vector.data.clone()
        self.pre_batch_exs[start_idx:end_idx] = batch_exs
        self.offset = end_idx % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def _predict_ent_embedding(
            self,
            tail_token_ids: torch.Tensor,
            tail_mask: torch.Tensor,
            tail_token_type_ids: torch.Tensor
    ) -> Dict:
        """Predict entity embeddings without gradient computation."""
        ent_vectors = self._encode(
            self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids
        )
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(
        pooling: str,
        cls_output: torch.Tensor,
        mask: torch.Tensor,
        last_hidden_state: torch.Tensor
) -> torch.Tensor:
    """Pool the output hidden states according to the specified pooling strategy."""
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
