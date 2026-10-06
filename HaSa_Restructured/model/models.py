from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .losses import HaSaLoss
from ..setting.config import uses_hasa
from ..utils.triplet_mask import construct_mask


def build_model(args) -> nn.Module:
    """Factory function to create the model for `args.method`."""
    if args.method == 'simkgc':
        return CustomBertModel(args)
    if uses_hasa(args.method):
        return HaSaModel(args)
    raise ValueError(f'Unknown method: {args.method}')


@dataclass
class ModelOutput:
    """Container for model output tensors."""
    logits: torch.Tensor
    labels: torch.Tensor
    inv_t: torch.Tensor
    hr_vector: torch.Tensor
    tail_vector: torch.Tensor


def _drop_unused_pooler(encoder: nn.Module) -> None:
    """Remove the encoder's pooler head when it is never used (it would break DDP otherwise)."""
    if getattr(encoder, 'pooler', None) is not None:
        encoder.pooler = None


def _raw_pool(pooling: str, cls_output: torch.Tensor, mask: torch.Tensor,
              last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Pool the hidden states ('cls', 'max' or 'mean'), no normalisation."""
    if pooling == 'cls':
        return cls_output

    if pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state_masked = last_hidden_state.clone()
        last_hidden_state_masked[input_mask_expanded == 0] = -1e4
        return torch.max(last_hidden_state_masked, 1)[0]

    if pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        return sum_embeddings / sum_mask

    raise ValueError(f'Unknown pooling mode: {pooling}')


def _pool_output(pooling: str, cls_output: torch.Tensor, mask: torch.Tensor,
                 last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Pool the hidden states and L2-normalize (SimKGC)."""
    return nn.functional.normalize(_raw_pool(pooling, cls_output, mask, last_hidden_state), dim=1)


# ===========================================================================
# SimKGC (baseline, behaviour identical to SimKGC_Restructured)
# ===========================================================================

class CustomBertModel(nn.Module, ABC):
    """SimKGC dual-encoder: (head, relation) encoder and tail encoder trained with InfoNCE."""

    NEGATIVE_INF = -1e4
    needs_negative_sampler = False
    score_normalized = True  # cosine scores in [-1, 1]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)

        # Inverse temperature parameter for scaling logits
        self.log_inv_t = nn.Parameter(
            torch.tensor(1.0 / args.t).log(),
            requires_grad=args.finetune_t
        )

        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch

        # Initialize pre-batch negative samples
        self._init_pre_batch_vectors()

        # Dual encoder architecture
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        _drop_unused_pooler(self.hr_bert)
        self.tail_bert = deepcopy(self.hr_bert)
        if args.gradient_checkpointing:
            self.hr_bert.gradient_checkpointing_enable()
            self.tail_bert.gradient_checkpointing_enable()

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

    def forward(
            self,
            hr_token_ids: torch.Tensor,
            hr_mask: torch.Tensor,
            hr_token_type_ids: torch.Tensor,
            tail_token_ids: torch.Tensor,
            tail_mask: torch.Tensor,
            tail_token_type_ids: torch.Tensor,
            head_token_ids: torch.Tensor,
            head_mask: torch.Tensor,
            head_token_type_ids: torch.Tensor,
            only_ent_embedding: bool = False,
            **kwargs
    ) -> Dict:
        if only_ent_embedding:
            return self._predict_ent_embedding(
                tail_token_ids, tail_mask, tail_token_type_ids
            )

        hr_vector = self._encode(
            self.hr_bert, hr_token_ids, hr_mask, hr_token_type_ids
        )
        tail_vector = self._encode(
            self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids
        )
        head_vector = self._encode(
            self.tail_bert, head_token_ids, head_mask, head_token_type_ids
        )

        # DataParallel only supports tensor/dict
        return {
            'hr_vector': hr_vector,
            'tail_vector': tail_vector,
            'head_vector': head_vector
        }

    def compute_loss(self, output_dict: Dict, batch_dict: Dict) -> Tuple[torch.Tensor, ModelOutput]:
        """Bidirectional InfoNCE: (h, r) -> tail and tail -> (h, r)."""
        outputs = ModelOutput(**self.compute_logits(output_dict, batch_dict))
        batch_size = outputs.logits.size(0)

        loss = F.cross_entropy(outputs.logits, outputs.labels)
        loss = loss + F.cross_entropy(outputs.logits[:, :batch_size].t(), outputs.labels)
        return loss, outputs

    def compute_logits(self, output_dict: Dict, batch_dict: Dict) -> Dict:
        """Compute InfoNCE logits (in-batch + optional pre-batch + optional self negatives)."""
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        labels = torch.arange(hr_vector.size(0), device=hr_vector.device)

        logits = self._compute_similarity_logits(
            hr_vector, tail_vector, batch_dict.get('triplet_mask', None)
        )

        # Add pre-batch negative logits
        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(
                hr_vector, tail_vector, batch_dict
            )
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        # Add self-negative logits (head entity as a negative tail)
        if self.args.use_self_negative and self.training:
            logits = self._add_self_negative_logits(
                logits, hr_vector, output_dict['head_vector'],
                batch_dict['self_negative_mask']
            )

        return {
            'logits': logits,
            'labels': labels,
            'inv_t': self.log_inv_t.detach().exp(),
            'hr_vector': hr_vector.detach(),
            'tail_vector': tail_vector.detach()
        }

    def _compute_similarity_logits(
            self,
            query_vectors: torch.Tensor,
            key_vectors: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute scaled cosine logits with additive margin (training) and optional masking."""
        logits = query_vectors.mm(key_vectors.t())

        if self.training:
            # Apply additive margin to the diagonal (positive pairs)
            logits = logits - torch.diag_embed(
                torch.full((logits.size(0),), self.add_margin, device=logits.device, dtype=logits.dtype)
            )

        logits = logits * self.log_inv_t.exp()

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
        """Append the (hr, head) similarity as one extra negative column."""
        self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
        self_neg_logits.masked_fill_(~self_negative_mask, self.NEGATIVE_INF)

        return torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

    def _compute_pre_batch_logits(
            self,
            hr_vector: torch.Tensor,
            tail_vector: torch.Tensor,
            batch_dict: Dict
    ) -> torch.Tensor:
        """Compute logits against pre-batch negative samples and update the buffer."""
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']

        # batch_size x num_pre_batch_vectors
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight

        # Mask out pre-batch tails that are also valid answers
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, self.NEGATIVE_INF)

        # Update pre-batch buffer
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


# ===========================================================================
# HaSa
# ===========================================================================

class HaSaModel(nn.Module):
    """HaSa encoder: ONE shared text encoder for head / relation / tail / negatives, a projection head
    (Linear -> LayerNorm -> Dropout), and a GRU that fuses [head, relation] into the (h, r) vector.
    Scores are raw dot products (no L2 normalisation)."""

    NEGATIVE_INF = -1e4
    needs_negative_sampler = True
    score_normalized = False

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)

        self.bert = AutoModel.from_pretrained(args.pretrained_model)
        if args.pooling != 'pooler':
            _drop_unused_pooler(self.bert)
        if args.gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()

        hidden = self.config.hidden_size
        self.dense = nn.Sequential(
            nn.Linear(hidden, args.em_dim, bias=True),
            nn.LayerNorm(args.em_dim, eps=1e-12, elementwise_affine=True),
            nn.Dropout(p=0.1, inplace=False),
        )
        self.gru = nn.GRU(args.em_dim, args.em_dim, 1, batch_first=True)

        self.loss_fn = HaSaLoss(
            method=args.method, tau=args.debias_tau, margin=args.hasa_margin,
            plus=args.plus, neg_count_mode=args.neg_count_mode,
        )

    # ---- encoders -----------------------------------------------------------------------------
    def _embed(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=token_ids, attention_mask=mask, return_dict=True)
        if self.args.pooling == 'pooler':
            pooled = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state
            pooled = _raw_pool(self.args.pooling, hidden[:, 0, :], mask, hidden)
        return self.dense(pooled)

    def _fuse(self, z_head: torch.Tensor, z_relation: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        _, hn = self.gru(torch.stack([z_head, z_relation], dim=1))
        return hn[0]

    def encode_hr(self, head_token_ids, head_mask, relation_token_ids, relation_mask) -> torch.Tensor:
        return self._fuse(self._embed(head_token_ids, head_mask),
                          self._embed(relation_token_ids, relation_mask))

    @torch.no_grad()
    def embed_entities(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._embed(token_ids, mask).detach()

    def forward(
            self,
            head_token_ids: torch.Tensor,
            head_mask: torch.Tensor,
            relation_token_ids: torch.Tensor,
            relation_mask: torch.Tensor,
            tail_token_ids: torch.Tensor,
            tail_mask: torch.Tensor,
            only_ent_embedding: bool = False,
            hard_token_ids: Optional[torch.Tensor] = None,
            hard_mask: Optional[torch.Tensor] = None,
            false_token_ids: Optional[torch.Tensor] = None,
            false_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict:
        if only_ent_embedding:
            return {'ent_vectors': self.embed_entities(tail_token_ids, tail_mask)}

        z_head = self._embed(head_token_ids, head_mask)
        z_relation = self._embed(relation_token_ids, relation_mask)
        z_tail = self._embed(tail_token_ids, tail_mask)

        out = {
            'hr_vector': self._fuse(z_head, z_relation),
            'head_vector': z_head,
            'tail_vector': z_tail,
        }
        if hard_token_ids is not None:
            out['hard_vectors'] = self._embed(hard_token_ids, hard_mask)
        if false_token_ids is not None:
            out['false_vectors'] = self._embed(false_token_ids, false_mask)
        return out

    # ---- logits / loss ------------------------------------------------------------------------
    def compute_logits(self, output_dict: Dict, batch_dict: Dict) -> Dict:
        """Plain in-batch dot-product logits (validation / accuracy)."""
        hr_vector = output_dict['hr_vector'].float()
        tail_vector = output_dict['tail_vector'].float()
        labels = torch.arange(hr_vector.size(0), device=hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        mask = batch_dict.get('triplet_mask', None)
        if mask is not None:
            logits = logits.masked_fill(~mask, self.NEGATIVE_INF)

        return {
            'logits': logits,
            'labels': labels,
            'inv_t': torch.ones((), device=logits.device),
            'hr_vector': hr_vector.detach(),
            'tail_vector': tail_vector.detach()
        }

    def compute_loss(self, output_dict: Dict, batch_dict: Dict) -> Tuple[torch.Tensor, ModelOutput]:
        loss, scores = self.loss_fn(
            output_dict['hr_vector'], output_dict['head_vector'], output_dict['tail_vector'],
            output_dict['hard_vectors'], output_dict.get('false_vectors'),
            batch_dict.get('prob'), batch_dict.get('f_prob'),
        )
        batch_size = scores.size(0)
        return loss, ModelOutput(
            logits=scores.detach(),
            labels=torch.arange(batch_size, device=scores.device),
            inv_t=torch.ones((), device=scores.device),
            hr_vector=output_dict['hr_vector'].detach(),
            tail_vector=output_dict['tail_vector'].detach(),
        )
