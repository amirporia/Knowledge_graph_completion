from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

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
    """Custom BERT model for relation prediction tasks."""

    NEGATIVE_INF = -1e4

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
        self._drop_unused_pooler(self.hr_bert)
        self.tail_bert = deepcopy(self.hr_bert)

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

    def forward(
            self,
            related_h_triple_token_ids_list: List[torch.Tensor],
            related_h_triple_mask_list: List[torch.Tensor],
            related_h_triple_token_type_ids_list: List[torch.Tensor],
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
            only_ent_embedding: bool = False,
            **kwargs
    ) -> Dict:
        """Forward pass for training or inference.

        Args:
            test_forward: If True, run in inference mode without related triplet processing
            only_ent_embedding: If True, only compute entity embeddings (inference only)
        """
        if test_forward:
            return self._forward_test(
                tail_token_ids, tail_mask, tail_token_type_ids,
                h_triple_token_ids, h_triple_mask, h_triple_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding
            )
        else:
            return self._forward_train(
                related_h_triple_token_ids_list,
                related_h_triple_mask_list,
                related_h_triple_token_type_ids_list,
                h_triple_token_ids, h_triple_mask, h_triple_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids
            )

    def _forward_test(
            self,
            tail_token_ids: torch.Tensor,
            tail_mask: torch.Tensor,
            tail_token_type_ids: torch.Tensor,
            h_triple_token_ids: torch.Tensor,
            h_triple_mask: torch.Tensor,
            h_triple_token_type_ids: torch.Tensor,
            head_token_ids: torch.Tensor,
            head_mask: torch.Tensor,
            head_token_type_ids: torch.Tensor,
            only_ent_embedding: bool
    ) -> Dict:
        """Forward pass for inference/testing."""
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
            'head_vector': head_vector,
            'related': False
        }

    def _forward_train(
            self,
            related_h_triple_token_ids_list: List[torch.Tensor],
            related_h_triple_mask_list: List[torch.Tensor],
            related_h_triple_token_type_ids_list: List[torch.Tensor],
            h_triple_token_ids: torch.Tensor,
            h_triple_mask: torch.Tensor,
            h_triple_token_type_ids: torch.Tensor,
            tail_token_ids: torch.Tensor,
            tail_mask: torch.Tensor,
            tail_token_type_ids: torch.Tensor,
            head_token_ids: torch.Tensor,
            head_mask: torch.Tensor,
            head_token_type_ids: torch.Tensor
    ) -> Dict:
        """Forward pass for training with related triplet processing."""
        # Encode main triplets
        tail_vector = self._encode(
            self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids
        )
        hr_vector = self._encode(
            self.hr_bert, h_triple_token_ids, h_triple_mask, h_triple_token_type_ids
        )
        head_vector = self._encode(
            self.tail_bert, head_token_ids, head_mask, head_token_type_ids
        )

        # Process related triplets
        final_hr_vector = self._encode_related_triplets(
            related_h_triple_token_ids_list,
            related_h_triple_mask_list,
            related_h_triple_token_type_ids_list
        )

        return {
            'related_hr_vector': final_hr_vector,
            'hr_vector': hr_vector,
            'tail_vector': tail_vector,
            'head_vector': head_vector,
            'related': True
        }

    def _encode_related_triplets(
            self,
            token_ids_list: List[torch.Tensor],
            mask_list: List[torch.Tensor],
            token_type_ids_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Encode related triplets and average their representations."""
        hr_vectors = []

        for token_ids, mask, token_type_ids in zip(token_ids_list, mask_list, token_type_ids_list):
            hr_vector = self._encode(
                self.hr_bert, token_ids, mask, token_type_ids
            )
            hr_vectors.append(hr_vector)

        # Average each group of vectors
        averaged_vectors = [torch.mean(vec, dim=0, keepdim=True) for vec in hr_vectors]

        return torch.cat(averaged_vectors, dim=0)

    def compute_logits(self, output_dict: Dict, batch_dict: Dict) -> Dict:
        """Compute logits for training/evaluation.

        Args:
            output_dict: Output from forward pass
            batch_dict: Batch data dictionary
        """
        if output_dict['related']:
            return self._compute_related_logits(output_dict, batch_dict)
        else:
            return self._compute_standard_logits(output_dict, batch_dict)

    def _compute_related_logits(self, output_dict: Dict, batch_dict: Dict) -> Dict:
        """Compute logits including related triplet information."""
        related_hr_vector = output_dict['related_hr_vector']
        tail_vector = output_dict['tail_vector']

        # Compute related logits
        related_labels = torch.arange(related_hr_vector.size(0), device=related_hr_vector.device)
        related_logits = self._compute_similarity_logits(
            related_hr_vector, tail_vector,
            batch_dict.get('related_triplet_mask')
        )

        # Compute standard HR logits
        hr_vector = output_dict['hr_vector']
        hr_labels = torch.arange(hr_vector.size(0), device=hr_vector.device)
        hr_logits = self._compute_similarity_logits(
            hr_vector, tail_vector,
            batch_dict.get('triplet_mask')
        )

        # Add self-negative logits if enabled
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
        """Compute standard logits without related triplet information."""
        hr_vector = output_dict['hr_vector']
        tail_vector = output_dict['tail_vector']

        hr_labels = torch.arange(hr_vector.size(0), device=hr_vector.device)
        hr_logits = self._compute_similarity_logits(
            hr_vector, tail_vector,
            batch_dict.get('triplet_mask')
        )

        # Add pre-batch negative logits
        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(
                hr_vector, tail_vector, batch_dict
            )
            hr_logits = torch.cat([hr_logits, pre_batch_logits], dim=-1)

        # Add self-negative logits if enabled
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
        """Compute similarity logits with optional margin and masking.

        Args:
            query_vectors: Query embeddings
            key_vectors: Key embeddings to compare against
            mask: Optional mask to apply to logits
        """
        logits = query_vectors.mm(key_vectors.t())

        if self.training:
            # Apply additive margin to diagonal (positive pairs)
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
        """Add self-negative logits to the existing logits."""
        self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
        self_neg_logits.masked_fill_(~self_negative_mask, self.NEGATIVE_INF)

        return torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

    def _compute_pre_batch_logits(
            self,
            hr_vector: torch.Tensor,
            tail_vector: torch.Tensor,
            batch_dict: Dict
    ) -> torch.Tensor:
        """Compute logits against pre-batch negative samples."""
        batch_exs = batch_dict['batch_data']

        # Compute similarity with pre-batch vectors
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight

        # Apply triplet mask if available
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


def _pool_output(
        pooling: str,
        cls_output: torch.Tensor,
        mask: torch.Tensor,
        last_hidden_state: torch.Tensor
) -> torch.Tensor:
    """Pool the output hidden states according to the specified pooling strategy.

    Args:
        pooling: Pooling strategy ('cls', 'max', or 'mean')
        cls_output: CLS token output
        mask: Attention mask
        last_hidden_state: Full hidden states from the last layer
    """
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