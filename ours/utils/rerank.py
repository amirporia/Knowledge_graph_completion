from typing import List

import torch

from .dict_hub import get_link_graph
from .doc import Example
from .triplet import EntityDict
from ..setting.config import args


def rerank_by_graph(related_batch_score: torch.tensor,
                    batch_score: torch.tensor,
                    examples: List[Example],
                    entity_dict: EntityDict) -> None:
    """
    Rerank batch scores using graph neighborhood information.

    Args:
        related_batch_score: Scores from related batch
        batch_score: Batch scores to be reranked
        examples: List of examples to process
        entity_dict: Entity dictionary for mapping
    """
    # Validate inductive setting
    if args.task == 'wiki5m_ind':
        assert args.neighbor_weight < 1e-6, 'Inductive setting cannot use re-rank strategy'

    # Early return if neighbor weight is negligible
    if args.neighbor_weight < 1e-6:
        return

    # Sort related batch scores
    related_sorted_scores, related_sorted_indices = torch.sort(
        related_batch_score, dim=-1, descending=True
    )

    # Process each example in the batch
    for idx in range(batch_score.size(0)):
        current_example = examples[idx]

        # Get n-hop neighbor indices for the current head entity
        neighbor_indices = get_link_graph().get_n_hop_entity_indices(
            current_example.head_id,
            entity_dict=entity_dict,
            n_hop=args.rerank_n_hop
        )

        # Add neighbor weight to batch scores
        if neighbor_indices:
            delta_weights = torch.tensor(
                [args.neighbor_weight for _ in neighbor_indices],
                device=batch_score.device
            )
            neighbor_tensor = torch.LongTensor(list(neighbor_indices)).to(batch_score.device)
            batch_score[idx].index_add_(0, neighbor_tensor, delta_weights)

            # Weight adjustment for related neighbors (currently zero - placeholder)
            top_related_indices = related_sorted_indices[idx][:len(neighbor_indices)]
            zero_weights = torch.zeros(len(top_related_indices), device=batch_score.device)
            batch_score[idx].index_add_(0, top_related_indices, zero_weights)
