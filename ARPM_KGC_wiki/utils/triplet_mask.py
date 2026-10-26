from typing import List, Optional

import torch

from .dict_hub import get_train_triplet_dict, get_entity_dict, EntityDict, TripletDict
from ..setting.config import args

entity_dict: EntityDict = get_entity_dict()
train_triplet_dict: Optional[TripletDict] = (
    get_train_triplet_dict() if not args.is_test else None
)


def construct_mask(
        row_exs: List,
        col_exs: Optional[List] = None
) -> torch.Tensor:
    """Construct a mask tensor for triplet filtering.

    Used to suppress in-batch false negatives -- entities that are actually valid
    tails for a given (head, relation) but happen to land in another row's column
    during in-batch contrastive training. ARPM-KGC reuses this identical mask for
    all three in-batch losses (L_query, L_proto, L_struct), since a false negative
    would otherwise corrupt every one of them, not just the S_q term.

    Args:
        row_exs: List of examples for rows
        col_exs: List of examples for columns (uses row_exs if None)

    Returns:
        Boolean mask tensor of shape (num_row, num_col)
    """
    positive_on_diagonal = col_exs is None
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs)

    row_entity_ids = torch.LongTensor([
        entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs
    ])

    if positive_on_diagonal:
        col_entity_ids = row_entity_ids
    else:
        col_entity_ids = torch.LongTensor([
            entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs
        ])

    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))

    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)

    _mask_known_neighbors(
        triplet_mask, row_exs, col_exs, num_row, num_col, positive_on_diagonal
    )

    return triplet_mask


def _mask_known_neighbors(
        mask,
        row_exs: List,
        col_exs: List,
        num_row: int,
        num_col: int,
        positive_on_diagonal: bool
) -> None:
    """Mask out triplets that exist in the training set."""
    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)

        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue

            tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                mask[i][j] = False


def construct_self_negative_mask(exs: List) -> torch.Tensor:
    """Create a mask identifying samples that have self as a valid neighbor.

    Returns True for samples where the head entity is NOT a valid neighbor.
    """
    mask = torch.ones(len(exs))

    for idx, ex in enumerate(exs):
        head_id, relation = ex.head_id, ex.relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)

        if head_id in neighbor_ids:
            mask[idx] = 0

    return mask.bool()
