"""
In-batch negative masking for the contrastive KGC objective (Sec 4.11).

`S(t | h, r)` is computed against every tail present in the current batch
(plus, optionally, pre-batch and self negatives). Some off-diagonal
(query_i, tail_j) pairs may *also* be true triples in the training graph
even though tail_j isn't query_i's labeled target in this batch -- scoring
them as negatives would punish a correct answer. `build_batch_mask` filters
those out, mirroring the standard filtered in-batch-negative protocol used
by contrastive KGC encoders.
"""

from typing import List

import torch

from .data.triplets import TripletDict


def build_batch_mask(head_ids: List[str], relations: List[str], tail_ids: List[str],
                      train_triplet_dict: TripletDict) -> torch.Tensor:
    """Returns a [B, B] bool tensor; True = valid scoring position.
    The diagonal (each query's own labeled target) is always True.
    """
    batch_size = len(head_ids)
    mask = torch.ones(batch_size, batch_size, dtype=torch.bool)

    for i in range(batch_size):
        known_tails = train_triplet_dict.get_neighbor_tails(head_ids[i], relations[i])
        if len(known_tails) <= 1:
            continue  # only the labeled target is known -> nothing extra to mask
        for j in range(batch_size):
            if i == j:
                continue
            if tail_ids[j] in known_tails:
                mask[i, j] = False

    return mask


def build_self_negative_mask(head_ids: List[str], relations: List[str],
                              train_triplet_dict: TripletDict) -> torch.Tensor:
    """Returns a [B] bool tensor; True = the head entity is safe to use as a
    self-negative (i.e. the head is not itself among the known valid tails
    for this (head, relation) pair)."""
    mask = torch.ones(len(head_ids), dtype=torch.bool)
    for i, (h, r) in enumerate(zip(head_ids, relations)):
        if h in train_triplet_dict.get_neighbor_tails(h, r):
            mask[i] = False
    return mask
