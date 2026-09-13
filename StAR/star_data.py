import random
from typing import List

import torch
import torch.utils.data.dataset

from config import args
from doc import Example, load_data, to_indices_and_mask
from dict_hub import get_entity_dict, get_train_triplet_dict, get_tokenizer

entity_dict = get_entity_dict()
train_triplet_dict = get_train_triplet_dict() if not args.is_test else None


def _sample_negatives(example: Example, num_negatives: int) -> List[Example]:
    """StAR §3.3.1: for a positive triple tp=(h,r,t), generate tp' by replacing either
    the head or the tail with a uniformly random entity, subject to tp' not already
    being a true triple in the graph. TripletDict stores both the relation and its
    inverse (see triplet.py::reverse_triplet), so `get_neighbors(t, 'inverse r')`
    gives the true heads for (?, r, t) — letting us filter head-corruptions the same
    way tail-corruptions are filtered.
    """
    negatives = []
    n_entities = len(entity_dict)
    known_tails = (train_triplet_dict.get_neighbors(example.head_id, example.relation)
                  if train_triplet_dict else set())
    known_heads = (train_triplet_dict.get_neighbors(example.tail_id, f'inverse {example.relation}')
                  if train_triplet_dict else set())

    attempts = 0
    max_attempts = max(50, num_negatives * 20)
    while len(negatives) < num_negatives and attempts < max_attempts:
        attempts += 1
        rand_ent = entity_dict.get_entity_by_idx(random.randrange(n_entities)).entity_id
        if random.random() < 0.5:
            if rand_ent == example.tail_id or rand_ent in known_tails:
                continue
            negatives.append(Example(head_id=example.head_id, relation=example.relation, tail_id=rand_ent))
        else:
            if rand_ent == example.head_id or rand_ent in known_heads:
                continue
            negatives.append(Example(head_id=rand_ent, relation=example.relation, tail_id=example.tail_id))

    # Extremely small graphs might not yield enough filtered negatives in time;
    # fall back to unfiltered tail corruption to guarantee a fixed batch shape.
    while len(negatives) < num_negatives:
        rand_ent = entity_dict.get_entity_by_idx(random.randrange(n_entities)).entity_id
        negatives.append(Example(head_id=example.head_id, relation=example.relation, tail_id=rand_ent))

    return negatives


class StarDataset(torch.utils.data.dataset.Dataset):
    """Each item is one positive triple plus `num_negatives` freshly-sampled
    negatives (resampled every epoch, since this is a plain map-style Dataset
    re-iterated by a DataLoader — matching the paper's per-epoch resampling).
    """

    def __init__(self, path: str, num_negatives: int):
        self.examples = load_data(path)
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        negatives = _sample_negatives(example, self.num_negatives)
        return {
            'pos': example.vectorize(),
            'negs': [neg.vectorize() for neg in negatives],
        }


def collate_star(batch: List[dict]) -> dict:
    """Flattens each (1 positive + K negative) group into a single padded batch of
    size B*(1+K), in group order [pos_0, neg_0_0, ..., neg_0_{K-1}, pos_1, ...].
    `batch_size`/`num_negatives` let the trainer reshape back into (B, 1+K, dim)
    after encoding to compute the per-group classification/hinge losses.
    """
    num_negatives = len(batch[0]['negs'])
    flat = []
    for item in batch:
        flat.append(item['pos'])
        flat.extend(item['negs'])

    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in flat],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in flat], need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in flat],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in flat], need_mask=False)

    return {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'batch_size': len(batch),
        'num_negatives': num_negatives,
    }
