import json
import os
import random
import zlib
from typing import Optional, List, Tuple

import torch
import torch.utils.data.dataset

from .dict_hub import get_entity_dict, get_link_graph, get_tokenizer, get_train_triplet_dict
from .triplet import reverse_triplet
from .triplet_mask import construct_mask, construct_self_negative_mask
from ..setting.config import args, MAX_ANCHORS
from ..setting.logger_config import logger

entity_dict = get_entity_dict()
# Anchors are ALWAYS drawn from the training graph G ∪ G_inv (paper, Definition 1).
# Loaded here (before DataLoader workers fork) so every worker shares the same copy.
train_triplet_dict = get_train_triplet_dict()

if args.use_link_graph:
    # Trigger lazy data loading
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None,
                     text_triplet: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()

    if text_triplet:
        full_text = f"{text_pair} [SEP] {text_triplet}"
        encoded_inputs = tokenizer(
            text=text,
            text_pair=full_text,
            add_special_tokens=True,
            max_length=args.max_num_tokens,
            return_token_type_ids=True,
            truncation=True
        )
    else:
        encoded_inputs = tokenizer(
            text=text,
            text_pair=text_pair if text_pair else None,
            add_special_tokens=True,
            max_length=args.max_num_tokens,
            return_token_type_ids=True,
            truncation=True
        )

    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    """Parse entity name, handling entities without names."""
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    """Concatenate entity name and description, avoiding duplication."""
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()

    if entity_desc:
        return f'{entity}: {entity_desc}'

    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    """Get neighbor descriptions for a given entity."""
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)

    # Avoid label leakage during training
    if not args.is_test and tail_id is not None:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]

    entities = [_parse_entity_name(entity_dict.get_entity_by_id(n_id).entity) for n_id in neighbor_ids]

    return ' '.join(entities)


# ---------------------------------------------------------------------------
# Relation-aware anchor generation (paper Definition 1, Eq. 1)
# ---------------------------------------------------------------------------

def sample_anchors(head_id: str,
                   relation: str,
                   exclude_tail_id: Optional[str] = None,
                   k: Optional[int] = None) -> List[str]:
    """T_k = random(T, k), k <= K = 5.

    T is the set of tail entities connected to `head_id` through `relation` in the training
    graph G ∪ G_inv.  During training the target tail is removed from T to avoid label leakage.
    At inference the sample is made deterministic (seeded by the query) so that evaluation is
    reproducible.  Returns [] when T is empty (the model then falls back to the classic query).
    """
    k = args.anchor_num if k is None else k
    k = min(k, MAX_ANCHORS)
    if k <= 0 or not head_id or not relation:
        return []

    pool = train_triplet_dict.get_neighbors(head_id, relation)
    pool = [t for t in pool if t != exclude_tail_id and t != head_id]
    if not pool:
        return []

    k = min(k, len(pool))
    return random.sample(pool, k)


class Example:
    """Represents a knowledge graph triplet example."""

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        if not self.tail_id:
            return ''
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        if not self.tail_id:
            return ''
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self, test=False) -> dict:
        """Convert example to tokenized tensors (classic query I_hr, head, and tail inputs)."""
        head_desc, tail_desc = self.head_desc, self.tail_desc

        # Augment with neighbor descriptions if using link graph
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(
                    head_id=self.head_id, tail_id=self.tail_id
                )
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(
                    head_id=self.tail_id, tail_id=self.head_id
                )

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_text = _concat_name_desc(tail_word, tail_desc)
        tail_encoded_inputs = _custom_tokenize(text=tail_text)

        if test:
            h_triple_encoded_inputs = _custom_tokenize(
                text=head_text, text_pair=self.relation
            )
        else:
            h_triple_encoded_inputs = _custom_tokenize(
                text=head_text,
                text_pair=self.relation,
                text_triplet=tail_text
            )

        return {
            'h_triple_token_ids': h_triple_encoded_inputs['input_ids'],
            'h_triple_token_type_ids': h_triple_encoded_inputs['token_type_ids'],
            'tail_token_ids': tail_encoded_inputs['input_ids'],
            'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
            'head_token_ids': head_encoded_inputs['input_ids'],
            'head_token_type_ids': head_encoded_inputs['token_type_ids'],
            'obj': self
        }

    def vectorize_anchor_query(self, anchor_id: str) -> dict:
        """Anchor-enhanced query I_hrta (Eq. 2) for ONE anchor t_i:

            [CLS] h_a : h_adesc [SEP] r_a [SEP] t_i : t_idesc [SEP]
        """
        head_desc = self.head_desc
        anchor_desc = entity_dict.get_entity_by_id(anchor_id).entity_desc

        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(anchor_desc.split()) < 20:
                anchor_desc += ' ' + get_neighbor_desc(head_id=anchor_id, tail_id=self.head_id)

        head_text = _concat_name_desc(_parse_entity_name(self.head), head_desc)
        anchor_name = _parse_entity_name(entity_dict.get_entity_by_id(anchor_id).entity)
        anchor_text = _concat_name_desc(anchor_name, anchor_desc)

        encoded = _custom_tokenize(
            text=head_text, text_pair=self.relation, text_triplet=anchor_text
        )
        return {
            'h_triple_token_ids': encoded['input_ids'],
            'h_triple_token_type_ids': encoded['token_type_ids'],
            'obj': Example(head_id=self.head_id, relation=self.relation, tail_id=anchor_id),
        }


class Dataset(torch.utils.data.dataset.Dataset):
    """Dataset for knowledge graph completion."""

    def __init__(self, path, test_set=False, examples=None):
        self.path_list = path.split(',')
        self.test_set = test_set

        assert all(os.path.exists(p) for p in self.path_list) or examples

        if examples:
            self.examples = examples
        else:
            self.examples = []
            for file_path in self.path_list:
                if not self.examples:
                    self.examples = load_data(file_path)
                else:
                    self.examples.extend(load_data(file_path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        example_vectorized = example.vectorize(test=True)

        if self.test_set:
            return example_vectorized

        # Never use the target to build anchors at inference; remove it during training (no leakage).
        exclude_tail_id = None if args.is_test else example.tail_id

        anchor_ids = sample_anchors(
            example.head_id, example.relation,
            exclude_tail_id=exclude_tail_id,
        )
        return {
            'example_vectorized': example_vectorized,
            'anchor_queries_vectorized': [example.vectorize_anchor_query(a) for a in anchor_ids],
        }


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    """Load examples from JSON file."""
    assert path.endswith('.json'), f'Unsupported format: {path}'
    assert add_forward_triplet or add_backward_triplet

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info(f'Loaded {len(data)} examples from {path}')

    examples = []
    for obj in data:
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))

    return examples


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    """Convert batch of tensors to padded indices and attention mask."""
    max_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, max_len).fill_(pad_token_id)

    if need_mask:
        mask = torch.ByteTensor(batch_size, max_len).fill_(0)

    for i, tensor in enumerate(batch_tensor):
        indices[i, :len(tensor)].copy_(tensor)
        if need_mask:
            mask[i, :len(tensor)].fill_(1)

    if need_mask:
        return indices, mask

    return indices


def _pad_triple_fields(examples: List[dict], prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded (token_ids, mask, token_type_ids) tensors for one field
    group, e.g. prefix='h_triple', 'tail', or 'head'.
    """
    token_ids, mask = to_indices_and_mask(
        [torch.LongTensor(ex[f'{prefix}_token_ids']) for ex in examples],
        pad_token_id=get_tokenizer().pad_token_id
    )
    token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex[f'{prefix}_token_type_ids']) for ex in examples],
        need_mask=False
    )
    return token_ids, mask, token_type_ids


def collate(batch_data: List[dict]) -> dict:
    """Collate function for training / validation batches.

    All anchor-enhanced queries of the batch are flattened into ONE padded tensor so BERT is
    run once; `anchor_group_ids[j]` tells which batch example anchor query j belongs to.
    """
    example_vecs = [ex['example_vectorized'] for ex in batch_data]

    h_triple_token_ids, h_triple_mask, h_triple_token_type_ids = _pad_triple_fields(
        example_vecs, 'h_triple'
    )
    tail_token_ids, tail_mask, tail_token_type_ids = _pad_triple_fields(
        example_vecs, 'tail'
    )
    head_token_ids, head_mask, head_token_type_ids = _pad_triple_fields(
        example_vecs, 'head'
    )

    anchor_vecs, anchor_group_ids = [], []
    for i, ex in enumerate(batch_data):
        for anchor_vec in ex['anchor_queries_vectorized']:
            anchor_vecs.append(anchor_vec)
            anchor_group_ids.append(i)

    if anchor_vecs:
        anchor_token_ids, anchor_mask, anchor_token_type_ids = _pad_triple_fields(anchor_vecs, 'h_triple')
        anchor_group_ids = torch.LongTensor(anchor_group_ids)
    else:  # no example in this batch has anchors -> model falls back to the classic query
        anchor_token_ids = anchor_mask = anchor_token_type_ids = anchor_group_ids = None

    batch_exs = [ex['obj'] for ex in example_vecs]
    # first anchor example per query (falls back to the example itself when it has no anchor)
    related_batch_exs = [
        (ex['anchor_queries_vectorized'][0]['obj'] if ex['anchor_queries_vectorized']
         else ex['example_vectorized']['obj'])
        for ex in batch_data
    ]

    return {
        'h_triple_token_ids': h_triple_token_ids,
        'h_triple_mask': h_triple_mask,
        'h_triple_token_type_ids': h_triple_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'anchor_token_ids': anchor_token_ids,
        'anchor_mask': anchor_mask,
        'anchor_token_type_ids': anchor_token_type_ids,
        'anchor_group_ids': anchor_group_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'related_triplet_mask': construct_mask(row_exs=related_batch_exs) if not args.is_test else None,
        'test_forward': False,
    }


def collate_test(batch_data: List[dict]) -> dict:
    """Collate function for entity-embedding batches."""
    h_triple_token_ids, h_triple_mask, h_triple_token_type_ids = _pad_triple_fields(
        batch_data, 'h_triple'
    )
    tail_token_ids, tail_mask, tail_token_type_ids = _pad_triple_fields(
        batch_data, 'tail'
    )
    head_token_ids, head_mask, head_token_type_ids = _pad_triple_fields(
        batch_data, 'head'
    )

    batch_exs = [ex['obj'] for ex in batch_data]

    return {
        'h_triple_token_ids': h_triple_token_ids,
        'h_triple_mask': h_triple_mask,
        'h_triple_token_type_ids': h_triple_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'test_forward': True,
    }