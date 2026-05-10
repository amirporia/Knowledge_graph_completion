import json
import os
from typing import Optional, List

import torch
import torch.utils.data.dataset

from .dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from .triplet import reverse_triplet
from .triplet_mask import construct_mask, construct_self_negative_mask
from ..setting.config import args
from ..setting.logger_config import logger

entity_dict = get_entity_dict()

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

    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]

    return ' '.join(entities)


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
        """Convert example to tokenized tensors."""
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

    def get_related_triplets(self, relation, tail_id):
        """Get triplets with same relation but different tail."""
        related_triplets = [
            ex for ex in self.examples
            if ex.relation == relation and ex.tail_id != tail_id
        ]
        return related_triplets

    def __getitem__(self, index):
        example = self.examples[index]
        example_vectorized = example.vectorize(test=True)

        if self.test_set:
            return example.vectorize(test=True)

        related_triplets = self.get_related_triplets(
            example.relation, example.tail_id
        )

        if len(related_triplets) == 0:
            return {
                'example_vectorized': example_vectorized,
                'related_triplets_vectorized': [example_vectorized]
            }

        # Limit to 3 related triplets
        if len(related_triplets) > 3:
            related_triplets = related_triplets[:3]

        related_triplets_vectorized = [
            triplet.vectorize(test=False) for triplet in related_triplets
        ]

        return {
            'example_vectorized': example_vectorized,
            'related_triplets_vectorized': related_triplets_vectorized
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


def collate(batch_data: List[dict]) -> dict:
    """Collate function for training batches."""
    # Extract and pad tensors for example data
    h_triple_token_ids, h_triple_mask = to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['h_triple_token_ids'])
         for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    h_triple_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['h_triple_token_type_ids'])
         for ex in batch_data],
        need_mask=False
    )

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['tail_token_ids'])
         for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['tail_token_type_ids'])
         for ex in batch_data],
        need_mask=False
    )

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['head_token_ids'])
         for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['head_token_type_ids'])
         for ex in batch_data],
        need_mask=False
    )

    # Process related triplets
    related_h_triple_token_ids_list = []
    related_h_triple_token_type_ids_list = []
    related_h_triple_mask_list = []

    for ex in batch_data:
        related_h_triple_token_ids, related_h_triple_mask = to_indices_and_mask(
            [torch.LongTensor(related_ex['h_triple_token_ids'])
             for related_ex in ex['related_triplets_vectorized']],
            pad_token_id=get_tokenizer().pad_token_id
        )
        related_h_triple_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(related_ex['h_triple_token_type_ids'])
             for related_ex in ex['related_triplets_vectorized']],
            need_mask=False
        )

        related_h_triple_token_ids_list.append(related_h_triple_token_ids)
        related_h_triple_mask_list.append(related_h_triple_mask)
        related_h_triple_token_type_ids_list.append(related_h_triple_token_type_ids)

    batch_exs = [ex['example_vectorized']['obj'] for ex in batch_data]
    batch_exs_list = [ex['related_triplets_vectorized'][0]['obj'] for ex in batch_data]

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
        'related_h_triple_token_ids_list': related_h_triple_token_ids_list,
        'related_h_triple_mask_list': related_h_triple_mask_list,
        'related_h_triple_token_type_ids_list': related_h_triple_token_type_ids_list,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs)
        if not args.is_test else None,
        'related_triplet_mask': construct_mask(row_exs=batch_exs_list)
        if not args.is_test else None,
        'test_forward': False,
    }


def collate_test(batch_data: List[dict]) -> dict:
    """Collate function for test batches."""
    h_triple_token_ids, h_triple_mask = to_indices_and_mask(
        [torch.LongTensor(ex['h_triple_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    h_triple_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['h_triple_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False
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
        'related_h_triple_token_ids_list': None,
        'related_h_triple_mask_list': None,
        'related_h_triple_token_type_ids_list': None,
        'related_head_token_ids_list': None,
        'related_head_token_type_ids_list': None,
        'related_head_mask_list': None,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs)
        if not args.is_test else None,
        'test_forward': True,
    }


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
    else:
        return indices
