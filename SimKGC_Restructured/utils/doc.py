import json
import os
from typing import Optional, List, Tuple

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
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    return tokenizer(
        text=text,
        text_pair=text_pair if text_pair else None,
        add_special_tokens=True,
        max_length=args.max_num_tokens,
        return_token_type_ids=True,
        truncation=True
    )


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1 -> family alcidae
        return ' '.join(entity.split('_')[:-2])
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    """Concatenate entity name and description, avoiding duplication."""
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()

    if entity_desc:
        return f'{entity}: {entity_desc}'

    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    """Get neighbor names (from the link graph) used to enrich a short description."""
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)

    # avoid label leakage during training
    if not args.is_test:
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

    def vectorize(self) -> dict:
        """Convert example to tokenized (hr, tail, head) inputs."""
        head_desc, tail_desc = self.head_desc, self.tail_desc

        # Augment with neighbor names if using link graph
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded_inputs = _custom_tokenize(text=head_text, text_pair=self.relation)

        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        return {
            'hr_token_ids': hr_encoded_inputs['input_ids'],
            'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
            'tail_token_ids': tail_encoded_inputs['input_ids'],
            'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
            'head_token_ids': head_encoded_inputs['input_ids'],
            'head_token_type_ids': head_encoded_inputs['token_type_ids'],
            'obj': self
        }


class Dataset(torch.utils.data.dataset.Dataset):
    """Dataset for knowledge graph completion."""

    def __init__(self, path, examples=None):
        self.path_list = path.split(',')

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
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    """Load examples from a preprocessed JSON file."""
    assert path.endswith('.json'), f'Unsupported format: {path}'
    assert add_forward_triplet or add_backward_triplet
    logger.info(f'In test mode: {args.is_test}')

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f'Load {len(data)} examples from {path}')

    examples = []
    for i in range(len(data)):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    """Convert a list of 1-D tensors to padded indices and an attention mask."""
    max_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, max_len).fill_(pad_token_id)

    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, max_len).fill_(0)

    for i, tensor in enumerate(batch_tensor):
        indices[i, :len(tensor)].copy_(tensor)
        if need_mask:
            mask[i, :len(tensor)].fill_(1)

    if need_mask:
        return indices, mask

    return indices


def _pad_fields(examples: List[dict], prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded (token_ids, mask, token_type_ids) for one field group.

    `prefix` is one of 'hr', 'tail' or 'head'; each example must contain
    '{prefix}_token_ids' and '{prefix}_token_type_ids'.
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
    """Collate function for training / validation / prediction batches."""
    hr_token_ids, hr_mask, hr_token_type_ids = _pad_fields(batch_data, 'hr')
    tail_token_ids, tail_mask, tail_token_type_ids = _pad_fields(batch_data, 'tail')
    head_token_ids, head_mask, head_token_type_ids = _pad_fields(batch_data, 'head')

    batch_exs = [ex['obj'] for ex in batch_data]

    return {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }
