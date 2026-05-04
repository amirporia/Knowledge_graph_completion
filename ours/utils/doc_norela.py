import json
import os
from typing import Optional, List, Dict, Any

import torch
import torch.utils.data.dataset

from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from ..setting.config import args
from ..setting.logger_config import logger

# Module-level constants and lazy loading
entity_dict = get_entity_dict()

if args.use_link_graph:
    get_link_graph()  # Trigger lazy data loading


def _custom_tokenize(
        text: str,
        text_pair: Optional[str] = None,
        text_triplet: Optional[str] = None
) -> Dict[str, Any]:
    """Tokenize text with optional triplet information."""
    tokenizer = get_tokenizer()

    if text_triplet:
        full_text = f"{text_pair} [SEP] {text_triplet}"
        return tokenizer(
            text=text,
            text_pair=full_text,
            add_special_tokens=True,
            max_length=args.max_num_tokens,
            return_token_type_ids=True,
            truncation=True
        )

    return tokenizer(
        text=text,
        text_pair=text_pair,
        add_special_tokens=True,
        max_length=args.max_num_tokens,
        return_token_type_ids=True,
        truncation=True
    )


def _parse_entity_name(entity: str) -> str:
    """Parse entity name based on dataset format."""
    if args.task.lower() == 'wn18rr':
        # Example: 'family_alcidae_NN_1' -> 'family alcidae'
        return ' '.join(entity.split('_')[:-2])

    # For wiki5m, some entities may not have names
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    """Concatenate entity name and description."""
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()

    if entity_desc:
        return f'{entity}: {entity_desc}'

    return entity


def get_neighbor_desc(head_id: str, tail_id: Optional[str] = None) -> str:
    """Get neighbor descriptions for an entity, excluding tail to prevent label leakage."""
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)

    # Avoid label leakage during training
    if not args.is_test and tail_id is not None:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]

    entities = [
        _parse_entity_name(entity_dict.get_entity_by_id(n_id).entity)
        for n_id in neighbor_ids
    ]

    return ' '.join(entities)


class Example:
    """Represents a single knowledge graph triplet example."""

    def __init__(self, head_id: str, relation: str, tail_id: str, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self) -> str:
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self) -> str:
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self) -> str:
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self) -> str:
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def _enrich_description(self, desc: str, head_id: str, tail_id: str) -> str:
        """Add neighbor information if description is too short."""
        if not args.use_link_graph:
            return desc

        if len(desc.split()) < 20:
            desc += ' ' + get_neighbor_desc(head_id=head_id, tail_id=tail_id)

        return desc

    def _prepare_entity_text(self, entity: str, desc: str) -> str:
        """Prepare entity text with name and description."""
        word = _parse_entity_name(entity)
        return _concat_name_desc(word, desc)

    def vectorize(self, test: bool = False) -> Dict[str, Any]:
        """Convert example to tokenized vectors."""
        head_desc = self._enrich_description(self.head_desc, self.head_id, self.tail_id)
        tail_desc = self._enrich_description(self.tail_desc, self.tail_id, self.head_id)

        head_text = self._prepare_entity_text(self.head, head_desc)
        tail_text = self._prepare_entity_text(self.tail, tail_desc)

        head_encoded = _custom_tokenize(text=head_text)
        tail_encoded = _custom_tokenize(text=tail_text)

        if test:
            h_triple_encoded = _custom_tokenize(text=head_text, text_pair=self.relation)
        else:
            h_triple_encoded = _custom_tokenize(
                text=head_text,
                text_pair=self.relation,
                text_triplet=tail_text
            )

        return {
            'h_triple_token_ids': h_triple_encoded['input_ids'],
            'h_triple_token_type_ids': h_triple_encoded['token_type_ids'],
            'tail_token_ids': tail_encoded['input_ids'],
            'tail_token_type_ids': tail_encoded['token_type_ids'],
            'head_token_ids': head_encoded['input_ids'],
            'head_token_type_ids': head_encoded['token_type_ids'],
            'obj': self
        }


class Dataset(torch.utils.data.dataset.Dataset):
    """Dataset class for knowledge graph triplets."""

    def __init__(self, path: str, test_set: bool = False, examples: Optional[List[Example]] = None):
        self.path_list = path.split(',')
        self.test_set = test_set

        if not examples:
            assert all(os.path.exists(p) for p in self.path_list), "All paths must exist"

        self.examples = examples or self._load_all_examples()

    def _load_all_examples(self) -> List[Example]:
        """Load examples from all paths."""
        examples = []
        for path in self.path_list:
            data = load_data(path)
            if not examples:
                examples = data
            else:
                examples.extend(data)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _get_filtered_triplets(
            self,
            head_id: str,
            relation: str,
            tail_id: str,
            same_relation: bool
    ) -> List[Example]:
        """Get related or non-related triplets based on filter criteria."""
        filtered = []
        for example in self.examples:
            if example.head_id == head_id:
                relation_match = example.relation == relation
                tail_match = example.tail_id != tail_id

                if same_relation and relation_match and tail_match:
                    filtered.append(example)
                elif not same_relation and not relation_match and tail_match:
                    filtered.append(example)

        return filtered

    def get_related_triplets(
            self,
            head_id: str,
            relation: str,
            tail_id: str
    ) -> List[Example]:
        """Get triplets with same head and relation but different tail."""
        return self._get_filtered_triplets(head_id, relation, tail_id, same_relation=True)

    def get_norelated_triplets(
            self,
            head_id: str,
            relation: str,
            tail_id: str
    ) -> List[Example]:
        """Get triplets with same head but different relation and tail."""
        return self._get_filtered_triplets(head_id, relation, tail_id, same_relation=False)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        example_vectorized = example.vectorize(test=True)

        if self.test_set:
            return example.vectorize(test=True)

        related_triplets = self.get_related_triplets(
            example.head_id,
            example.relation,
            example.tail_id
        )

        # Limit related triplets to max 2
        related_triplets = related_triplets[:2] if len(related_triplets) > 2 else related_triplets

        if not related_triplets:
            related_triplets_vectorized = [example_vectorized]
        else:
            related_triplets_vectorized = [
                triplet.vectorize(test=False)
                for triplet in related_triplets
            ]

        return {
            'example_vectorized': example_vectorized,
            'related_triplets_vectorized': related_triplets_vectorized
        }


def load_data(
        path: str,
        add_forward_triplet: bool = True,
        add_backward_triplet: bool = True
) -> List[Example]:
    """Load and parse triplet data from JSON file."""
    assert path.endswith('.json'), f'Unsupported format: {path}'
    assert add_forward_triplet or add_backward_triplet

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f'Loaded {len(data)} examples from {path}')

    examples = []
    for obj in data:
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))

    return examples


def collate(batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for training batches."""
    # Extract main features
    h_triple_token_ids, h_triple_mask = _to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['h_triple_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    h_triple_token_type_ids = _to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['h_triple_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    tail_token_ids, tail_mask = _to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    tail_token_type_ids = _to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['tail_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    head_token_ids, head_mask = _to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    head_token_type_ids = _to_indices_and_mask(
        [torch.LongTensor(ex['example_vectorized']['head_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    # Extract related triplet features
    related_features = _extract_related_triplet_features(batch_data)

    # Create batch dictionary
    batch_dict = {
        'h_triple_token_ids': h_triple_token_ids,
        'h_triple_mask': h_triple_mask,
        'h_triple_token_type_ids': h_triple_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        **related_features,
        'triplet_mask': construct_mask(
            row_exs=[ex['example_vectorized']['obj'] for ex in batch_data]
        ) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(
            [ex['example_vectorized']['obj'] for ex in batch_data]
        ) if not args.is_test else None,
        'related_triplet_mask': construct_mask(
            row_exs=[ex['related_triplets_vectorized'][0]['obj'] for ex in batch_data]
        ) if not args.is_test else None,
        'test_forward': False,
    }

    return batch_dict


def _extract_related_triplet_features(batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract features from related triplets in batch."""
    related_h_triple_token_ids_list = []
    related_h_triple_mask_list = []
    related_h_triple_token_type_ids_list = []

    for ex in batch_data:
        related_h_triple_token_ids, related_h_triple_mask = _to_indices_and_mask(
            [torch.LongTensor(related_ex['h_triple_token_ids'])
             for related_ex in ex['related_triplets_vectorized']],
            pad_token_id=get_tokenizer().pad_token_id
        )
        related_h_triple_token_type_ids = _to_indices_and_mask(
            [torch.LongTensor(related_ex['h_triple_token_type_ids'])
             for related_ex in ex['related_triplets_vectorized']],
            need_mask=False
        )

        related_h_triple_token_ids_list.append(related_h_triple_token_ids)
        related_h_triple_mask_list.append(related_h_triple_mask)
        related_h_triple_token_type_ids_list.append(related_h_triple_token_type_ids)

    return {
        'related_h_triple_token_ids_list': related_h_triple_token_ids_list,
        'related_h_triple_mask_list': related_h_triple_mask_list,
        'related_h_triple_token_type_ids_list': related_h_triple_token_type_ids_list,
    }


def collate_test(batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for test batches."""
    h_triple_token_ids, h_triple_mask = _to_indices_and_mask(
        [torch.LongTensor(ex['h_triple_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    h_triple_token_type_ids = _to_indices_and_mask(
        [torch.LongTensor(ex['h_triple_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    tail_token_ids, tail_mask = _to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    tail_token_type_ids = _to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    head_token_ids, head_mask = _to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    head_token_type_ids = _to_indices_and_mask(
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
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'test_forward': True,
    }


def _to_indices_and_mask(
        batch_tensor: List[torch.Tensor],
        pad_token_id: int = 0,
        need_mask: bool = True
):
    """Convert list of tensors to padded indices and optional mask."""
    max_len = max(t.size(0) for t in batch_tensor)
    batch_size = len(batch_tensor)

    indices = torch.LongTensor(batch_size, max_len).fill_(pad_token_id)

    if need_mask:
        mask = torch.ByteTensor(batch_size, max_len).fill_(0)

    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)

    if need_mask:
        return indices, mask

    return indices
