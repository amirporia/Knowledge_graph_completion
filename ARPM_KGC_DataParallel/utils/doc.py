import json
import os
from typing import Optional, List, Tuple

import torch
import torch.utils.data.dataset

from .candidate_pool import get_candidate_pool_builder, CandidateAnchor, NO_HOP
from .dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from .triplet import reverse_triplet
from .triplet_mask import construct_mask, construct_self_negative_mask
from ..setting.config import args
from ..setting.logger_config import logger

entity_dict = get_entity_dict()

if args.use_link_graph:
    # Trigger lazy data loading (also used by CandidatePoolBuilder for local anchors)
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

    if not args.is_test and tail_id is not None:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]

    entities = [_parse_entity_name(entity_dict.get_entity_by_id(n_id).entity) for n_id in neighbor_ids]

    return ' '.join(entities)


class Example:
    """Represents a knowledge graph triplet example.

    `vectorize(test=True)` tokenizes (head, relation) ONLY.
     `vectorize(test=False)` tokenizes
    (head, relation, tail) together -- this is exactly the anchor encoding
    a_i = E_0(h_i, r, t_i). Both the query example and every candidate anchor
    reuse the same `hr_bert` encoder (see model/models.py), so E_0 is a single
    shared module applied with two different input compositions.
    """

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


# A single reusable "empty" candidate used to pad every example's candidate list
# up to the batch's max candidate count. Its embedding is always masked out
# downstream (candidate_valid_mask=False), so its exact content never influences
# any score -- it only exists to keep every tensor in a batch the same shape.
_DUMMY_CANDIDATE_VECTORIZED = None


def _get_dummy_candidate_vectorized() -> dict:
    global _DUMMY_CANDIDATE_VECTORIZED
    if _DUMMY_CANDIDATE_VECTORIZED is None:
        dummy = Example(head_id='', relation='', tail_id='')
        _DUMMY_CANDIDATE_VECTORIZED = dummy.vectorize(test=False)
    return _DUMMY_CANDIDATE_VECTORIZED


class Dataset(torch.utils.data.dataset.Dataset):
    """Dataset for knowledge graph completion with ARPM-KGC candidate retrieval."""

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

        # `test_set=True` is only used for entity-only encoding (see
        # evaluation/predict.py::predict_by_entities), which needs no candidates.
        self.pool_builder = None if self.test_set else get_candidate_pool_builder()

    def __len__(self):
        return len(self.examples)

    def _build_candidates(self, example: Example) -> Tuple[List[dict], List[int], List[bool]]:
        """Retrieve A(h,r) for `example` and vectorize every candidate
        as an anchor encoding a_i = E_0(h_i, r, t_i)."""
        candidates: List[CandidateAnchor] = self.pool_builder.build(
            example.head_id, example.relation, example.tail_id
        )

        if not candidates:
            return [], [], []

        candidates_vectorized = []
        hops = []
        is_local = []
        for cand in candidates:
            cand_example = Example(head_id=cand.head_id, relation=cand.relation, tail_id=cand.tail_id)
            candidates_vectorized.append(cand_example.vectorize(test=False))
            hops.append(cand.hop)
            is_local.append(cand.is_local)

        return candidates_vectorized, hops, is_local

    def __getitem__(self, index):
        example = self.examples[index]
        example_vectorized = example.vectorize(test=True)

        if self.test_set:
            return example_vectorized

        candidates_vectorized, hops, is_local = self._build_candidates(example)

        return {
            'example_vectorized': example_vectorized,
            'candidates_vectorized': candidates_vectorized,
            'candidate_hops': hops,
            'candidate_is_local': is_local,
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
    """Build padded (token_ids, mask, token_type_ids) tensors for one field group."""
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
    """Collate function for ARPM-KGC batches (train and eval alike; the only
    difference between the two is whether args.is_test gates triplet_mask /
    self_negative_mask).

    Produces, in addition to the usual (h_triple, tail, head) fields:
      - candidate_token_ids / candidate_mask_tok / candidate_token_type_ids:
        flattened (batch_size * max_candidates, seq_len) token tensors for every
        candidate anchor in the batch (padded slots use the dummy candidate).
      - candidate_valid_mask: (batch_size, max_candidates) bool, True where a
        real candidate anchor exists.
      - candidate_hop_id: (batch_size, max_candidates) long, hop SLOT
        (0-indexed: slot 0 = same head & relation as the query / graph
        distance 0, slots 1..num_hops = increasing graph distance -- e.g.
        --num-hops 2 gives slots {0,1,2}) for local anchors, NO_HOP (-1) for
        global anchors AND for padding slots -- padding must not use 0,
        since that would collide with genuine "local hop 0" (see
        utils/candidate_pool.py).
      - candidate_is_local: (batch_size, max_candidates) bool.
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

    batch_size = len(batch_data)
    max_candidates = max(1, max(len(ex['candidates_vectorized']) for ex in batch_data))

    dummy_candidate = _get_dummy_candidate_vectorized()
    flat_candidates: List[dict] = []
    candidate_valid_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    candidate_hop_id = torch.full((batch_size, max_candidates), NO_HOP, dtype=torch.long)
    candidate_is_local = torch.zeros(batch_size, max_candidates, dtype=torch.bool)

    for b, ex in enumerate(batch_data):
        cands = ex['candidates_vectorized']
        hops = ex['candidate_hops']
        locals_ = ex['candidate_is_local']
        n_real = len(cands)
        for n in range(max_candidates):
            if n < n_real:
                flat_candidates.append(cands[n])
                candidate_valid_mask[b, n] = True
                candidate_hop_id[b, n] = hops[n]
                candidate_is_local[b, n] = locals_[n]
            else:
                flat_candidates.append(dummy_candidate)

    candidate_token_ids, candidate_mask_tok, candidate_token_type_ids = _pad_triple_fields(
        flat_candidates, 'h_triple'
    )

    batch_exs = [ex['obj'] for ex in example_vecs]

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
        'candidate_token_ids': candidate_token_ids,
        'candidate_mask_tok': candidate_mask_tok,
        'candidate_token_type_ids': candidate_token_type_ids,
        'candidate_valid_mask': candidate_valid_mask,
        'candidate_hop_id': candidate_hop_id,
        'candidate_is_local': candidate_is_local,
        'max_candidates': max_candidates,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'batch_data': batch_exs,
        'test_forward': False,
    }


def collate_entity(batch_data: List[dict]) -> dict:
    """Collate function for entity-only encoding (Dataset(test_set=True)); used
    solely to compute E_1(t) for every entity in the dictionary` -- no candidates / query encoding required."""
    tail_token_ids, tail_mask, tail_token_type_ids = _pad_triple_fields(batch_data, 'tail')

    return {
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'only_ent_embedding': True,
    }
