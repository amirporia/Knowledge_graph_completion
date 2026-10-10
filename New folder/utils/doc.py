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
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None,
                     text_triplet: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()

    if text_triplet:
        full_text = f"{text_pair} [SEP] {text_triplet}"
        return tokenizer(text=text, text_pair=full_text, add_special_tokens=True,
                         max_length=args.max_num_tokens, return_token_type_ids=True, truncation=True)
    return tokenizer(text=text, text_pair=text_pair if text_pair else None, add_special_tokens=True,
                     max_length=args.max_num_tokens, return_token_type_ids=True, truncation=True)


def _parse_entity_name(entity: str) -> str:
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return f'{entity}: {entity_desc}'
    return entity


def get_neighbor_desc(entity_id: str, exclude_ids=frozenset()) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(entity_id)
    if not args.is_test and exclude_ids:
        neighbor_ids = [n for n in neighbor_ids if n not in exclude_ids]
    return ' '.join(_parse_entity_name(entity_dict.get_entity_by_id(n).entity) for n in neighbor_ids)


class Example:
    """A knowledge-graph triplet.

    `hidden_ids` are entities that must never appear in neighbour-text augmentation
    (for candidate anchors: the gold tail of the query that spawned them)."""

    def __init__(self, head_id, relation, tail_id, hidden_ids=None, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        self.hidden_ids = frozenset(hidden_ids) if hidden_ids else frozenset()

    @property
    def head_desc(self):
        return entity_dict.get_entity_by_id(self.head_id).entity_desc if self.head_id else ''

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc if self.tail_id else ''

    @property
    def head(self):
        return entity_dict.get_entity_by_id(self.head_id).entity if self.head_id else ''

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity if self.tail_id else ''

    def _tail_text(self) -> str:
        tail_desc = self.tail_desc
        if args.use_link_graph and len(tail_desc.split()) < 20:
            tail_desc += ' ' + get_neighbor_desc(self.tail_id, self.hidden_ids | {self.head_id})
        return _concat_name_desc(_parse_entity_name(self.tail), tail_desc)

    def vectorize(self, test=False) -> dict:
        """test=True: (head, relation) query; test=False: (head, relation, tail) triple."""
        head_desc = self.head_desc
        if args.use_link_graph and len(head_desc.split()) < 20:
            head_desc += ' ' + get_neighbor_desc(self.head_id, self.hidden_ids | {self.tail_id})

        head_text = _concat_name_desc(_parse_entity_name(self.head), head_desc)
        tail_text = self._tail_text()

        head_enc = _custom_tokenize(text=head_text)
        tail_enc = _custom_tokenize(text=tail_text)

        if test:
            triple_enc = _custom_tokenize(text=head_text, text_pair=self.relation)
        else:
            triple_enc = _custom_tokenize(text=head_text, text_pair=self.relation, text_triplet=tail_text)

        return {
            'h_triple_token_ids': triple_enc['input_ids'],
            'h_triple_token_type_ids': triple_enc['token_type_ids'],
            'tail_token_ids': tail_enc['input_ids'],
            'tail_token_type_ids': tail_enc['token_type_ids'],
            'head_token_ids': head_enc['input_ids'],
            'head_token_type_ids': head_enc['token_type_ids'],
            'obj': self
        }

    def vectorize_tail(self) -> dict:
        """Tail text only (what the entity encoder E_1 sees). Used for memory candidates."""
        enc = _custom_tokenize(text=self._tail_text())
        return {'ids': enc['input_ids'], 'tt': enc['token_type_ids']}


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path, test_set=False, examples=None):
        self.path_list = path.split(',')
        self.test_set = test_set

        assert all(os.path.exists(p) for p in self.path_list) or examples

        if examples:
            self.examples = examples
        else:
            self.examples = []
            for file_path in self.path_list:
                self.examples.extend(load_data(file_path))

        # test_set=True is only used for entity-only encoding and needs no candidates.
        self.pool_builder = None if self.test_set else get_candidate_pool_builder()

    def __len__(self):
        return len(self.examples)

    def _build_candidates(self, example: Example) -> dict:
        """Retrieve A(h,r). Every candidate yields its TAIL tokens (memory embedding E_1(t_i));
        the first `anchor_num` hop-0 candidates additionally yield the (h, r, t_i) triple tokens
        for the RAA-KGC anchor-enhanced query path."""
        candidates: List[CandidateAnchor] = self.pool_builder.build(
            example.head_id, example.relation, example.tail_id
        )
        hidden = None if args.is_test else {example.tail_id}

        cand_tails, hops, is_local, anchors = [], [], [], []
        for cand in candidates:
            ce = Example(head_id=cand.head_id, relation=cand.relation,
                         tail_id=cand.tail_id, hidden_ids=hidden)
            if cand.is_local and cand.hop == 0 and len(anchors) < args.anchor_num:
                vec = ce.vectorize(test=False)
                anchors.append({'ids': vec['h_triple_token_ids'], 'tt': vec['h_triple_token_type_ids']})
                cand_tails.append({'ids': vec['tail_token_ids'], 'tt': vec['tail_token_type_ids']})
            else:
                cand_tails.append(ce.vectorize_tail())
            hops.append(cand.hop)
            is_local.append(cand.is_local)

        return {'cand_tails': cand_tails, 'cand_hops': hops, 'cand_is_local': is_local, 'anchors': anchors}

    def __getitem__(self, index):
        example = self.examples[index]
        example_vectorized = example.vectorize(test=True)

        if self.test_set:
            return example_vectorized

        item = self._build_candidates(example)
        item['example_vectorized'] = example_vectorized
        return item


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
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
    token_ids, mask = to_indices_and_mask(
        [torch.LongTensor(ex[f'{prefix}_token_ids']) for ex in examples],
        pad_token_id=get_tokenizer().pad_token_id
    )
    token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex[f'{prefix}_token_type_ids']) for ex in examples],
        need_mask=False
    )
    return token_ids, mask, token_type_ids


def _pad_nested(groups: List[List[dict]]):
    """Pad a list (batch) of lists (slots) of {'ids','tt'} into BATCH-FIRST tensors
    (B, N, L) so nn.DataParallel's dim-0 scatter keeps every example's slots together.
    Returns ids, mask, token_type_ids, valid (B, N)."""
    pad_id = get_tokenizer().pad_token_id
    batch_size = len(groups)
    max_n = max(1, max(len(g) for g in groups))
    max_len = max([len(it['ids']) for g in groups for it in g] or [2])

    ids = torch.full((batch_size, max_n, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(batch_size, max_n, max_len, dtype=torch.long)
    tt = torch.zeros(batch_size, max_n, max_len, dtype=torch.long)
    valid = torch.zeros(batch_size, max_n, dtype=torch.bool)

    for b, g in enumerate(groups):
        for n, it in enumerate(g):
            length = len(it['ids'])
            ids[b, n, :length] = torch.LongTensor(it['ids'])
            tt[b, n, :length] = torch.LongTensor(it['tt'])
            mask[b, n, :length] = 1
            valid[b, n] = True
    return ids, mask, tt, valid


def collate(batch_data: List[dict]) -> dict:
    """Batch-first collate. In addition to (h_triple, tail, head):
      cand_tail_*        (B, N, L)  tail tokens of every memory candidate
      candidate_valid_mask / candidate_hop_id / candidate_is_local  (B, N)
          hop slot: 0 = same (h,r), 1..num_hops = graph distance, NO_HOP(-1) = global / padding
      anchor_*           (B, Ka, L) hop-0 (h, r, t_i) triples for the RAA anchor-enhanced query
    """
    example_vecs = [ex['example_vectorized'] for ex in batch_data]

    h_triple_token_ids, h_triple_mask, h_triple_token_type_ids = _pad_triple_fields(example_vecs, 'h_triple')
    tail_token_ids, tail_mask, tail_token_type_ids = _pad_triple_fields(example_vecs, 'tail')
    head_token_ids, head_mask, head_token_type_ids = _pad_triple_fields(example_vecs, 'head')

    cand_ids, cand_mask, cand_tt, cand_valid = _pad_nested([ex['cand_tails'] for ex in batch_data])
    anc_ids, anc_mask, anc_tt, anc_valid = _pad_nested([ex['anchors'] for ex in batch_data])

    batch_size, max_candidates = cand_valid.shape
    candidate_hop_id = torch.full((batch_size, max_candidates), NO_HOP, dtype=torch.long)
    candidate_is_local = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    for b, ex in enumerate(batch_data):
        for n, (hop, loc) in enumerate(zip(ex['cand_hops'], ex['cand_is_local'])):
            candidate_hop_id[b, n] = hop
            candidate_is_local[b, n] = loc

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
        'cand_tail_token_ids': cand_ids,
        'cand_tail_mask': cand_mask,
        'cand_tail_token_type_ids': cand_tt,
        'candidate_valid_mask': cand_valid,
        'candidate_hop_id': candidate_hop_id,
        'candidate_is_local': candidate_is_local,
        'anchor_token_ids': anc_ids,
        'anchor_mask': anc_mask,
        'anchor_token_type_ids': anc_tt,
        'anchor_valid': anc_valid,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'batch_data': batch_exs,
        'test_forward': False,
    }


def collate_entity(batch_data: List[dict]) -> dict:
    tail_token_ids, tail_mask, tail_token_type_ids = _pad_triple_fields(batch_data, 'tail')
    return {
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'only_ent_embedding': True,
    }
