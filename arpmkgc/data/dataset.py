"""
Query and anchor tokenization, PyTorch Dataset, and collate functions.

Encoder usage follows Sec 4.2 literally and *without* the train/test
special-casing baseline dual-encoder KGC pipelines sometimes use:

  - Query encoder:  q  = E0(h, r)          -- NEVER sees the true tail,
                                               at train time or at test time
                                               (Sec 4.12, step 1: "no target
                                               tail is used in ... anchor
                                               selection" generalizes to the
                                               query encoding itself, which
                                               Sec 4.2 defines as a function
                                               of (h, r) only).
  - Anchor encoder: a_i = E0(h_i, r, t_i)  -- always sees the anchor's own
                                               (head, relation, tail); this
                                               is a different training
                                               example than the query, so no
                                               leakage occurs.
  - Entity encoder: e_t = E1(t)            -- entity text alone.

E0 and E1 are two separate encoder towers built in `modules.encoders`; both
are exercised here purely for *tokenization*.
"""

import json
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset as TorchDataset

from .dict_hub import get_entity_dict, get_link_graph, get_relation_vocab, get_tokenizer, \
    get_train_triplet_dict
from .triplets import reverse_triplet
from ..candidates import CandidateAnchor, CandidatePoolBuilder, GLOBAL_HOP
from ..config import ARPMConfig
from ..logging_utils import logger


def load_examples(path: str, add_forward: bool = True, add_backward: bool = True) -> List[dict]:
    assert path.endswith(".json"), f"Unsupported data format: {path}"
    assert add_forward or add_backward
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    logger.info(f"Loaded {len(raw)} raw examples from {path}")

    examples = []
    for obj in raw:
        if add_forward:
            examples.append(obj)
        if add_backward:
            examples.append(reverse_triplet(obj))
    return examples


def _entity_text(entity_name: str, entity_desc: str) -> str:
    entity_name = entity_name or ""
    if entity_desc.startswith(entity_name):
        entity_desc = entity_desc[len(entity_name):].strip()
    return f"{entity_name}: {entity_desc}" if entity_desc else entity_name


class Tokenization:
    """Thin wrapper bundling the tokenizer + entity dict + relation vocab
    needed to turn (head_id, relation, tail_id) tuples into tensors."""

    def __init__(self, config: ARPMConfig):
        self.config = config
        self.tokenizer = get_tokenizer(config)
        self.entity_dict = get_entity_dict(config)
        self.relation_vocab = get_relation_vocab(config)

    def entity_text(self, entity_id: str) -> str:
        if not entity_id:
            return ""
        ex = self.entity_dict.get_by_id(entity_id)
        return _entity_text(ex.entity, ex.entity_desc)

    def encode_query(self, head_id: str, relation: str) -> dict:
        """E0(h, r): head text + relation, no tail."""
        enc = self.tokenizer(
            text=self.entity_text(head_id),
            text_pair=relation,
            add_special_tokens=True,
            max_length=self.config.max_num_tokens,
            truncation=True,
            return_token_type_ids=True,
        )
        return {"input_ids": enc["input_ids"], "token_type_ids": enc["token_type_ids"]}

    def encode_anchor(self, head_id: str, relation: str, tail_id: str) -> dict:
        """E0(h_i, r, t_i): head text + [relation [SEP] tail text]."""
        pair_text = f"{relation} [SEP] {self.entity_text(tail_id)}"
        enc = self.tokenizer(
            text=self.entity_text(head_id),
            text_pair=pair_text,
            add_special_tokens=True,
            max_length=self.config.max_num_tokens,
            truncation=True,
            return_token_type_ids=True,
        )
        return {"input_ids": enc["input_ids"], "token_type_ids": enc["token_type_ids"]}

    def encode_entity(self, entity_id: str) -> dict:
        """E1(t): entity text alone."""
        enc = self.tokenizer(
            text=self.entity_text(entity_id),
            add_special_tokens=True,
            max_length=self.config.max_num_tokens,
            truncation=True,
            return_token_type_ids=True,
        )
        return {"input_ids": enc["input_ids"], "token_type_ids": enc["token_type_ids"]}

    def relation_id(self, relation: str) -> int:
        return self.relation_vocab.to_id(relation)


class ARPMDataset(TorchDataset):
    """Yields, per query example: the tokenized query/head/tail plus a fresh
    (epoch-varying) sample of candidate anchors from `CandidatePoolBuilder`.

    Candidate retrieval always draws from the *training* link graph and
    triplet index (see `data.dict_hub.get_link_graph` /
    `get_train_triplet_dict`) regardless of whether this dataset instance is
    being used for training, validation, or test examples -- Sec 4.12's
    inference procedure mirrors training exactly except that the query's own
    label is never consulted (candidate exclusion only ever drops the
    literal (h, r, t) triple being scored, which is a no-op safety check at
    eval time since eval triples are never members of the training pool).
    """

    def __init__(self, config: ARPMConfig, path: str, tokenization: Optional[Tokenization] = None,
                 seed: Optional[int] = None):
        self.config = config
        self.examples = load_examples(path, add_forward=True, add_backward=True)
        self.tok = tokenization or Tokenization(config)

        triplet_dict = get_train_triplet_dict(config)
        link_graph = get_link_graph(config)
        rng = random.Random(seed)
        self.pool_builder = CandidatePoolBuilder(config, triplet_dict, link_graph, rng=rng)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        head_id, relation, tail_id = ex["head_id"], ex["relation"], ex["tail_id"]

        anchors = self.pool_builder.build(head_id, relation, tail_id)
        item = {
            "head_id": head_id,
            "relation": relation,
            "tail_id": tail_id,
            "relation_id": self.tok.relation_id(relation),
            "query": self.tok.encode_query(head_id, relation),
            "head_entity": self.tok.encode_entity(head_id),
            "tail_entity": self.tok.encode_entity(tail_id),
            "anchors": [self._vectorize_anchor(a, relation) for a in anchors],
        }
        return item

    def _vectorize_anchor(self, anchor: CandidateAnchor, relation: str) -> dict:
        return {
            "token_ids": self.tok.encode_anchor(anchor.head_id, relation, anchor.tail_id),
            "hop": anchor.hop,
            "is_local": anchor.is_local,
            "tail_id": anchor.tail_id,
        }


def _pad_1d(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=1)
    max_len = max(max_len, 1)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) > 0:
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def _pad_encoded(encoded_list: List[dict], pad_id: int) -> dict:
    ids = _pad_1d([e["input_ids"] for e in encoded_list], pad_id)
    type_ids = _pad_1d([e["token_type_ids"] for e in encoded_list], 0)
    # Built from true sequence lengths (not `ids != pad_id`) so a real token
    # that happens to equal pad_id is never mis-masked.
    lengths = [len(e["input_ids"]) for e in encoded_list]
    mask = torch.zeros_like(ids)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": type_ids}


class Collator:
    """Builds padded batch tensors, including the ragged anchor dimension
    (padded to the batch's max anchor count, with an explicit anchor mask
    that every downstream attention/aggregation module respects)."""

    def __init__(self, config: ARPMConfig, tokenization: Optional[Tokenization] = None):
        self.config = config
        self.tok = tokenization or Tokenization(config)
        self.pad_id = self.tok.tokenizer.pad_token_id

    def __call__(self, batch: List[dict]) -> dict:
        query = _pad_encoded([b["query"] for b in batch], self.pad_id)
        head_entity = _pad_encoded([b["head_entity"] for b in batch], self.pad_id)
        tail_entity = _pad_encoded([b["tail_entity"] for b in batch], self.pad_id)
        relation_ids = torch.tensor([b["relation_id"] for b in batch], dtype=torch.long)

        max_anchors = max((len(b["anchors"]) for b in batch), default=0)
        max_anchors = max(max_anchors, 1)  # keep tensors non-degenerate even if a batch has none

        anchor_mask = torch.zeros((len(batch), max_anchors), dtype=torch.bool)
        anchor_hop = torch.zeros((len(batch), max_anchors), dtype=torch.long)
        anchor_is_local = torch.zeros((len(batch), max_anchors), dtype=torch.bool)

        anchor_token_lists: List[List[dict]] = [b["anchors"] for b in batch]
        max_anchor_tok_len = 1
        for anchors in anchor_token_lists:
            for a in anchors:
                max_anchor_tok_len = max(max_anchor_tok_len, len(a["token_ids"]["input_ids"]))

        anchor_ids = torch.full((len(batch), max_anchors, max_anchor_tok_len), self.pad_id, dtype=torch.long)
        anchor_type_ids = torch.zeros((len(batch), max_anchors, max_anchor_tok_len), dtype=torch.long)
        anchor_attn_mask = torch.zeros((len(batch), max_anchors, max_anchor_tok_len), dtype=torch.long)

        for bi, anchors in enumerate(anchor_token_lists):
            for ai, a in enumerate(anchors):
                enc = a["token_ids"]
                length = len(enc["input_ids"])
                anchor_ids[bi, ai, :length] = torch.tensor(enc["input_ids"], dtype=torch.long)
                anchor_type_ids[bi, ai, :length] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
                anchor_attn_mask[bi, ai, :length] = 1
                anchor_mask[bi, ai] = True
                anchor_hop[bi, ai] = a["hop"]
                anchor_is_local[bi, ai] = a["is_local"]

        # Padding anchor slots (anchor_index beyond this row's real anchor count) are
        # left as all-PAD tokens with an all-zero attention mask. Feeding
        # BERT a sequence whose attention mask is entirely zero produces a
        # 0/0 softmax -> NaN, which then survives the `anchor_mask`
        # zeroing-out downstream (NaN * 0 = NaN) and can poison the whole
        # batch's gradients. Forcing position 0 to be attended to sidesteps
        # this: real anchors already have position 0 = 1 (their true first
        # token), so this only changes behavior for pure-padding rows,
        # turning them into a harmless single-PAD-token sequence that is
        # still correctly excluded everywhere via `anchor_mask`.
        anchor_attn_mask[:, :, 0] = 1

        return {
            "head_ids": [b["head_id"] for b in batch],
            "relations": [b["relation"] for b in batch],
            "tail_ids": [b["tail_id"] for b in batch],
            "relation_ids": relation_ids,
            "query": query,
            "head_entity": head_entity,
            "tail_entity": tail_entity,
            "anchor_input_ids": anchor_ids,
            "anchor_token_type_ids": anchor_type_ids,
            "anchor_attention_mask": anchor_attn_mask,
            "anchor_mask": anchor_mask,
            "anchor_hop": anchor_hop,
            "anchor_is_local": anchor_is_local,
            "anchor_tail_ids": [[a["tail_id"] for a in anchors] for anchors in anchor_token_lists],
        }
