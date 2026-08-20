"""
Triplet indices used for candidate retrieval (Sec 4.3), filtered ranking
(Sec 7.2), and query-conditioned prototype attention.

`TripletDict` builds three complementary indices over the same triple set:
  - hr2tails[(head_id, relation)] -> {tail_id, ...}       filtered-ranking lookups
  - relation2heads[relation]      -> {head_id, ...}       "which heads have an r-edge?"
    (used to find A_local^(l): relation-r triples whose heads sit at hop l)
  - relation2pairs[relation]      -> [(head_id, tail_id), ...]   A_global(r)
"""

import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def reverse_triplet(triplet: dict) -> dict:
    """Constructs the inverse-relation triplet used for backward (tail->head)
    prediction, matching KGC convention (relation r -> "inverse r")."""
    return {
        "head_id": triplet["tail_id"],
        "head": triplet["tail"],
        "relation": f"inverse {triplet['relation']}",
        "tail_id": triplet["head_id"],
        "tail": triplet["head"],
    }


class TripletDict:
    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        self.hr2tails: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.relation2heads: Dict[str, Set[str]] = defaultdict(set)
        self.relation2pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.relations: Set[str] = set()
        self.triplet_cnt = 0
        self.tail2heads: Dict[str, Set[str]] = defaultdict(set)

        for path in path_list:
            self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        all_examples = list(examples) + [reverse_triplet(ex) for ex in examples]
        for ex in all_examples:
            h, r, t = ex["head_id"], ex["relation"], ex["tail_id"]
            self.relations.add(r)
            self.hr2tails[(h, r)].add(t)
            self.relation2heads[r].add(h)
            self.relation2pairs[r].append((h, t))
            self.tail2heads[t].add(h)
            self.triplet_cnt += 1

    def get_neighbor_tails(self, head_id: str, relation: str) -> Set[str]:
        return self.hr2tails.get((head_id, relation), set())

    def get_relation_heads(self, relation: str) -> Set[str]:
        return self.relation2heads.get(relation, set())

    def get_relation_pairs(self, relation: str) -> List[Tuple[str, str]]:
        return self.relation2pairs.get(relation, [])

    def get_predecessors(self, entity_id: str) -> Set[str]:
        return self.tail2heads.get(entity_id, set())


class RelationVocab:
    """A small learned-embedding-friendly vocabulary over relation strings
    (built from the union of all splits, including inverse relations),
    used wherever the proposal's notation passes `r` explicitly into a
    scoring function (S_A(q, a_i, r, d_i), G_hop(q, r, l), G_lambda(q, r)).
    """

    def __init__(self, relations: Set[str]):
        self.relation2id = {rel: i for i, rel in enumerate(sorted(relations))}
        self.id2relation = {i: rel for rel, i in self.relation2id.items()}
        # reserved id for any relation encountered at inference time that
        # was not seen during vocabulary construction (defensive fallback)
        self.unk_id = len(self.relation2id)

    def __len__(self) -> int:
        return len(self.relation2id) + 1  # + UNK

    def to_id(self, relation: str) -> int:
        return self.relation2id.get(relation, self.unk_id)

    @classmethod
    def from_triplet_dict(cls, triplet_dict: TripletDict) -> "RelationVocab":
        return cls(triplet_dict.relations)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.relation2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "RelationVocab":
        with open(path, "r", encoding="utf-8") as f:
            relation2id = json.load(f)
        vocab = cls(set())
        vocab.relation2id = relation2id
        vocab.id2relation = {i: rel for rel, i in relation2id.items()}
        vocab.unk_id = len(relation2id)
        return vocab
