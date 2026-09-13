import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class KGIndex:
    """Entity / relation vocabularies shared by every embedding baseline.

    Relations are augmented with an inverse copy (r_inv_id = r_id + num_base_relations),
    exactly the trick already used by Baseline/utils/triplet.py::reverse_triplet and
    SimKGC/triplet.py for the text-based models. This lets every model answer both
    (h, r, ?) and (?, r, t) queries as a single "predict tail" call, which keeps the
    training loop, negative sampler and evaluator identical across all 7 baselines.
    """
    entity2id: Dict[str, int]
    id2entity: List[str]
    relation2id: Dict[str, int]
    id2relation: List[str]
    num_base_relations: int

    @property
    def num_entities(self) -> int:
        return len(self.id2entity)

    @property
    def num_relations(self) -> int:
        # includes inverse relations
        return len(self.id2relation)

    def inverse_relation_id(self, rel_id: int) -> int:
        if rel_id < self.num_base_relations:
            return rel_id + self.num_base_relations
        return rel_id - self.num_base_relations


def _load_json(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_kg_index(entities_path: str, triple_paths: List[str]) -> KGIndex:
    """Build entity/relation vocabularies from entities.json plus every triple file
    (train/valid/test) so relation ids are stable regardless of split.
    """
    entities = _load_json(entities_path)
    id2entity = [e['entity_id'] for e in entities]
    entity2id = {eid: i for i, eid in enumerate(id2entity)}

    base_relations: Set[str] = set()
    for path in triple_paths:
        for ex in _load_json(path):
            base_relations.add(ex['relation'])
    id2relation_base = sorted(base_relations)
    relation2id = {r: i for i, r in enumerate(id2relation_base)}
    num_base = len(id2relation_base)

    id2relation = list(id2relation_base) + [f'inverse {r}' for r in id2relation_base]
    for r in id2relation_base:
        relation2id[f'inverse {r}'] = relation2id[r] + num_base

    logger.info(f'KG index: {len(id2entity)} entities, {num_base} base relations '
                f'({2 * num_base} with inverses)')
    return KGIndex(entity2id=entity2id, id2entity=id2entity,
                   relation2id=relation2id, id2relation=id2relation,
                   num_base_relations=num_base)


def load_triples(path: str, kg_index: KGIndex, add_inverse: bool = True) -> torch.LongTensor:
    """Load a *.txt.json triple file into an (N, 3) LongTensor of (h_id, r_id, t_id).
    When add_inverse=True every triple (h, r, t) also yields (t, r_inv, h), matching
    the augmented relation vocabulary above.
    """
    examples = _load_json(path)
    rows = []
    for ex in examples:
        h = kg_index.entity2id[ex['head_id']]
        t = kg_index.entity2id[ex['tail_id']]
        r = kg_index.relation2id[ex['relation']]
        rows.append((h, r, t))
        if add_inverse:
            r_inv = kg_index.relation2id.get(f"inverse {ex['relation']}")
            rows.append((t, r_inv, h))
    return torch.tensor(rows, dtype=torch.long)


def build_true_tail_filter(*triple_tensors: torch.LongTensor) -> Dict[Tuple[int, int], Set[int]]:
    """(h, r) -> set of all known true tails, across every split given. Used both for
    filtered negative sampling during training and filtered ranking at eval time.
    """
    filt: Dict[Tuple[int, int], Set[int]] = {}
    for triples in triple_tensors:
        for h, r, t in triples.tolist():
            filt.setdefault((h, r), set()).add(t)
    return filt


class NegSamplingDataset(Dataset):
    """For each positive (h, r, t), sample `neg_size` corrupted tails, filtered against
    every known true tail for that (h, r) so we don't accidentally sample a false negative.
    Because relations are already inverse-augmented, corrupting only the tail is enough to
    cover both head- and tail-prediction directions.
    """

    def __init__(self, triples: torch.LongTensor, num_entities: int, neg_size: int,
                 true_tail_filter: Dict[Tuple[int, int], Set[int]]):
        self.triples = triples
        self.num_entities = num_entities
        self.neg_size = neg_size
        self.true_tail_filter = true_tail_filter

    def __len__(self) -> int:
        return self.triples.size(0)

    def _sample_negatives(self, h: int, r: int) -> torch.LongTensor:
        true_tails = self.true_tail_filter.get((h, r))
        negs = torch.empty(self.neg_size, dtype=torch.long)
        filled = 0
        # Rejection sampling in small batches; for realistic entity vocab sizes the
        # collision probability with true_tails is tiny so this almost always finishes
        # in a single round.
        while filled < self.neg_size:
            cand = torch.randint(0, self.num_entities, (self.neg_size - filled,))
            if true_tails:
                keep_mask = torch.tensor([c.item() not in true_tails for c in cand])
                cand = cand[keep_mask]
            n = cand.size(0)
            negs[filled:filled + n] = cand
            filled += n
        return negs

    def __getitem__(self, idx: int):
        h, r, t = self.triples[idx].tolist()
        neg_t = self._sample_negatives(h, r)
        return {
            'h': h, 'r': r, 't': t,
            'neg_h': torch.full((self.neg_size,), h, dtype=torch.long),
            'neg_r': torch.full((self.neg_size,), r, dtype=torch.long),
            'neg_t': neg_t,
        }


def collate_neg_sampling(batch: List[dict]) -> dict:
    return {
        'h': torch.tensor([b['h'] for b in batch], dtype=torch.long),
        'r': torch.tensor([b['r'] for b in batch], dtype=torch.long),
        't': torch.tensor([b['t'] for b in batch], dtype=torch.long),
        'neg_h': torch.stack([b['neg_h'] for b in batch], dim=0),
        'neg_r': torch.stack([b['neg_r'] for b in batch], dim=0),
        'neg_t': torch.stack([b['neg_t'] for b in batch], dim=0),
    }


def build_message_passing_graph(train_triples: torch.LongTensor, num_entities: int,
                                 num_relations: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """Build the edge list RGCN convolves over directly from the (already inverse-
    augmented) training triples, with degree normalization as in Schlichtkrull et al.
    2018.

    Normalization: Eq. 2 defines a general c_{i,r} (e.g. per-relation in-degree
    |N_i^r|), but for the *link prediction* experiments specifically (our use case
    here) the paper reports: "we found a normalization constant defined as c_{i,r} =
    c_i = sum_r |N_i^r| -- in other words, applied across relation types -- to work
    best" (Results section) -- i.e. total in-degree of the destination node across
    *all* relations, not per-relation in-degree. That's what's implemented below.

    Returns:
        edge_index: (2, E) src/dst node ids
        edge_type:  (E,) relation id per edge
        edge_norm:  (E,) 1 / total in-degree(dst), matching c_i above
    """
    src = train_triples[:, 0]
    rel = train_triples[:, 1]
    dst = train_triples[:, 2]

    # c_i = total in-degree of dst across all relations (see docstring above)
    _, inverse_idx, counts = torch.unique(dst, return_inverse=True, return_counts=True)
    edge_norm = 1.0 / counts[inverse_idx].float()

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, rel, edge_norm
