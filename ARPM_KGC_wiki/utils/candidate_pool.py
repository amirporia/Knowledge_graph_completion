"""Relation-aware candidate anchor retrieval (v2).

Same pool definition as before, A(h,r) = A_local(h,r) U A_global(r), with hop slots
0..num_hops for local anchors and NO_HOP (-1) for global ones.

Differences from the previous version:
  * hop-0 anchors (other known tails of the same (h,r)) are NEVER dropped by the
    total-budget cap: they feed the RAA-KGC anchor-enhanced query path, which is the
    strongest component of the baseline. Only hop>=1 and global candidates are
    sub-sampled to fit `anchor_budget`.
  * hop-0 needs no graph traversal, so it is kept even with --disable-link-graph
    (only hop>=1 requires the link graph).
  * Hub-safe: for (h,r) pairs with more than LARGE_NEIGHBORHOOD known tails (Wikidata5M has
    pairs with >1M tails) the same-head candidates are drawn with rng.sample over a cached sorted
    tuple instead of sorting + shuffling the whole set for every example. Smaller neighbourhoods
    (all of wn18rr / fb15k237 in practice) take the original code path unchanged.
Retrieval reads only the TRAINING graph, so no valid/test label can enter the pool.
"""
import random
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from array import array
from .dict_hub import get_link_graph, get_train_triplet_dict, get_entity_dict
from ..setting.config import args
from ..setting.logger_config import logger

NO_HOP = -1
LARGE_NEIGHBORHOOD = 5000


class CandidateAnchor:
    __slots__ = ('head_id', 'relation', 'tail_id', 'hop', 'is_local')

    def __init__(self, head_id: str, relation: str, tail_id: str, hop: int, is_local: bool):
        self.head_id = head_id
        self.relation = relation
        self.tail_id = tail_id
        self.hop = hop
        self.is_local = is_local


class CandidatePoolBuilder:
    def __init__(self, num_hops: int, local_per_hop_budget: int,
                 global_budget: int, total_budget: int, use_link_graph: bool):
        self.num_hops = num_hops
        self.local_per_hop_budget = local_per_hop_budget
        self.global_budget = global_budget
        self.total_budget = total_budget

        self.link_graph = get_link_graph() if use_link_graph else None
        self.train_triplet_dict = get_train_triplet_dict()

        self._relation2pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._build_relation_index()

    def _build_relation_index(self) -> None:
        ent = get_entity_dict()
        self._ids = [e.entity_id for e in ent.entity_exs]  # idx -> id (shared strings)
        to_idx = ent.entity_to_idx
        hs, ts_ = defaultdict(lambda: array('i')), defaultdict(lambda: array('i'))
        for (head_id, relation), tail_ids in self.train_triplet_dict.hr2tails.items():
            hi, a, b = to_idx(head_id), hs[relation], ts_[relation]
            for t in sorted(tail_ids):
                a.append(hi);
                b.append(to_idx(t))
        self._rel_h = {r: np.frombuffer(a, dtype=np.int32) for r, a in hs.items()}
        self._rel_t = {r: np.frombuffer(a, dtype=np.int32) for r, a in ts_.items()}
        logger.info(f'CandidatePoolBuilder: indexed {len(self._rel_h)} relations')

    @staticmethod
    def _get_rng(head_id: str, relation: str):
        # Evaluation: deterministic per-query sampling. Training: global `random`.
        if args.is_test:
            return random.Random(f'{head_id}\t{relation}')
        return random

    @lru_cache(maxsize=8)
    def _sorted_tails_cached(self, head_id, relation):
        """Sorted tails of a hub (h,r); cached per process so the sort happens once, not per example."""
        return tuple(sorted(self.train_triplet_dict.get_neighbors(head_id, relation)))

    def _same_head_candidates(self, head_id, relation, tail_id, rng):
        neighbors = self.train_triplet_dict.get_neighbors(head_id, relation)

        if len(neighbors) > LARGE_NEIGHBORHOOD:
            pool = self._sorted_tails_cached(head_id, relation)
            picks = rng.sample(range(len(pool)), min(len(pool), self.local_per_hop_budget + 1))
            out = []
            for i in picks:
                if len(out) >= self.local_per_hop_budget:
                    break
                if pool[i] == tail_id:
                    continue
                out.append(CandidateAnchor(head_id, relation, pool[i], 0, True))
            return out

        candidate_tails = sorted(neighbors)
        rng.shuffle(candidate_tails)
        candidates, found = [], 0
        for cand_tail in candidate_tails:
            if found >= self.local_per_hop_budget:
                break
            if cand_tail == tail_id:
                continue
            candidates.append(CandidateAnchor(head_id, relation, cand_tail, 0, True))
            found += 1
        return candidates

    def _local_candidates(self, head_id, relation, tail_id, rng, exclude_edge=None):
        candidates = self._same_head_candidates(head_id, relation, tail_id, rng)
        if self.link_graph is None or self.num_hops < 1:
            return candidates

        hop_layers = self.link_graph.get_hop_layers(
            head_id, max_hop=self.num_hops, exclude_edge=exclude_edge
        )
        for hop_slot, layer_nodes in enumerate(hop_layers, start=1):
            if not layer_nodes or self.local_per_hop_budget <= 0:
                continue
            nodes = sorted(layer_nodes)
            rng.shuffle(nodes)
            found = 0
            for node in nodes:
                if found >= self.local_per_hop_budget:
                    break
                cand_tail_ids = self.train_triplet_dict.get_neighbors(node, relation)
                if not cand_tail_ids:
                    continue
                if len(cand_tail_ids) > LARGE_NEIGHBORHOOD:  # hub: don't sort 1M ids for every candidate
                    cand_tail = rng.choice(self._sorted_tails_cached(node, relation))
                else:
                    cand_tail = rng.choice(sorted(cand_tail_ids))
                candidates.append(CandidateAnchor(node, relation, cand_tail, hop_slot, True))
                found += 1
        return candidates

    def _global_candidates(self, head_id, tail_id, relation, rng):
        h_arr = self._rel_h.get(relation)
        if h_arr is None or self.global_budget <= 0:
            return []
        t_arr, ids, n = self._rel_t[relation], self._ids, len(h_arr)
        picks = rng.sample(range(n), self.global_budget) if n > self.global_budget else range(n)
        out = []
        for i in picks:
            h = ids[h_arr[i]]
            if h != head_id:
                out.append(CandidateAnchor(h, relation, ids[t_arr[i]], NO_HOP, False))
        return out

    def build(self, head_id, relation, tail_id):
        training = not args.is_test
        exclude_tail = tail_id if training else None
        exclude_edge = (head_id, tail_id) if training else None
        rng = self._get_rng(head_id, relation)

        local = self._local_candidates(head_id, relation, exclude_tail, rng, exclude_edge)
        glob = self._global_candidates(head_id, exclude_tail, relation, rng)

        hop0 = [c for c in local if c.hop == 0]
        others = [c for c in local if c.hop != 0] + glob
        room = max(self.total_budget - len(hop0), 0)
        if len(others) > room:
            others = rng.sample(others, room)
        return (hop0 + others)[:self.total_budget]


_pool_builder: 'CandidatePoolBuilder' = None


def get_candidate_pool_builder() -> CandidatePoolBuilder:
    global _pool_builder
    if _pool_builder is None:
        _pool_builder = CandidatePoolBuilder(
            num_hops=args.num_hops,
            local_per_hop_budget=args.local_per_hop_budget,
            global_budget=args.global_budget,
            total_budget=args.anchor_budget,
            use_link_graph=args.use_link_graph,
        )
    return _pool_builder