"""Relation-aware candidate anchor retrieval.

Implements "Relation-Aware Candidate Memory":
for a query (h, r, ?) this builds the candidate pool

    A(h, r) = A_local(h, r)  U  A_global(r)

where A_local(h, r) = U_{l=0..num_hops} A_local^(l)(h, r) is made of
num_hops+1 local hop categories total:
  - hop 0: OTHER valid tails for the exact SAME (h, r) pair -- i.e. graph
    distance 0, no traversal at all (excludes the query's own true tail, to
    avoid leaking the label). This is the tightest possible local evidence:
    "what else does this exact head-relation pair connect to" -- most useful
    for one-to-many relations.
  - hop l (1 <= l <= num_hops): relation-r training triples whose heads sit
    at graph distance l from h (one hop layer at a time, via
    LinkGraph.get_hop_layers).
`--num-hops N` therefore yields N+1 categories: hop-0, hop-1, ..., hop-N
(e.g. `--num-hops 2` gives hop-0, hop-1, AND hop-2 -- not just two).
A_global(r) is a bounded sample from the relation-r training set T_r,
EXCLUDING ground truth triples.

Every candidate is tagged with its hop slot (-1 for global / unknown distance,
0..num_hops for local, 0-indexed so slot l lines up directly with the l-th row
of hop-specific structural memory m^(l) and the l-th output channel of G_hop;
G_hop/m^(l) are therefore allocated num_hops+1 slots, see model/models.py) so
the model can later build hop-specific structural memory on top of
exactly the same weighted anchor set used for retrieval and prototype
construction.

Retrieval only ever reads the *training* graph (via get_train_triplet_dict /
get_link_graph), independent of args.is_test, so no validation/test labels can
leak into the candidate pool at evaluation time.
"""
import random
import zlib
from collections import defaultdict
from typing import Dict, List, Tuple

from .dict_hub import get_link_graph, get_train_triplet_dict
from ..setting.config import args
from ..setting.logger_config import logger

NO_HOP = -1  # sentinel hop value for global candidates (and, in utils/doc.py's


# collate, for padding slots) -- distinct from every valid local
# hop slot 0..num_hops (including local hop 0), so it can never
# be mistaken for a genuine local anchor.

def _make_rng(head_id: str, relation: str):
    """Eval: per-query seeded RNG, so anchors (and MRR) are reproducible.
    Training: the global `random` module (fresh samples every epoch)."""
    if args.is_test:
        return random.Random(zlib.crc32(f'{args.eval_seed}|{head_id}|{relation}'.encode('utf-8')))
    return random


def _ordered(items):
    """Stable order at eval, independent of PYTHONHASHSEED; cheap list() in training."""
    return sorted(items) if args.is_test else list(items)


class CandidateAnchor:
    __slots__ = ('head_id', 'relation', 'tail_id', 'hop', 'is_local')

    def __init__(self, head_id: str, relation: str, tail_id: str, hop: int, is_local: bool):
        self.head_id = head_id
        self.relation = relation
        self.tail_id = tail_id
        self.hop = hop  # NO_HOP (-1) = global; 0..num_hops = local hop slot
        # (0 = same head & relation, graph distance 0;
        # 1..num_hops = increasing graph distance)
        self.is_local = is_local


class CandidatePoolBuilder:
    """Builds A_local(h,r) and A_global(r) candidate pools per query, each bounded
    by a configurable budget so the neural retrieval/prototype/structural stages
    that follow operate on a manageable, fixed-shape candidate set."""

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
        """Index every training (head_id, tail_id) pair by relation once, so global
        candidate sampling is O(sample size) instead of a linear scan of the whole
        training set per query."""
        for (head_id, relation), tail_ids in self.train_triplet_dict.hr2tails.items():
            pairs = self._relation2pairs[relation]
            for tail_id in tail_ids:
                pairs.append((head_id, tail_id))
        logger.info(
            f'CandidatePoolBuilder: indexed {len(self._relation2pairs)} relations '
            f'for global candidate retrieval (num_hops={self.num_hops}, '
            f'anchor_budget={self.total_budget})'
        )
        for pairs in self._relation2pairs.values():
            pairs.sort()

    def _same_head_candidates(self, head_id, relation, tail_id, rng) -> List[CandidateAnchor]:
        tails = [t for t in _ordered(self.train_triplet_dict.get_neighbors(head_id, relation))
                 if t != tail_id]
        rng.shuffle(tails)
        budget = max(self.local_per_hop_budget, 0)
        return [CandidateAnchor(head_id, relation, t, 0, True) for t in tails[:budget]]

    def _local_candidates(self, head_id, relation, tail_id, rng, blocked) -> List[CandidateAnchor]:
        if self.link_graph is None:
            return []

        candidates = self._same_head_candidates(head_id, relation, tail_id, rng)
        if self.num_hops < 1 or self.local_per_hop_budget <= 0:
            return candidates

        # `blocked` removes the query's own h-t edge during training, so t is not a hop-1
        # node and nodes reachable only through t do not appear at hop 2.
        hop_layers = self.link_graph.get_hop_layers(head_id, max_hop=self.num_hops, blocked=blocked)

        for hop_slot, layer_nodes in enumerate(hop_layers, start=1):
            if not layer_nodes:
                continue
            nodes = _ordered(layer_nodes)
            rng.shuffle(nodes)
            found = 0
            for node in nodes:
                if found >= self.local_per_hop_budget:
                    break
                cand_tails = self.train_triplet_dict.get_neighbors(node, relation)
                if not cand_tails:
                    continue
                cand_tail = rng.choice(_ordered(cand_tails))
                candidates.append(CandidateAnchor(node, relation, cand_tail, hop_slot, True))
                found += 1
        return candidates

    def _global_candidates(self, head_id, relation, rng) -> List[CandidateAnchor]:
        """A_global(r): a different head from the query (same-head pairs are hop-0)."""
        pool = self._relation2pairs.get(relation, [])
        if not pool or self.global_budget <= 0:
            return []
        sampled = rng.sample(pool, self.global_budget) if len(pool) > self.global_budget else pool
        return [CandidateAnchor(h, relation, t, NO_HOP, False)
                for h, t in sampled if h != head_id]

    def build(self, head_id: str, relation: str, tail_id: str = None, blocked=None) -> List[CandidateAnchor]:
        """`tail_id` is passed ONLY in training (None at eval: no label use at test time).
        `blocked=(h, t)` removes the query's own edge from the graph (training only)."""
        rng = _make_rng(head_id, relation)
        local = self._local_candidates(head_id, relation, tail_id, rng, blocked)
        seen = {(c.head_id, c.tail_id) for c in local}
        glob = [c for c in self._global_candidates(head_id, relation, rng)
                if (c.head_id, c.tail_id) not in seen]  # true set union
        candidates = local + glob
        if len(candidates) > self.total_budget:
            candidates = rng.sample(candidates, max(self.total_budget, 0))
        return candidates


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
