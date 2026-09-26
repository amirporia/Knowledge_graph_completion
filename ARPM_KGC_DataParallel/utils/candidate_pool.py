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
from collections import defaultdict
from typing import Dict, List, Tuple

from .dict_hub import get_link_graph, get_train_triplet_dict
from ..setting.config import args
from ..setting.logger_config import logger

NO_HOP = -1  # sentinel hop value for global candidates (and, in utils/doc.py's
             # collate, for padding slots) -- distinct from every valid local
             # hop slot 0..num_hops (including local hop 0), so it can never
             # be mistaken for a genuine local anchor.


class CandidateAnchor:
    __slots__ = ('head_id', 'relation', 'tail_id', 'hop', 'is_local')

    def __init__(self, head_id: str, relation: str, tail_id: str, hop: int, is_local: bool):
        self.head_id = head_id
        self.relation = relation
        self.tail_id = tail_id
        self.hop = hop            # NO_HOP (-1) = global; 0..num_hops = local hop slot
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

    def _same_head_candidates(self, head_id: str, relation: str, tail_id: str) -> List[CandidateAnchor]:
        """Hop 0: other valid tails for the exact same (h, r) pair (graph
        distance 0 -- no traversal), excluding the query's own true tail."""
        candidate_tails = list(self.train_triplet_dict.get_neighbors(head_id, relation))
        random.shuffle(candidate_tails)

        candidates = []
        found = 0
        for cand_tail in candidate_tails:
            if found >= self.local_per_hop_budget:
                break
            if cand_tail == tail_id:
                continue
            candidates.append(CandidateAnchor(head_id, relation, cand_tail, 0, True))
            found += 1

        return candidates

    def _local_candidates(self, head_id: str, relation: str, tail_id: str) -> List[CandidateAnchor]:
        """A_local(h,r) = U_{l=0}^{num_hops} A_local^(l)(h,r): hop 0 is the
        same-head-and-relation category (see `_same_head_candidates`); hops
        1..num_hops sample a bounded number of nodes reachable at each exact
        graph distance and, if they participate in a relation-r training
        triple, add that triple as a local anchor tagged with its hop slot.

        `--num-hops N` therefore yields N+1 local categories in total:
        hop-0 (same head & relation), hop-1 (graph distance 1), ...,
        hop-N (graph distance N) -- e.g. `--num-hops 2` gives hop-0, hop-1,
        AND hop-2."""
        if self.link_graph is None:
            return []

        candidates: List[CandidateAnchor] = self._same_head_candidates(head_id, relation, tail_id)

        if self.num_hops < 1:
            return candidates

        # hop_layers[0] = nodes at graph distance 1, ..., hop_layers[num_hops-1]
        # = nodes at graph distance num_hops -- labeled here as local hop
        # SLOTS 1..num_hops (enumerate start=1), on top of hop-0 above.
        hop_layers = self.link_graph.get_hop_layers(head_id, max_hop=self.num_hops)

        for hop_slot, layer_nodes in enumerate(hop_layers, start=1):
            if not layer_nodes or self.local_per_hop_budget <= 0:
                continue
            nodes = list(layer_nodes)
            random.shuffle(nodes)

            found = 0
            for node in nodes:
                # Check the budget BEFORE adding, not after: with the check
                # only after append+increment, local_per_hop_budget=0 would
                # still let exactly one candidate through per non-empty hop.
                if found >= self.local_per_hop_budget:
                    break
                cand_tail_ids = self.train_triplet_dict.get_neighbors(node, relation)
                if not cand_tail_ids:
                    continue
                cand_tail = next(iter(cand_tail_ids))
                candidates.append(CandidateAnchor(node, relation, cand_tail, hop_slot, True))
                found += 1

        return candidates

    def _global_candidates(self, head_id: str, tail_id: str, relation: str) -> List[CandidateAnchor]:
        """A_global(r) subset of T_r, excluding ground truth triples."""
        pool = self._relation2pairs.get(relation, [])
        if not pool:
            return []

        sampled = random.sample(pool, self.global_budget) if len(pool) > self.global_budget else pool

        candidates = []
        for cand_head, cand_tail in sampled:
            if cand_head == head_id and cand_tail == tail_id:
                continue
            candidates.append(CandidateAnchor(cand_head, relation, cand_tail, NO_HOP, False))
        return candidates

    def build(self, head_id: str, relation: str, tail_id: str) -> List[CandidateAnchor]:
        """A(h,r) = A_local(h,r) U A_global(r), capped at `total_budget`.

        May legitimately return an empty list (isolated head with no graph
        neighbors, no other tail for the same (h,r) pair, and a relation seen
        nowhere else) -- callers must handle a query with zero candidate
        anchors gracefully rather than falling back to the query's own true
        triple, which would leak the label.
        """
        candidates = self._local_candidates(head_id, relation, tail_id) + \
            self._global_candidates(head_id, tail_id, relation)

        if len(candidates) > self.total_budget:
            candidates = random.sample(candidates, self.total_budget)

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
