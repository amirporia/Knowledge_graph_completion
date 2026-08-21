"""Relation-aware candidate anchor retrieval.

Implements ARPM_KGC_Proposal.tex Sec. 3.2 ("Relation-Aware Candidate Memory"):
for a query (h, r, ?) this builds the candidate pool

    A(h, r) = A_local(h, r)  U  A_global(r)

where A_local(h, r) = U_{l=1..L} A_local^(l)(h, r) is made of relation-r training
triples whose heads sit at graph distance l from h (one hop layer at a time, via
LinkGraph.get_hop_layers), and A_global(r) is a bounded sample from the relation-r
training set T_r. Every candidate is tagged with its hop distance (0 for global /
unknown distance) so the model can later build hop-specific structural memory
(Sec. 3.5) on top of exactly the same weighted anchor set used for retrieval and
prototype construction (Sec. 3.3-3.4).

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


class CandidateAnchor:
    __slots__ = ('head_id', 'relation', 'tail_id', 'hop', 'is_local')

    def __init__(self, head_id: str, relation: str, tail_id: str, hop: int, is_local: bool):
        self.head_id = head_id
        self.relation = relation
        self.tail_id = tail_id
        self.hop = hop            # 0 = global / no known structural distance, 1..L = local hop
        self.is_local = is_local


class CandidatePoolBuilder:
    """Builds A_local(h,r) and A_global(r) candidate pools per query, each bounded
    by a configurable budget so the neural retrieval/prototype/structural stages
    that follow operate on a manageable, fixed-shape candidate set (Sec. 3.2,
    "Practical retrieval constraint")."""

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

    def _local_candidates(self, head_id: str, relation: str) -> List[CandidateAnchor]:
        """A_local(h,r) = U_l A_local^(l)(h,r): for each hop layer, sample a bounded
        number of nodes reachable at that exact distance and, if they participate in
        a relation-r training triple, add that triple as a hop-l local anchor."""
        if self.link_graph is None or self.num_hops <= 0:
            return []

        hop_layers = self.link_graph.get_hop_layers(head_id, max_hop=self.num_hops)
        candidates: List[CandidateAnchor] = []

        for hop_idx, layer_nodes in enumerate(hop_layers, start=1):
            if not layer_nodes:
                continue
            nodes = list(layer_nodes)
            random.shuffle(nodes)

            found = 0
            for node in nodes:
                tail_ids = self.train_triplet_dict.get_neighbors(node, relation)
                if not tail_ids:
                    continue
                tail_id = next(iter(tail_ids))
                candidates.append(CandidateAnchor(node, relation, tail_id, hop_idx, True))
                found += 1
                if found >= self.local_per_hop_budget:
                    break

        return candidates

    def _global_candidates(self, head_id: str, tail_id: str, relation: str) -> List[CandidateAnchor]:
        """A_global(r) subset of T_r, excluding the query's own (h, t) pair so the
        candidate pool never trivially contains the answer being predicted."""
        pool = self._relation2pairs.get(relation, [])
        if not pool:
            return []

        sampled = random.sample(pool, self.global_budget) if len(pool) > self.global_budget else pool

        candidates = []
        for cand_head, cand_tail in sampled:
            if cand_head == head_id and cand_tail == tail_id:
                continue
            candidates.append(CandidateAnchor(cand_head, relation, cand_tail, 0, False))
        return candidates

    def build(self, head_id: str, relation: str, tail_id: str) -> List[CandidateAnchor]:
        """A(h,r) = A_local(h,r) U A_global(r), capped at `total_budget`.

        May legitimately return an empty list (isolated head with no graph
        neighbors and a relation seen nowhere else) -- callers must handle a
        query with zero candidate anchors gracefully rather than falling back to
        the query's own true triple, which would leak the label.
        """
        candidates = self._local_candidates(head_id, relation) + \
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
