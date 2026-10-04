"""Relation-aware candidate anchor retrieval.

For a query (h, r, ?) this builds the candidate pool

    A(h, r) = A_local(h, r)  U  A_global(r)

A_local(h, r) = U_{l=0..num_hops} A_local^(l)(h, r)   (num_hops + 1 hop categories)
  - hop 0: OTHER valid tails of the exact same (h, r) pair (query's own tail excluded).
  - hop l (1..num_hops): relation-r training triples whose head sits at graph distance l.
A_global(r): relation-r training pairs with a head different from h.

Every candidate is tagged with its hop slot (-1 for global, 0..num_hops for local).

Filter-and-fill selection (this version)
----------------------------------------
Every filter (query's own tail, duplicate (head, tail) pair, same-head pair in the global
pool, ...) is applied WHILE candidates are being picked, and picking continues down the
candidate list until the budget is met or the source is exhausted. No filter runs after
selection, so budgets are only undershot when the data genuinely has too few candidates.

  1. Local: each hop fills up to `local_per_hop_budget` DISTINCT (head, tail) pairs.
  2. Local is trimmed to `anchor_budget` only if it alone exceeds it.
  3. Global quota = min(global_budget + local_shortfall, anchor_budget - |local|), where
     local_shortfall is the unused local quota (`--no-fill-anchor-budget` disables the
     shortfall term, so global never exceeds `global_budget`). global_budget == 0 (A6)
     always means no global anchors.
  4. Global anchors are taken in rank order (facility mode) or by random draw (random
     mode), skipping same-head pairs and pairs already present locally, until the quota
     is met.

Retrieval only ever reads the *training* graph, so no validation/test label can leak.
"""
import random
import zlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .dict_hub import get_link_graph, get_train_triplet_dict
from ..setting.config import args
from ..setting.logger_config import logger

NO_HOP = -1  # sentinel hop for global candidates (and padding slots in utils/doc.py collate)


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
        self.hop = hop
        self.is_local = is_local


class CandidatePoolBuilder:
    """Builds A_local(h,r) and A_global(r) per query with filter-and-fill selection."""

    def __init__(self, num_hops: int, local_per_hop_budget: int, global_budget: int,
                 total_budget: int, use_link_graph: bool,
                 global_anchor_mode: str = 'facility', global_reserve: int = 10,
                 global_candidate_cap: int = 500, global_weight_power: float = 1.0,
                 global_select_batch_size: int = 128, fill_anchor_budget: bool = True):
        self.num_hops = num_hops
        self.local_per_hop_budget = local_per_hop_budget
        self.global_budget = global_budget
        self.total_budget = total_budget
        self.global_anchor_mode = global_anchor_mode
        self.fill_anchor_budget = fill_anchor_budget

        self.link_graph = get_link_graph() if use_link_graph else None
        self.train_triplet_dict = get_train_triplet_dict()

        self._relation2pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._build_relation_index()

        # Max local anchors the local stage could ever return (used for the fill shortfall).
        if self.link_graph is None or local_per_hop_budget <= 0:
            self._local_cap = 0
        else:
            self._local_cap = (num_hops + 1) * local_per_hop_budget

        self._global_table: Dict[str, List[Tuple[str, str]]] = {}
        if global_anchor_mode == 'facility' and global_budget > 0:
            from .global_anchor_selector import load_or_build_global_anchor_table
            # Reserve anchors: head-skips, local-duplicate skips and fill-in all consume
            # extra entries, so the table is longer than global_budget.
            table_size = global_budget + max(global_reserve, 1)
            self._global_table = load_or_build_global_anchor_table(
                self._relation2pairs, table_size=table_size, cap=global_candidate_cap,
                power=global_weight_power, batch_size=global_select_batch_size)

    def _build_relation_index(self) -> None:
        for (head_id, relation), tail_ids in self.train_triplet_dict.hr2tails.items():
            pairs = self._relation2pairs[relation]
            for tail_id in tail_ids:
                pairs.append((head_id, tail_id))
        for pairs in self._relation2pairs.values():
            pairs.sort()
        logger.info(
            f'CandidatePoolBuilder: indexed {len(self._relation2pairs)} relations '
            f'(num_hops={self.num_hops}, anchor_budget={self.total_budget}, '
            f'global_mode={self.global_anchor_mode}, fill={self.fill_anchor_budget})'
        )

    # ------------------------------------------------------------------ local

    def _same_head_candidates(self, head_id, relation, tail_id, rng) -> List[CandidateAnchor]:
        # The query's own tail is filtered BEFORE the budget cut, so the budget fills
        # with other valid tails instead of losing a slot to the filter.
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

        seen: Set[Tuple[str, str]] = {(c.head_id, c.tail_id) for c in candidates}
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
                cand_tails = _ordered(self.train_triplet_dict.get_neighbors(node, relation))
                if not cand_tails:
                    continue
                # Pick a tail whose (node, tail) pair is not already in the pool; a node
                # whose every pair is a duplicate is skipped and the next node is tried.
                pick = rng.choice(cand_tails)
                if (node, pick) in seen:
                    unseen = [t for t in cand_tails if (node, t) not in seen]
                    if not unseen:
                        continue
                    pick = rng.choice(unseen)
                seen.add((node, pick))
                candidates.append(CandidateAnchor(node, relation, pick, hop_slot, True))
                found += 1
        return candidates

    # ----------------------------------------------------------------- global

    def _global_quota(self, n_local: int) -> int:
        if self.global_budget <= 0:
            return 0
        room = max(self.total_budget - n_local, 0)
        want = self.global_budget
        if self.fill_anchor_budget:
            want += max(self._local_cap - n_local, 0)
        return min(want, room)

    def _global_candidates(self, head_id, relation, rng, seen, quota) -> List[CandidateAnchor]:
        """A_global(r): different head from the query, not already in the local pool."""
        if quota <= 0:
            return []
        if self.global_anchor_mode == 'facility':
            return self._global_from_table(head_id, relation, seen, quota)
        return self._global_random(head_id, relation, rng, seen, quota)

    def _global_from_table(self, head_id, relation, seen, quota) -> List[CandidateAnchor]:
        out: List[CandidateAnchor] = []
        for h, t in self._global_table.get(relation, []):
            if len(out) >= quota:
                break
            if h == head_id or (h, t) in seen:  # filter while selecting; next ranked fills in
                continue
            out.append(CandidateAnchor(h, relation, t, NO_HOP, False))
        return out

    def _global_random(self, head_id, relation, rng, seen, quota) -> List[CandidateAnchor]:
        pool = self._relation2pairs.get(relation, [])
        if not pool:
            return []
        m = quota * 3 + len(seen)
        while True:
            drawn = rng.sample(pool, min(m, len(pool)))
            out = [CandidateAnchor(h, relation, t, NO_HOP, False)
                   for h, t in drawn if h != head_id and (h, t) not in seen]
            if len(out) >= quota or m >= len(pool):
                return out[:quota]
            m *= 2  # too many draws were filtered: draw more until the quota is met

    # ------------------------------------------------------------------ build

    def build(self, head_id: str, relation: str, tail_id: str = None, blocked=None) -> List[CandidateAnchor]:
        """`tail_id` is passed ONLY in training (None at eval: no label use at test time).
        `blocked=(h, t)` removes the query's own edge from the graph (training only)."""
        rng = _make_rng(head_id, relation)
        budget = max(self.total_budget, 0)

        local = self._local_candidates(head_id, relation, tail_id, rng, blocked)
        if len(local) > budget:
            local = rng.sample(local, budget)

        seen = {(c.head_id, c.tail_id) for c in local}
        glob = self._global_candidates(head_id, relation, rng, seen,
                                       self._global_quota(len(local)))
        return local + glob


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
            global_anchor_mode=args.global_anchor_mode,
            global_reserve=args.global_reserve,
            global_candidate_cap=args.global_candidate_cap,
            global_weight_power=args.global_weight_power,
            global_select_batch_size=args.global_select_batch_size,
            fill_anchor_budget=args.fill_anchor_budget,
        )
    return _pool_builder