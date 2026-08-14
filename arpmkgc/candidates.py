"""
Relation-aware candidate memory construction (Proposal Sec 4.3).

    A_local^(l)(h, r):  relation-r triples whose heads sit at graph-hop
                        distance l from h (l = 1..L)
    A_global(r):        a sample from the full relation-r training set T_r
    A(h, r) = A_local(h, r) union A_global(r), bounded to a candidate
              budget M before neural scoring (Sec 4.3, "practical
              retrieval constraint").

Structural leakage (Limitation 6, Sec 10): both the link graph and the
triplet indices used here are built exclusively from the training split
(see `data.dict_hub.get_link_graph` / `get_train_triplet_dict`), so no
validation/test triple is ever reachable as a candidate anchor.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from .config import ARPMConfig
from .data.graph import LinkGraph
from .data.triplets import TripletDict

GLOBAL_HOP = 0  # reserved hop id meaning "no local structural distance"


@dataclass(frozen=True)
class CandidateAnchor:
    head_id: str
    tail_id: str
    hop: int          # GLOBAL_HOP for global candidates, 1..L for local
    is_local: bool


class CandidatePoolBuilder:
    """Builds the bounded candidate anchor pool A(h, r) for a query,
    using only the training-split link graph and triplet index."""

    def __init__(self, config: ARPMConfig, triplet_dict: TripletDict, link_graph: LinkGraph,
                 rng: Optional[random.Random] = None):
        self.config = config
        self.triplet_dict = triplet_dict
        self.link_graph = link_graph
        self.rng = rng or random.Random()

    def build(self, head_id: str, relation: str, tail_id: str) -> List[CandidateAnchor]:
        local = self._build_local(head_id, relation) if self.config.use_local_candidates else []
        glob_ = self._build_global(relation) if self.config.use_global_candidates else []

        pool = self._dedupe_excluding_query(local + glob_, head_id, tail_id)
        return self._apply_budget(pool)

    # -- local structural candidates -----------------------------------
    def _build_local(self, head_id: str, relation: str) -> List[CandidateAnchor]:
        max_hop = self.config.max_hop
        per_hop = self.config.local_candidates_per_hop
        layers = self.link_graph.hop_layers(head_id, max_hop)
        relation_heads = self.triplet_dict.get_relation_heads(relation)

        candidates: List[CandidateAnchor] = []
        for hop in range(1, max_hop + 1):
            eligible_heads = list(layers.get(hop, set()) & relation_heads)
            if not eligible_heads:
                continue
            self.rng.shuffle(eligible_heads)
            for hi in eligible_heads[:per_hop]:
                tails = self.triplet_dict.get_neighbor_tails(hi, relation)
                if not tails:
                    continue
                ti = self.rng.choice(sorted(tails))
                candidates.append(CandidateAnchor(head_id=hi, tail_id=ti, hop=hop, is_local=True))
        return candidates

    # -- global relation-level candidates --------------------------------
    def _build_global(self, relation: str) -> List[CandidateAnchor]:
        pairs = self.triplet_dict.get_relation_pairs(relation)
        if not pairs:
            return []
        budget = min(self.config.global_candidates, len(pairs))
        sampled = self.rng.sample(pairs, budget)
        return [CandidateAnchor(head_id=h, tail_id=t, hop=GLOBAL_HOP, is_local=False)
                for h, t in sampled]

    @staticmethod
    def _dedupe_excluding_query(candidates: List[CandidateAnchor], head_id: str, tail_id: str,
                                 ) -> List[CandidateAnchor]:
        seen: Set[Tuple[str, str]] = set()
        deduped = []
        for c in candidates:
            key = (c.head_id, c.tail_id)
            if key == (head_id, tail_id):
                continue  # never let the query's own triple leak in as an anchor
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped

    def _apply_budget(self, pool: List[CandidateAnchor]) -> List[CandidateAnchor]:
        budget = self.config.candidate_budget
        if len(pool) <= budget:
            return pool
        local = [c for c in pool if c.is_local]
        glob_ = [c for c in pool if not c.is_local]
        self.rng.shuffle(local)
        self.rng.shuffle(glob_)
        # Preserve local/global proportionality when trimming to budget so a
        # single dominant source cannot silently starve the other.
        n_local = round(budget * len(local) / max(len(pool), 1))
        n_local = min(n_local, len(local), budget)
        n_global = budget - n_local
        trimmed = local[:n_local] + glob_[:n_global]
        self.rng.shuffle(trimmed)
        return trimmed
