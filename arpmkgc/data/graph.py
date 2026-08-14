"""
Undirected structural graph over training triples, used to build the local
candidate pool A_local^(l)(h, r) of Section 4.3: for each hop distance l,
the set of entities reachable from h at exactly distance l.
"""

import json
from collections import deque
from typing import Dict, List, Set


class LinkGraph:
    """Adjacency built once from the training split; supports exact-hop BFS
    layering up to a configurable maximum hop `max_hop`.
    """

    def __init__(self, train_path: str):
        self.graph: Dict[str, Set[str]] = {}
        with open(train_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        for ex in examples:
            h, t = ex["head_id"], ex["tail_id"]
            self.graph.setdefault(h, set()).add(t)
            self.graph.setdefault(t, set()).add(h)

    def hop_layers(self, entity_id: str, max_hop: int) -> Dict[int, Set[str]]:
        """Returns {1: {nodes at distance 1}, 2: {nodes at distance 2}, ...}
        up to `max_hop`. Distances are exact (a node appears in exactly one
        layer -- the layer for its shortest-path distance from `entity_id`).
        """
        layers: Dict[int, Set[str]] = {}
        seen = {entity_id}
        frontier = {entity_id}

        for hop in range(1, max_hop + 1):
            next_frontier: Set[str] = set()
            for node in frontier:
                for neighbor in self.graph.get(node, ()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        next_frontier.add(neighbor)
            layers[hop] = next_frontier
            frontier = next_frontier
            if not frontier:
                break

        for hop in range(1, max_hop + 1):
            layers.setdefault(hop, set())

        return layers

    def get_n_hop_neighbor_ids(self, entity_id: str, n_hop: int, max_nodes: int = 100_000) -> Set[str]:
        """Cumulative (<= n_hop) neighborhood, used for lightweight
        neighbor-description augmentation of entity text (optional)."""
        if n_hop < 0:
            return set()

        seen = {entity_id}
        queue = deque([entity_id])
        for _ in range(n_hop):
            for _ in range(len(queue)):
                current = queue.popleft()
                for neighbor in self.graph.get(current, ()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
                        if len(seen) > max_nodes:
                            return set()
        seen.discard(entity_id)
        return seen
