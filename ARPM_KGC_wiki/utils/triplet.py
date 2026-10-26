import heapq
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Optional, Set, Tuple

from ..setting.logger_config import logger


def _intern(s):
    """Intern strings so the many copies of an entity / relation id share one object (saves GBs on Wikidata5M)."""
    return sys.intern(s) if isinstance(s, str) else s


def iter_json_array(path: str, chunk_size: int = 1 << 24) -> Iterator[dict]:
    """Yield the elements of a (possibly multi-GB) top-level JSON array one at a time.

    `json.load` on the 20.6M-triple Wikidata5M train file materialises every dict at once
    (~11 GB) on top of the structures that are already in RAM. This reads the file in
    `chunk_size` pieces and decodes one object at a time, so the peak is only the final
    data structure. Works for both the compact and the indent=4 files written by preprocess.
    """
    decoder = json.JSONDecoder()
    with open(path, 'r', encoding='utf-8') as f:
        buf, pos, started = f.read(chunk_size), 0, False
        while True:
            # skip whitespace / commas; refill when the buffer is exhausted
            while True:
                while pos < len(buf) and buf[pos] in ' \t\r\n,':
                    pos += 1
                if pos < len(buf):
                    break
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf, pos = chunk, 0

            ch = buf[pos]
            if not started:
                if ch != '[':
                    raise ValueError(f'{path}: expected a JSON array')
                started = True
                pos += 1
                continue
            if ch == ']':
                return

            while True:
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    # the object straddles the end of the buffer: read more and retry
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf, pos = buf[pos:] + chunk, 0
            yield obj
            pos = end


@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:
    """Stores knowledge graph triplets and provides lookup by head-relation pairs."""

    def __init__(self, path_list: List[str]) -> None:
        self.path_list = path_list
        logger.info(f'Triplets path: {self.path_list}')

        self.relations: Set[str] = set()
        self.hr2tails = {}
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)

        logger.info(
            f'Triplet statistics: {len(self.relations)} relations, '
            f'{self.triplet_cnt} triplets'
        )

    def _load(self, path: str) -> None:
        # Streamed: forward and inverse triplets are inserted directly (same content as
        # `examples + [reverse_triplet(o) for o in examples]`), nothing is kept in a list.
        n = 0
        hr2tails, relations = self.hr2tails, self.relations
        for obj in iter_json_array(path):
            head_id, tail_id = _intern(obj['head_id']), _intern(obj['tail_id'])
            rel = _intern(obj['relation'])
            inv_rel = _intern(f"inverse {obj['relation']}")

            relations.add(rel)
            relations.add(inv_rel)
            hr2tails.setdefault((head_id, rel), set()).add(tail_id)
            hr2tails.setdefault((tail_id, inv_rel), set()).add(head_id)
            n += 1

        self.triplet_cnt += 2 * n

    def get_neighbors(self, head_id: str, relation: str) -> Set[str]:
        return self.hr2tails.get((head_id, relation), set())


class EntityDict:
    """Manages entity examples with ID-based and index-based lookups."""

    def __init__(
            self,
            entity_dict_dir: str,
            inductive_test_path: Optional[str] = None
    ) -> None:
        entities_path = os.path.join(entity_dict_dir, 'entities.json')
        if not os.path.exists(entities_path):
            raise FileNotFoundError(f'Entities file not found: {entities_path}')

        self.entity_exs = [EntityExample(**obj) for obj in iter_json_array(entities_path)]

        if inductive_test_path:
            self._filter_inductive_entities(inductive_test_path)

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}

        logger.info(f'Load {len(self.id2entity)} entities from {entities_path}')

    def _filter_inductive_entities(self, inductive_test_path: str) -> None:
        with open(inductive_test_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)

        valid_entity_ids = set()
        for ex in examples:
            valid_entity_ids.add(ex['head_id'])
            valid_entity_ids.add(ex['tail_id'])

        self.entity_exs = [
            ex for ex in self.entity_exs
            if ex.entity_id in valid_entity_ids
        ]

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self) -> int:
        return len(self.entity_exs)


class LinkGraph:
    """Undirected graph built from training triplets for neighbor queries."""

    def __init__(self, train_path: str) -> None:
        logger.info(f'Start to build link graph from {train_path}')

        self.graph = {}

        for ex in iter_json_array(train_path):
            head_id, tail_id = _intern(ex['head_id']), _intern(ex['tail_id'])
            self.graph.setdefault(head_id, set()).add(tail_id)
            self.graph.setdefault(tail_id, set()).add(head_id)

        logger.info(f'Done build link graph with {len(self.graph)} nodes')

    def get_neighbor_ids(self, entity_id: str, max_to_keep: int = 10) -> List[str]:
        """Returns the `max_to_keep` smallest neighbor IDs (== sorted(...)[:k], but O(n) for hub nodes)."""
        return heapq.nsmallest(max_to_keep, self.graph.get(entity_id, ()))

    def get_n_hop_entity_indices(
            self,
            entity_id: str,
            entity_dict: EntityDict,
            n_hop: int = 2,
            max_nodes: int = 100000
    ) -> Set[int]:
        """Returns entity indices within n_hop distance (cumulative). Returns empty
        set if max_nodes exceeded. Kept for backward-compatibility / reuse by other
        analyses; ARPM-KGC's structural memory itself uses `get_hop_layers` below,
        which separates each hop distance instead of merging them."""
        if n_hop < 0:
            return set()

        seen_eids = {entity_id}
        queue = deque([entity_id])

        for _ in range(n_hop):
            for _ in range(len(queue)):
                current = queue.popleft()
                for neighbor in self.graph.get(current, set()):
                    if neighbor not in seen_eids:
                        seen_eids.add(neighbor)
                        queue.append(neighbor)
                        if len(seen_eids) > max_nodes:
                            return set()

        return {entity_dict.entity_to_idx(e_id) for e_id in seen_eids}

    def get_hop_layers(self, entity_id: str, max_hop: int = 2,
                       max_nodes_per_hop: int = 2000,
                       exclude_edge: Optional[Tuple[str, str]] = None) -> List[Set[str]]:
        """Layer-wise BFS: element l-1 is the set of entity IDs at EXACT distance l.
        `exclude_edge=(a, b)` hides that undirected edge from the traversal (used in
        training to hide the query's own (h, t) edge so the gold tail cannot leak in)."""
        layers: List[Set[str]] = []
        seen = {entity_id}
        frontier = {entity_id}
        blocked = None if exclude_edge is None else (exclude_edge, exclude_edge[::-1])

        for _ in range(max_hop):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.get(node, set()):
                    if neighbor in seen or neighbor in next_frontier:
                        continue
                    if blocked is not None and (node, neighbor) in blocked:
                        continue
                    next_frontier.add(neighbor)
                    if len(next_frontier) >= max_nodes_per_hop:
                        break
                if len(next_frontier) >= max_nodes_per_hop:
                    break

            seen |= next_frontier
            layers.append(next_frontier)
            frontier = next_frontier

            if not frontier:
                break

        while len(layers) < max_hop:
            layers.append(set())

        return layers


def reverse_triplet(triplet: dict) -> dict:
    """Creates an inverse version of a triplet."""
    return {
        'head_id': triplet['tail_id'],
        'head': triplet['tail'],
        'relation': f"inverse {triplet['relation']}",
        'tail_id': triplet['head_id'],
        'tail': triplet['head']
    }