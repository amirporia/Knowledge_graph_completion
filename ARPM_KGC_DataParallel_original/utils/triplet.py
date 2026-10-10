import json
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set

from ..setting.logger_config import logger


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
        with open(path, 'r', encoding='utf-8') as f:
            examples = json.load(f)

        examples += [reverse_triplet(obj) for obj in examples]

        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            self.hr2tails.setdefault(key, set()).add(ex['tail_id'])

        self.triplet_cnt = len(examples)

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

        with open(entities_path, 'r', encoding='utf-8') as f:
            self.entity_exs = [EntityExample(**obj) for obj in json.load(f)]

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

        with open(train_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)

        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            self.graph.setdefault(head_id, set()).add(tail_id)
            self.graph.setdefault(tail_id, set()).add(head_id)

        logger.info(f'Done build link graph with {len(self.graph)} nodes')

    def get_neighbor_ids(self, entity_id: str, max_to_keep: int = 10) -> List[str]:
        """Returns sorted list of neighbor IDs (deterministic order)."""
        neighbor_ids = self.graph.get(entity_id, set())
        return sorted(neighbor_ids)[:max_to_keep]

    def get_n_hop_entity_indices(
            self,
            entity_id: str,
            entity_dict: EntityDict,
            n_hop: int = 2,
            max_nodes: int = 100000
    ) -> Set[int]:
        """Returns entity indices within n_hop distance. Returns empty set if max_nodes exceeded."""
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


def reverse_triplet(triplet: dict) -> dict:
    """Creates an inverse version of a triplet."""
    return {
        'head_id': triplet['tail_id'],
        'head': triplet['tail'],
        'relation': f"inverse {triplet['relation']}",
        'tail_id': triplet['head_id'],
        'tail': triplet['head']
    }
