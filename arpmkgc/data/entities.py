"""Entity vocabulary: id <-> index mapping and text lookup."""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ""


class EntityDict:
    """Loads `entities.json` (list of {entity_id, entity, entity_desc}) and
    exposes id<->index lookups used throughout scoring/ranking.
    """

    def __init__(self, entity_dict_dir: str, inductive_filter_path: Optional[str] = None) -> None:
        entities_path = os.path.join(entity_dict_dir, "entities.json")
        if not os.path.exists(entities_path):
            raise FileNotFoundError(f"Entities file not found: {entities_path}")

        with open(entities_path, "r", encoding="utf-8") as f:
            self.entity_exs: List[EntityExample] = [EntityExample(**obj) for obj in json.load(f)]

        if inductive_filter_path is not None:
            self._filter_to(inductive_filter_path)

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}

    def _filter_to(self, examples_path: str) -> None:
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        keep = set()
        for ex in examples:
            keep.add(ex["head_id"])
            keep.add(ex["tail_id"])
        self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in keep]

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self) -> int:
        return len(self.entity_exs)
