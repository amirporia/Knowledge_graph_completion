"""
Independent raw-data preprocessing for WN18RR and FB15k-237 (Sec 7.1's two
primary benchmarks), producing the JSON format consumed by `data.dataset`:

    <split>.txt.json  ->  [{head_id, head, relation, tail_id, tail}, ...]
    entities.json     ->  [{entity_id, entity, entity_desc}, ...]
    relations.json    ->  {relation_string: relation_id, ...}

This is a from-scratch reimplementation (not imported from `ours`), kept
independent so this package never depends on the separate baseline
codebase. Expected raw layout (standard public releases of each dataset),
under `<script_dir>/data/<TASK>/`:

    WN18RR:     train.txt, valid.txt, test.txt, wordnet-mlj12-definitions.txt
    FB15k-237:  train.txt, valid.txt, test.txt,
                FB15k_mid2name.txt, FB15k_mid2description.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent.parent.absolute()


# ---------------------------------------------------------------------------
# WN18RR
# ---------------------------------------------------------------------------

def _load_wn18rr_definitions(path: str) -> Dict[str, Tuple[str, str]]:
    """entity_id -> (name, description), from wordnet-mlj12-definitions.txt
    (format: entity_id \t __name_pos_num__ \t description)."""
    id2ent = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid WN18RR definitions line: {line.strip()}")
            entity_id, name, desc = parts
            id2ent[entity_id] = (name.replace("__", "").replace("_", " "), desc)
    return id2ent


def preprocess_wn18rr(raw_dir: str, out_dir: str) -> None:
    id2ent = _load_wn18rr_definitions(os.path.join(raw_dir, "wordnet-mlj12-definitions.txt"))
    relation_id_to_str = {}

    for split in ("train", "valid", "test"):
        src = os.path.join(raw_dir, f"{split}.txt")
        examples = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) != 3:
                    continue
                head_id, relation, tail_id = fields
                relation_norm = relation.replace("_", " ").strip()
                relation_id_to_str[relation] = relation_norm
                head_name, _ = id2ent.get(head_id, (head_id, ""))
                tail_name, _ = id2ent.get(tail_id, (tail_id, ""))
                examples.append({
                    "head_id": head_id, "head": head_name, "relation": relation_norm,
                    "tail_id": tail_id, "tail": tail_name,
                })
        _dump(examples, os.path.join(out_dir, f"{split}.txt.json"))

    entities = [
        {"entity_id": eid, "entity": name, "entity_desc": desc}
        for eid, (name, desc) in id2ent.items()
    ]
    _dump(entities, os.path.join(out_dir, "entities.json"))
    _dump_relations(relation_id_to_str, out_dir)


# ---------------------------------------------------------------------------
# FB15k-237
# ---------------------------------------------------------------------------

def _load_fb15k_descriptions(path: str, max_words: int = 50) -> Dict[str, str]:
    id2desc = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            entity_id, desc = parts
            id2desc[entity_id] = " ".join(desc.split()[:max_words])
    return id2desc


def _load_fb15k_names(path: str, id2desc: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    id2ent = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            entity_id, name = parts
            id2ent[entity_id] = (name.replace("_", " ").strip(), id2desc.get(entity_id, ""))
    return id2ent


def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace("./", "/").replace("_", " ").strip().split("/")
    deduped: List[str] = []
    for tok in tokens:
        if tok not in deduped[-3:]:
            deduped.append(tok)
    reversed_tokens = deduped[::-1]
    out_tokens = [t for idx, t in enumerate(reversed_tokens)
                  if idx == 0 or reversed_tokens[idx] != reversed_tokens[idx - 1]]
    return " ".join(out_tokens)


def preprocess_fb15k237(raw_dir: str, out_dir: str) -> None:
    id2desc = _load_fb15k_descriptions(os.path.join(raw_dir, "FB15k_mid2description.txt"))
    id2ent = _load_fb15k_names(os.path.join(raw_dir, "FB15k_mid2name.txt"), id2desc)
    relation_id_to_str = {}

    for split in ("train", "valid", "test"):
        src = os.path.join(raw_dir, f"{split}.txt")
        examples = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) != 3:
                    continue
                head_id, relation, tail_id = fields
                relation_norm = _normalize_fb15k237_relation(relation)
                relation_id_to_str[relation] = relation_norm
                head_name, _ = id2ent.get(head_id, (head_id, ""))
                tail_name, _ = id2ent.get(tail_id, (tail_id, ""))
                examples.append({
                    "head_id": head_id, "head": head_name, "relation": relation_norm,
                    "tail_id": tail_id, "tail": tail_name,
                })
        _dump(examples, os.path.join(out_dir, f"{split}.txt.json"))

    entities = [
        {"entity_id": eid, "entity": name, "entity_desc": desc}
        for eid, (name, desc) in id2ent.items()
    ]
    _dump(entities, os.path.join(out_dir, "entities.json"))
    _dump_relations(relation_id_to_str, out_dir)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(obj)} records to {path}")


def _dump_relations(relation_id_to_str: Dict[str, str], out_dir: str) -> None:
    """Relation vocabulary (Sec 4.2's `r` embeddings) includes the inverse
    ('inverse <relation>') of every forward relation, matching
    `data.triplets.reverse_triplet`'s convention used at load time."""
    all_relations = sorted(set(relation_id_to_str.values()))
    all_relations += [f"inverse {r}" for r in all_relations]
    relation2id = {r: i for i, r in enumerate(sorted(set(all_relations)))}
    _dump(relation2id, os.path.join(out_dir, "relations.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ARPM-KGC data preprocessing")
    parser.add_argument("--task", choices=["wn18rr", "fb15k237"], required=True)
    parser.add_argument("--raw-dir", type=str, default=None,
                         help="Directory with raw train/valid/test.txt (+ dataset-specific "
                              "text files). Defaults to data/<TASK_UPPER>/")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Output directory. Defaults to data/<TASK_UPPER>")
    args = parser.parse_args()

    task_dir = SCRIPT_DIR / "data" / args.task.upper()
    raw_dir = args.raw_dir or str(task_dir)
    out_dir = args.out_dir or str(task_dir)

    if args.task == "wn18rr":
        preprocess_wn18rr(raw_dir, out_dir)
    elif args.task == "fb15k237":
        preprocess_fb15k237(raw_dir, out_dir)

    print("Done")


if __name__ == "__main__":
    main()
