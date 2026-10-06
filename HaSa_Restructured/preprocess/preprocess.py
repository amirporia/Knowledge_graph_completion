import argparse
import json
import multiprocessing as mp
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ============================================================================
# Configuration
# ============================================================================

# Current task name
CURRENT_TASK_NAME = "wn18rr"

# Project root (the folder that contains this package and `data/`)
SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()

SUPPORTED_TASKS = {'wn18rr', 'fb15k237', 'wiki5m_trans', 'wiki5m_ind'}

# Dataset lookups. They are module-level on purpose: worker processes are forked after they
# are filled, so large tables (wikidata5m) are shared instead of being pickled per task.
wn18rr_id2ent: Dict[str, tuple] = {}
fb15k_id2ent: Dict[str, tuple] = {}
fb15k_id2desc: Dict[str, str] = {}
wiki5m_id2rel: Dict[str, str] = {}
wiki5m_id2ent: Dict[str, str] = {}
wiki5m_id2text: Dict[str, str] = {}


# ============================================================================
# Argument Parser Setup
# ============================================================================

def setup_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--task', default=CURRENT_TASK_NAME, type=str.lower, choices=sorted(SUPPORTED_TASKS),
                        help='dataset name')
    parser.add_argument('--workers', default=2, type=int, help='number of workers')
    parser.add_argument('--train-path', type=str, help='path to training data')
    parser.add_argument('--valid-path', type=str, help='path to validation data')
    parser.add_argument('--test-path', type=str, help='path to test data')
    return parser


def set_default_paths(args, script_dir: Path):
    """Set default paths if not provided."""
    if not args.train_path:
        args.train_path = str(script_dir / 'data' / args.task / 'train.txt')
    if not args.valid_path:
        args.valid_path = str(script_dir / 'data' / args.task / 'valid.txt')
    if not args.test_path:
        args.test_path = str(script_dir / 'data' / args.task / 'test.txt')
    return args


# ============================================================================
# Shared helpers
# ============================================================================

def _truncate(text: str, max_len: int) -> str:
    """Truncate text to at most max_len words."""
    return ' '.join(text.split()[:max_len])


def _check_sanity(relation_id_to_str: dict) -> None:
    """Verify that no two relations are normalized to the same surface form."""
    relation_str_to_id = {}

    for rel_id, rel_str in relation_id_to_str.items():
        if rel_str is None:
            continue

        if rel_str not in relation_str_to_id:
            relation_str_to_id[rel_str] = rel_id
        elif relation_str_to_id[rel_str] != rel_id:
            raise ValueError(
                f"Relations {relation_str_to_id[rel_str]} and {rel_id} "
                f"are both normalized to '{rel_str}'"
            )


def _normalize_relations(examples: List[dict], normalize_fn: Callable,
                         relations_out_path: Optional[str] = None) -> None:
    """Normalize relation strings in place; optionally save the id -> string mapping."""
    relation_id_to_str = {}

    for ex in examples:
        rel_str = normalize_fn(ex['relation'])
        relation_id_to_str[ex['relation']] = rel_str
        ex['relation'] = rel_str

    _check_sanity(relation_id_to_str)

    if relations_out_path:
        with open(relations_out_path, 'w', encoding='utf-8') as writer:
            json.dump(relation_id_to_str, writer, ensure_ascii=False, indent=4)
        print(f'Save {len(relation_id_to_str)} relations to {relations_out_path}')


def _map_lines(fn: Callable, lines: List[str], num_workers: int) -> List[dict]:
    """Apply `fn` to every line, in parallel when possible."""
    # Windows has no fork(): children would not see the module-level lookup tables.
    if num_workers <= 1 or sys.platform == 'win32':
        return [fn(line) for line in lines]

    with Pool(processes=num_workers) as pool:
        return pool.map(fn, lines)


def _save_examples(examples: List[dict], path: str) -> None:
    out_path = path + '.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)
    print(f'Save {len(examples)} examples to {out_path}')


def _read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def _split_line(line: str, expected: int) -> List[str]:
    fields = line.strip().split('\t')
    assert len(fields) == expected, f'Expect {expected} fields for {line.strip()}'
    return fields


def _build_example(head_id: str, relation: str, tail_id: str, head, tail) -> dict:
    return {'head_id': head_id, 'head': head, 'relation': relation,
            'tail_id': tail_id, 'tail': tail}


# ============================================================================
# WN18RR
# ============================================================================

def _load_wn18rr_texts(path: str) -> None:
    for line in _read_lines(path):
        entity_id, word, desc = _split_line(line, 3)
        wn18rr_id2ent[entity_id] = (entity_id, word.replace('__', ''), desc)
    print(f'Load {len(wn18rr_id2ent)} entities from {path}')


def _process_line_wn18rr(line: str) -> dict:
    head_id, relation, tail_id = _split_line(line, 3)
    _, head, _ = wn18rr_id2ent[head_id]
    _, tail, _ = wn18rr_id2ent[tail_id]
    return _build_example(head_id, relation, tail_id, head, tail)


def preprocess_wn18rr(path: str, num_workers: int, relations_out_path: Optional[str]) -> List[dict]:
    if not wn18rr_id2ent:
        _load_wn18rr_texts(os.path.join(os.path.dirname(path), 'wordnet-mlj12-definitions.txt'))

    examples = _map_lines(_process_line_wn18rr, _read_lines(path), num_workers)

    _normalize_relations(examples, lambda rel: rel.replace('_', ' ').strip(), relations_out_path)

    _save_examples(examples, path)
    return examples


# ============================================================================
# FB15k-237
# ============================================================================

def _load_fb15k237_desc(path: str) -> None:
    for line in _read_lines(path):
        entity_id, desc = _split_line(line, 2)
        fb15k_id2desc[entity_id] = _truncate(desc, 50)
    print(f'Load {len(fb15k_id2desc)} entity descriptions from {path}')


def _load_fb15k237_wikidata(path: str) -> None:
    for line in _read_lines(path):
        entity_id, name = _split_line(line, 2)
        name = name.replace('_', ' ').strip()
        if entity_id not in fb15k_id2desc:
            print(f'No desc found for {entity_id}')
        fb15k_id2ent[entity_id] = (entity_id, name, fb15k_id2desc.get(entity_id, ''))
    print(f'Load {len(fb15k_id2ent)} entity names from {path}')


def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').replace('_', ' ').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    return ' '.join([t for idx, t in enumerate(relation_tokens)
                     if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])


def _process_line_fb15k237(line: str) -> dict:
    head_id, relation, tail_id = _split_line(line, 3)
    _, head, _ = fb15k_id2ent[head_id]
    _, tail, _ = fb15k_id2ent[tail_id]
    return _build_example(head_id, relation, tail_id, head, tail)


def preprocess_fb15k237(path: str, num_workers: int, relations_out_path: Optional[str]) -> List[dict]:
    if not fb15k_id2desc:
        _load_fb15k237_desc(os.path.join(os.path.dirname(path), 'FB15k_mid2description.txt'))
    if not fb15k_id2ent:
        _load_fb15k237_wikidata(os.path.join(os.path.dirname(path), 'FB15k_mid2name.txt'))

    examples = _map_lines(_process_line_fb15k237, _read_lines(path), num_workers)

    _normalize_relations(examples, _normalize_fb15k237_relation, relations_out_path)

    _save_examples(examples, path)
    return examples


# ============================================================================
# Wikidata5M (transductive / inductive)
# ============================================================================

def _load_wiki5m_id2rel(path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            fs = line.strip().split('\t')
            assert len(fs) >= 2, f'Invalid line: {line.strip()}'
            wiki5m_id2rel[fs[0]] = _truncate(fs[1], 10)
    print(f'Load {len(wiki5m_id2rel)} relations from {path}')


def _load_wiki5m_id2ent(path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            fs = line.strip().split('\t')
            assert len(fs) >= 2, f'Invalid line: {line.strip()}'
            wiki5m_id2ent[fs[0]] = _truncate(fs[1], 10)
    print(f'Load {len(wiki5m_id2ent)} entity names from {path}')


def _load_wiki5m_id2text(path: str, max_len: int = 30) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            fs = line.strip().split('\t')
            assert len(fs) >= 2, f'Invalid line: {line.strip()}'
            wiki5m_id2text[fs[0]] = _truncate(' '.join(fs[1:]), max_len)
    print(f'Load {len(wiki5m_id2text)} entity texts from {path}')


def _has_none_value(ex: dict) -> bool:
    return any(v is None for v in ex.values())


def _process_line_wiki5m(line: str) -> dict:
    head_id, relation_id, tail_id = _split_line(line, 3)
    return _build_example(head_id, relation_id, tail_id,
                          wiki5m_id2ent.get(head_id, None),
                          wiki5m_id2ent.get(tail_id, None))


def preprocess_wiki5m(path: str, num_workers: int, relations_out_path: Optional[str]) -> List[dict]:
    base_dir = os.path.dirname(path)
    if not wiki5m_id2rel:
        _load_wiki5m_id2rel(os.path.join(base_dir, 'wikidata5m_relation.txt'))
    if not wiki5m_id2ent:
        _load_wiki5m_id2ent(os.path.join(base_dir, 'wikidata5m_entity.txt'))
    if not wiki5m_id2text:
        _load_wiki5m_id2text(os.path.join(base_dir, 'wikidata5m_text.txt'))

    is_train = relations_out_path is not None
    examples = _map_lines(_process_line_wiki5m, _read_lines(path), num_workers)

    _normalize_relations(examples, lambda rel_id: wiki5m_id2rel.get(rel_id, None), relations_out_path)

    invalid_examples = [ex for ex in examples if _has_none_value(ex)]
    print(f'Find {len(invalid_examples)} invalid examples in {path}')
    if is_train:
        # P2439 P1962 P3484 do not exist in wikidata5m_relation.txt
        # so after filtering, there are 819 relations instead of 822 relations
        examples = [ex for ex in examples if not _has_none_value(ex)]
    else:
        # Even though it's invalid (contains null values), we should not change validation/test dataset
        print('Invalid examples: {}'.format(json.dumps(invalid_examples, ensure_ascii=False, indent=4)))

    _save_examples(examples, path)
    return examples


# ============================================================================
# Entities dump
# ============================================================================

def dump_all_entities(examples: List[dict], out_path: str, id2text: dict) -> None:
    id2entity = {}
    relations = set()

    for ex in examples:
        relations.add(ex['relation'])

        for id_key, name_key in (('head_id', 'head'), ('tail_id', 'tail')):
            ent_id = ex[id_key]
            if ent_id not in id2entity:
                id2entity[ent_id] = {
                    'entity_id': ent_id,
                    'entity': ex[name_key],
                    'entity_desc': id2text[ent_id]
                }

    print(f'Get {len(id2entity)} entities, {len(relations)} relations in total')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(list(id2entity.values()), f, ensure_ascii=False, indent=4)


def get_entity_mapping(task: str) -> Dict:
    """Get the entity -> description mapping for the given task."""
    mappings = {
        'wn18rr': lambda: {k: v[2] for k, v in wn18rr_id2ent.items()},
        'fb15k237': lambda: {k: v[2] for k, v in fb15k_id2ent.items()},
        'wiki5m_trans': lambda: wiki5m_id2text,
        'wiki5m_ind': lambda: wiki5m_id2text,
    }

    if task not in mappings:
        raise ValueError(f'Unknown task: {task}')

    return mappings[task]()


# ============================================================================
# Main
# ============================================================================

TASK_PREPROCESSORS = {
    'wn18rr': preprocess_wn18rr,
    'fb15k237': preprocess_fb15k237,
    'wiki5m_trans': preprocess_wiki5m,
    'wiki5m_ind': preprocess_wiki5m,
}


def setup_multiprocessing() -> None:
    """Use fork where available so workers share the module-level lookup tables."""
    if sys.platform != 'win32':
        mp.set_start_method('fork', force=True)


def validate_file_paths(paths: List[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File with path '{path}' does not exist...")


def load_and_preprocess_data(args):
    """Preprocess train/valid/test files and return all examples."""
    file_paths = [args.train_path, args.valid_path, args.test_path]
    validate_file_paths(file_paths)

    preprocessor = TASK_PREPROCESSORS.get(args.task)
    if preprocessor is None:
        raise ValueError(f'Unknown task: {args.task}')

    train_relations_path = os.path.join(os.path.dirname(args.train_path), 'relations.json')

    all_examples = []
    for path in file_paths:
        print(f'Process {path}...')
        relations_out_path = train_relations_path if path == args.train_path else None
        all_examples += preprocessor(path, args.workers, relations_out_path)

    return all_examples


def main():
    args = setup_parser().parse_args()
    args = set_default_paths(args, SCRIPT_DIR)

    setup_multiprocessing()

    all_examples = load_and_preprocess_data(args)

    id2text = get_entity_mapping(args.task)
    dump_all_entities(
        all_examples,
        out_path=os.path.join(os.path.dirname(args.train_path), 'entities.json'),
        id2text=id2text
    )

    print('Done')


if __name__ == '__main__':
    main()
