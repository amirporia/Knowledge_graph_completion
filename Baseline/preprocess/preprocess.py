import argparse
import json
import multiprocessing as mp
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Any

# ============================================================================
# Configuration
# ============================================================================

# Current task name
CURRENT_TASK_NAME = "WN18RR"

# Get the script's directory (works everywhere)
SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()

# Dataset global variables (initialized lazily or as empty dicts)
DATASET_VARS = {
    'WN18RR': {
        'id2ent': {}
    },
    'FB15k237': {
        'id2ent': {},
        'id2desc': {}
    },
    'WiKi5m': {
        'id2rel': {},
        'id2ent': {},
        'id2text': {}
    }
}

# Backward compatibility references
wn18rr_id2ent = DATASET_VARS['WN18RR']['id2ent']
fb15k_id2ent = DATASET_VARS['FB15k237']['id2ent']
fb15k_id2desc = DATASET_VARS['FB15k237']['id2desc']
wiki5m_id2rel = DATASET_VARS['WiKi5m']['id2rel']
wiki5m_id2ent = DATASET_VARS['WiKi5m']['id2ent']
wiki5m_id2text = DATASET_VARS['WiKi5m']['id2text']

# Constants
SUPPORTED_TASKS = {'wn18rr', 'FB15k237', 'wiki5m_trans', 'wiki5m_ind'}


# ============================================================================
# Argument Parser Setup
# ============================================================================

def setup_parser():
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(description='Preprocess dataset')

    parser.add_argument(
        '--task',
        default=CURRENT_TASK_NAME,
        type=str,
        help='dataset name'
    )
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='number of workers'
    )
    parser.add_argument(
        '--train-path',
        type=str,
        help='path to training data'
    )
    parser.add_argument(
        '--valid-path',
        type=str,
        help='path to validation data'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        help='path to test data'
    )

    return parser


def set_default_paths(args, script_dir):
    """Set default paths if not provided."""
    if not args.train_path:
        args.train_path = str(script_dir / 'data' / args.task / 'train.txt')
    if not args.valid_path:
        args.valid_path = str(script_dir / 'data' / args.task / 'valid.txt')
    if not args.test_path:
        args.test_path = str(script_dir / 'data' / args.task / 'test.txt')
    return args


# ============================================================================
# Relation Normalization
# ============================================================================

def _check_sanity(relation_id_to_str: dict) -> None:
    """
    Verify that no two relations are normalized to the same surface form.

    Args:
        relation_id_to_str: Mapping from relation ID to its normalized string
    """
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


def _normalize_relations(
        examples: List[dict],
        normalize_fn: callable,
        train_path: str = None
) -> None:
    """
    Normalize relation strings in examples and optionally save the mapping.

    Args:
        examples: List of example dictionaries containing 'relation' keys
        normalize_fn: Function to normalize relation strings
        train_path: Optional path to training file for saving relation mapping
    """
    relation_id_to_str = {}

    # Normalize all relations
    for example in examples:
        original_relation = example['relation']
        normalized_relation = normalize_fn(original_relation)

        relation_id_to_str[original_relation] = normalized_relation
        example['relation'] = normalized_relation

    # Verify no duplicate normalizations
    _check_sanity(relation_id_to_str)

    # Save mapping if training path is provided
    if train_path:
        _save_relation_mapping(relation_id_to_str, train_path)


def _save_relation_mapping(relation_id_to_str: dict, train_path: str) -> None:
    """Save relation ID to string mapping to a JSON file."""
    output_dir = os.path.dirname(train_path)
    output_path = os.path.join(output_dir, 'relations.json')

    with open(output_path, 'w', encoding='utf-8') as writer:
        json.dump(relation_id_to_str, writer, ensure_ascii=False, indent=4)

    print(f"Saved {len(relation_id_to_str)} relations to {output_path}")


# ============================================================================
# Load Datasets
# ============================================================================

def _load_wn18rr_texts(path: str) -> None:
    """Load WordNet18RR entity data from file."""
    global wn18rr_id2ent

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) != 3:
                raise ValueError(f'Invalid line (expected 3 columns): {line.strip()}')

            entity_id, word, desc = parts
            word = word.replace('__', '')
            wn18rr_id2ent[entity_id] = (entity_id, word, desc)

    print(f'Loaded {len(wn18rr_id2ent)} entities from {path}')


def _load_fb15k237_wikidata(path: str) -> None:
    """Load FB15k-237 Wikidata entity names and descriptions."""
    global fb15k_id2ent, fb15k_id2desc

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) != 2:
                raise ValueError(f'Invalid line (expected 2 columns): {line.strip()}')

            entity_id, name = parts
            name = name.replace('_', ' ').strip()

            if entity_id not in fb15k_id2desc:
                print(f'Warning: No description found for {entity_id}')

            description = fb15k_id2desc.get(entity_id, '')
            fb15k_id2ent[entity_id] = (entity_id, name, description)

    print(f'Loaded {len(fb15k_id2ent)} entity names from {path}')


def _load_fb15k237_desc(path: str) -> None:
    """
    Load FB15k-237 entity descriptions from a tab-separated file.

    Args:
        path: Path to the description file with format: entity_id\tDescription text

    The function stores descriptions in the global variable fb15k_id2desc,
    truncating each description to 50 characters.
    """
    global fb15k_id2desc

    try:
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Description file not found: {path}")

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        parts = line.split('\t')

        if len(parts) != 2:
            raise ValueError(
                f"Invalid format at line {line_num}: Expected 2 tab-separated fields, "
                f"got {len(parts)} in: {line}"
            )

        entity_id, description = parts[0], parts[1]
        fb15k_id2desc[entity_id] = _truncate(description, 50)

    print(f"Loaded {len(fb15k_id2desc)} entity descriptions from {path}")


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to at most max_len words."""
    words = text.split()
    return ' '.join(words[:max_len])


def _load_wiki5m_id2rel(path: str) -> None:
    """Load Wiki5M relation data."""
    global wiki5m_id2rel

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) < 2:
                raise ValueError(f'Invalid line (expected at least 2 columns): {line.strip()}')

            rel_id, rel_text = parts[0], parts[1]
            wiki5m_id2rel[rel_id] = _truncate(rel_text, 10)

    print(f'Loaded {len(wiki5m_id2rel)} relations from {path}')


def _load_wiki5m_id2ent(path: str) -> None:
    """Load Wiki5M entity names."""
    global wiki5m_id2ent

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) < 2:
                raise ValueError(f'Invalid line (expected at least 2 columns): {line.strip()}')

            ent_id, ent_name = parts[0], parts[1]
            wiki5m_id2ent[ent_id] = _truncate(ent_name, 10)

    print(f'Loaded {len(wiki5m_id2ent)} entity names from {path}')


def _load_wiki5m_id2text(path: str, max_len: int = 30) -> None:
    """Load Wiki5M entity text descriptions."""
    global wiki5m_id2text

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) < 2:
                raise ValueError(f'Invalid line (expected at least 2 columns): {line.strip()}')

            ent_id = parts[0]
            ent_text = ' '.join(parts[1:])
            wiki5m_id2text[ent_id] = _truncate(ent_text, max_len)

    print(f'Loaded {len(wiki5m_id2text)} entity texts from {path}')


# ============================================================================
# Process Line
# ============================================================================

def _process_line(line: str, id2ent: dict, dataset: str = "wn18rr") -> dict:
    """
    Process a line from a knowledge graph dataset into an example dict.

    Args:
        line: Tab-separated string with head, relation, tail
        id2ent: Mapping from entity IDs to entity information
        dataset: Dataset name ("wn18rr", "FB15k237", or "wiki5m")

    Returns:
        Dictionary with head_id, head, relation, tail_id, tail
    """
    fields = line.strip().split('\t')
    assert len(fields) == 3, f'Expected 3 fields, got {len(fields)}: {line.strip()}'

    head_id, relation, tail_id = fields[0], fields[1], fields[2]

    if dataset == "wiki5m":
        return {
            'head_id': head_id,
            'head': id2ent.get(head_id, None),
            'relation': relation,
            'tail_id': tail_id,
            'tail': id2ent.get(tail_id, None)
        }
    else:  # wn18rr or FB15k237
        _, head, _ = id2ent[head_id]
        _, tail, _ = id2ent[tail_id]
        return {
            'head_id': head_id,
            'head': head,
            'relation': relation,
            'tail_id': tail_id,
            'tail': tail
        }


# Convenience wrappers for backward compatibility
def _process_line_wn18rr(line: str, id2ent: dict) -> dict:
    return _process_line(line, id2ent, "wn18rr")


def _process_line_fb15k237(line: str, id2ent: dict) -> dict:
    return _process_line(line, id2ent, "FB15k237")


def _process_line_wiki5m(line: str, id2ent: dict) -> dict:
    return _process_line(line, id2ent, "wiki5m")


def preprocess_wn18rr(path, num_workers: int, train_path: str):
    if not wn18rr_id2ent:
        _load_wn18rr_texts(os.path.join(os.path.dirname(path), 'wordnet-mlj12-definitions.txt'))

    lines = open(path, 'r', encoding='utf-8').readlines()

    # Create a partial function with the dictionaries
    from functools import partial
    process_func = partial(_process_line_wn18rr,
                           id2ent=wn18rr_id2ent)

    pool = Pool(processes=num_workers)
    examples = pool.map(process_func, lines)
    pool.close()
    pool.join()

    _normalize_relations(examples, normalize_fn=lambda rel: rel.replace('_', ' ').strip(),
                         train_path=train_path if train_path == path else None)

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').replace('_', ' ').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens)
                         if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])
    return relation


def preprocess_fb15k237(path, num_workers: int, train_path: str):
    if not fb15k_id2desc:
        _load_fb15k237_desc(os.path.join(os.path.dirname(path), 'FB15k_mid2description.txt'))
    if not fb15k_id2ent:
        _load_fb15k237_wikidata(os.path.join(os.path.dirname(path), 'FB15k_mid2name.txt'))

    lines = open(path, 'r', encoding='utf-8').readlines()

    # Create a partial function with the dictionaries
    from functools import partial
    process_func = partial(_process_line_fb15k237, id2ent=fb15k_id2ent)
    pool = Pool(processes=num_workers)
    examples = pool.map(process_func, lines)
    pool.close()
    pool.join()

    _normalize_relations(examples, normalize_fn=_normalize_fb15k237_relation,
                         train_path=train_path if train_path == path else None)

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


def _has_none_value(ex: dict) -> bool:
    return any(v is None for v in ex.values())


def preprocess_wiki5m(path: str, num_workers: int, train_path: str) -> List[dict]:
    if not wiki5m_id2rel:
        _load_wiki5m_id2rel(path=os.path.join(os.path.dirname(path), 'wikidata5m_relation.txt'))
    if not wiki5m_id2ent:
        _load_wiki5m_id2ent(path=os.path.join(os.path.dirname(path), 'wikidata5m_entity.txt'))
    if not wiki5m_id2text:
        _load_wiki5m_id2text(path=os.path.join(os.path.dirname(path), 'wikidata5m_text.txt'))

    lines = open(path, 'r', encoding='utf-8').readlines()

    from functools import partial
    process_func = partial(_process_line_wiki5m,
                           id2ent=_load_wiki5m_id2ent)

    pool = Pool(processes=num_workers)
    examples = pool.map(process_func, lines)
    pool.close()
    pool.join()

    _normalize_relations(examples, normalize_fn=lambda rel_id: wiki5m_id2rel.get(rel_id, None),
                         train_path=train_path if train_path == path else None)

    invalid_examples = [ex for ex in examples if _has_none_value(ex)]
    print('Find {} invalid examples in {}'.format(len(invalid_examples), path))
    if train_path == path:
        # P2439 P1962 P3484 do not exist in wikidata5m_relation.txt
        # so after filtering, there are 819 relations instead of 822 relations
        examples = [ex for ex in examples if not _has_none_value(ex)]
    else:
        # Even though it's invalid (contains null values), we should not change validation/test dataset
        print('Invalid examples: {}'.format(json.dumps(invalid_examples, ensure_ascii=False, indent=4)))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


def dump_all_entities(examples, out_path, id2text: dict):
    id2entity = {}
    relations = set()

    for ex in examples:
        head_id = ex['head_id']
        tail_id = ex['tail_id']

        relations.add(ex['relation'])

        # Add head entity if not exists
        if head_id not in id2entity:
            id2entity[head_id] = {
                'entity_id': head_id,
                'entity': ex['head'],
                'entity_desc': id2text[head_id]
            }

        # Add tail entity if not exists
        if tail_id not in id2entity:
            id2entity[tail_id] = {
                'entity_id': tail_id,
                'entity': ex['tail'],
                'entity_desc': id2text[tail_id]
            }

    print(f'Get {len(id2entity)} entities, {len(relations)} relations in total')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(list(id2entity.values()), f, ensure_ascii=False, indent=4)


# Task-specific entity mappings
def get_entity_mapping(task: str) -> Dict:
    """Get the entity to text mapping for the given task."""
    mappings = {
        'wn18rr': lambda: {k: v[2] for k, v in wn18rr_id2ent.items()},
        'FB15k237': lambda: {k: v[2] for k, v in fb15k_id2ent.items()},
        'wiki5m_trans': lambda: wiki5m_id2text,
        'wiki5m_ind': lambda: wiki5m_id2text,
    }

    if task not in mappings:
        raise ValueError(f'Unknown task: {task}')

    return mappings[task]()


def setup_multiprocessing():
    """Configure multiprocessing start method based on platform."""
    if sys.platform != 'win32':
        mp.set_start_method('fork', force=True)
    else:
        mp.set_start_method('spawn', force=True)


def validate_file_paths(paths: List[str]) -> None:
    """Validate that all file paths exist."""
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File with path '{path}' does not exist...")


def load_and_preprocess_data(args: Any):
    """Load and preprocess all data files."""
    all_examples = []
    file_paths = [args.train_path, args.valid_path, args.test_path]

    validate_file_paths(file_paths)

    task_name = args.task.lower()

    # Task-specific preprocessing functions
    TASK_PREPROCESSORS = {
        'wn18rr': preprocess_wn18rr,
        'fb15k237': preprocess_fb15k237,
        'wiki5m_trans': preprocess_wiki5m,
        'wiki5m_ind': preprocess_wiki5m,
    }
    preprocessor = TASK_PREPROCESSORS.get(task_name)

    if not preprocessor:
        raise ValueError(f'Unknown task: {task_name}')

    for path in file_paths:
        print(f'Processing {path}...')
        examples = preprocessor(path, args.workers, args.train_path)
        all_examples.extend(examples)

    return all_examples, task_name


def dump_entities_to_file(all_examples: List, output_dir: str, id2text: Dict) -> None:
    """Dump all entities to a JSON file."""
    output_path = os.path.join(output_dir, 'entities.json')
    dump_all_entities(all_examples, out_path=output_path, id2text=id2text)


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    args = set_default_paths(args, SCRIPT_DIR)

    # Configure multiprocessing
    setup_multiprocessing()

    # Load and preprocess data
    all_examples, task_name = load_and_preprocess_data(args)

    # Get entity mapping
    id2text = get_entity_mapping(task_name)

    # Dump entities
    output_dir = os.path.dirname(args.test_path)
    dump_entities_to_file(all_examples, output_dir, id2text)

    print('Done')


if __name__ == '__main__':
    main()
