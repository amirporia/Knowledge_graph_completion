import json
import os
from typing import Dict

import torch

from .evaluate import evaluate_predictor, resolve_coefs, current_coefs, BEST_COEFS_NAME
from .predict import ARPMPredictor
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict

SHARD_SIZE = 1_000_000
EVAL_BATCH_SIZE = 64          # queries scored per step (scoring is chunked over entities, see evaluate.py)
TASK_REQUIREMENT = 'wiki5m_trans'

assert args.task == TASK_REQUIREMENT, (
    f'This script is only used for {TASK_REQUIREMENT} transduction setting'
)

entity_dict = get_entity_dict()


def get_shard_path(shard_id: int = 0) -> str:
    # ".fp16" in the name: shards written by the older fp32 version are never picked up by mistake
    return os.path.join(args.model_dir, f'shard_{shard_id}.fp16')


def dump_entity_embeddings(predictor: ARPMPredictor) -> None:
    """Encode all entities and store them in fp16 shards of 1M entities (7.4 GB instead of 14.8 GB on disk)."""
    for start in range(0, len(entity_dict), SHARD_SIZE):
        end = min(start + SHARD_SIZE, len(entity_dict))
        shard_id = start // SHARD_SIZE
        shard_path = get_shard_path(shard_id)

        if os.path.exists(shard_path):
            logger.info(f'{shard_path} already exists')
            continue

        logger.info(f'Processing shard_id={shard_id}, from {start} to {end}')

        shard_entity_exs = entity_dict.entity_exs[start:end]
        shard_entity_tensor = predictor.predict_by_entities(shard_entity_exs)

        tmp_path = shard_path + '.tmp'            # atomic: a killed run never leaves a truncated shard
        torch.save(shard_entity_tensor.half().cpu(), tmp_path)
        os.replace(tmp_path, shard_path)
        del shard_entity_tensor
        torch.cuda.empty_cache()

        logger.info(f'Completed shard_id={shard_id}')


def load_entity_embeddings(device: torch.device) -> torch.Tensor:
    """Load all shards straight into ONE fp16 tensor on `device` (4.8M x 768 x 2 B = 7.4 GB).
    Nothing is concatenated on the CPU (that needed 2 x 14.8 GB of host RAM before)."""
    n = len(entity_dict)
    assert os.path.exists(get_shard_path(0)), f'No shards found at {get_shard_path(0)}'

    entity_tensor, offset = None, 0
    for start in range(0, n, SHARD_SIZE):
        shard_path = get_shard_path(start // SHARD_SIZE)
        shard = torch.load(shard_path, map_location='cpu')
        if entity_tensor is None:
            entity_tensor = torch.empty(n, shard.size(1), dtype=torch.float16, device=device)
        entity_tensor[offset:offset + shard.size(0)] = shard.to(device)
        offset += shard.size(0)
        logger.info(f'Loaded {shard.size(0)} entity embeddings from {shard_path}')
        del shard

    assert offset == n, f'Expected {n} embeddings, got {offset}'
    logger.info(f'Total entity embeddings: {offset} on {device} ({entity_tensor.dtype})')
    return entity_tensor


def validate_paths() -> None:
    required_paths = {
        'valid_path': args.valid_path,
        'train_path': args.train_path,
        'eval_model_path': args.eval_model_path,
    }
    for path_name, path in required_paths.items():
        assert os.path.exists(path), f'{path_name} does not exist: {path}'


def save_metrics(
        forward_metrics: Dict[str, float],
        backward_metrics: Dict[str, float],
        averaged_metrics: Dict[str, float]
) -> None:
    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)

    output_path = os.path.join(prefix, f'metrics_{split}_{basename}.json')

    c_h, c_p, c_s = current_coefs()
    metrics_data = {
        'coefficients': {'coef_h': c_h, 'coef_p': c_p, 'coef_s': c_s},
        'forward_metrics': forward_metrics,
        'backward_metrics': backward_metrics,
        'average_metrics': averaged_metrics,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f'Metrics saved to {output_path}')


def predict_by_split() -> None:
    # NOTE: the old `args.batch_size = max(args.batch_size, n_gpu * 1024)` is gone. It was meant for plain
    # entity encoding, but also became the QUERY batch size (~30 sequences per query -> 60k sequences per step).
    # Entity encoding uses --entity-batch-size instead.
    validate_paths()

    predictor = ARPMPredictor()
    predictor.load(ckt_path=args.eval_model_path, use_data_parallel=True)

    dump_entity_embeddings(predictor)

    # keep the entity matrix on the second GPU (if any): GPU 0 hosts the DataParallel master / gathers outputs
    ent_device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
    entity_tensor = load_entity_embeddings(ent_device)

    coefs_path = os.path.join(os.path.dirname(args.eval_model_path), BEST_COEFS_NAME)
    resolve_coefs(predictor, entity_tensor, coefs_path, batch_size=EVAL_BATCH_SIZE)

    if args.tune_only:
        logger.info(f'--tune-only: stopping after coefficient tuning ({current_coefs()})')
        return

    result = evaluate_predictor(predictor, entity_tensor=entity_tensor,
                                batch_size=EVAL_BATCH_SIZE, save_details=True)
    forward_metrics, backward_metrics, averaged_metrics = (
        result['forward'], result['backward'], result['average']
    )
    logger.info(f'Coefficients used (c_h, c_p, c_s): {current_coefs()}  '
                f'Averaged metrics: {averaged_metrics}')

    save_metrics(forward_metrics, backward_metrics, averaged_metrics)


if __name__ == '__main__':
    predict_by_split()