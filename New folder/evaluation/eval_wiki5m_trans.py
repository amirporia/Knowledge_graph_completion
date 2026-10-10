import json
import os
from typing import Dict

import torch

from .evaluate import evaluate_predictor
from .predict import ARPMPredictor
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict

SHARD_SIZE = 1_000_000
EVAL_BATCH_SIZE = 32
TASK_REQUIREMENT = 'wiki5m_trans'

assert args.task == TASK_REQUIREMENT, (
    f'This script is only used for {TASK_REQUIREMENT} transduction setting'
)

entity_dict = get_entity_dict()


def get_shard_path(shard_id: int = 0) -> str:
    return os.path.join(args.model_dir, f'shard_{shard_id}')


def dump_entity_embeddings(predictor: ARPMPredictor) -> None:
    """Generate and save entity embeddings in shards."""
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
        torch.save(shard_entity_tensor, shard_path)

        logger.info(f'Completed shard_id={shard_id}')


def load_entity_embeddings() -> torch.Tensor:
    """Load and combine all sharded entity embeddings."""
    first_shard_path = get_shard_path()
    assert os.path.exists(first_shard_path), f'No shards found at {first_shard_path}'

    shard_tensors = []
    for start in range(0, len(entity_dict), SHARD_SIZE):
        shard_id = start // SHARD_SIZE
        shard_path = get_shard_path(shard_id)

        shard_entity_tensor = torch.load(
            shard_path,
            map_location=lambda storage, loc: storage
        )

        logger.info(
            f'Loaded {shard_entity_tensor.size(0)} entity embeddings '
            f'from {shard_path}'
        )
        shard_tensors.append(shard_entity_tensor)

    entity_tensor = torch.cat(shard_tensors, dim=0)
    logger.info(f'Total entity embeddings: {entity_tensor.size(0)}')

    assert entity_tensor.size(0) == len(entity_dict.entity_exs), \
        f'Expected {len(entity_dict.entity_exs)} embeddings, got {entity_tensor.size(0)}'

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

    metrics_data = {
        'forward_metrics': forward_metrics,
        'backward_metrics': backward_metrics,
        'average_metrics': averaged_metrics,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f'Metrics saved to {output_path}')


def predict_by_split() -> None:
    args.batch_size = max(args.batch_size, torch.cuda.device_count() * 1024)

    validate_paths()

    predictor = ARPMPredictor()
    predictor.load(ckt_path=args.eval_model_path, use_data_parallel=True)

    dump_entity_embeddings(predictor)
    entity_tensor = load_entity_embeddings().cuda()

    result = evaluate_predictor(predictor, entity_tensor=entity_tensor,
                                 batch_size=EVAL_BATCH_SIZE, save_details=True)
    forward_metrics, backward_metrics, averaged_metrics = (
        result['forward'], result['backward'], result['average']
    )
    logger.info(f'Averaged metrics: {averaged_metrics}')

    save_metrics(forward_metrics, backward_metrics, averaged_metrics)


if __name__ == '__main__':
    predict_by_split()
