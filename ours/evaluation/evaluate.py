import json
import os
from dataclasses import dataclass, asdict
from time import time
from typing import List, Tuple, Dict

import torch
import tqdm

from predict import BertPredictor
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict, get_all_triplet_dict
from ..utils.doc import load_data, Example
from ..utils.rerank import rerank_by_graph
from ..utils.triplet import EntityDict


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


# ---------------------------------------------------------------------------
# Setup Functions
# ---------------------------------------------------------------------------

def _setup_entity_dict() -> EntityDict:
    """Initialize entity dictionary based on task configuration."""
    if args.task == 'wiki5m_ind':
        return EntityDict(
            entity_dict_dir=os.path.dirname(args.valid_path),
            inductive_test_path=args.valid_path,
        )
    return get_entity_dict()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _collect_mask_indices(
        entity_id: str,
        current_tail_id: str,
        all_triplet_dicts,
        mask_indices: List[int],
) -> List[int]:
    """
    Find all head entities that connect to the given entity_id as tail.

    Args:
        entity_id: Current head entity ID
        current_tail_id: Current tail entity ID to exclude from masking
        all_triplet_dicts: Dictionary containing all triples
        mask_indices: Existing list of indices to mask

    Returns:
        Updated list of mask indices
    """
    for (head_id, _), tail_ids in all_triplet_dicts.hr2tails.items():
        if entity_id in tail_ids and head_id != current_tail_id:
            mask_indices.append(entity_dict.entity_to_idx(head_id))

    return mask_indices


def _filter_known_triplets(
        batch_score: torch.Tensor,
        examples: List[Example],
        start_idx: int,
        entity_dictionary: EntityDict,
        all_triplet_dicts,
) -> None:
    """
    Mask scores for known triplets in the batch.

    Args:
        batch_score: Score tensor to mask
        examples: List of evaluation examples
        start_idx: Starting index in the examples list
        entity_dictionary: Entity dictionary for index mapping
        all_triplet_dicts: Dictionary of all known triplets
    """
    for idx in range(batch_score.size(0)):
        example = examples[start_idx + idx]

        # Get known neighbors for the head-relation pair
        gold_neighbor_ids = all_triplet_dicts.get_neighbors(example.head_id, example.relation)

        if len(gold_neighbor_ids) > 10000:
            logger.debug(
                f'{example.head_id} - {example.relation} has {len(gold_neighbor_ids)} neighbors'
            )

        # Build mask for known entities (excluding the correct answer)
        mask_indices = [
            entity_dictionary.entity_to_idx(e_id)
            for e_id in gold_neighbor_ids
            if e_id != example.tail_id
        ]

        # Also mask entities that have the head as their tail
        mask_indices = _collect_mask_indices(
            example.head_id, example.tail_id, all_triplet_dicts, mask_indices
        )

        if mask_indices:
            mask_tensor = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_tensor, -1)


# ---------------------------------------------------------------------------
# Global Initialization
# ---------------------------------------------------------------------------

entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()


# ---------------------------------------------------------------------------
# Core Computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
        related_hr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor,
        entities_tensor: torch.Tensor,
        target: List[int],
        examples: List[Example],
        top_k: int = 20,
        batch_size: int = 256,
) -> Tuple[List, List, Dict, List]:
    """
    Compute evaluation metrics for link prediction.

    Args:
        related_hr_tensor: Embeddings (head-anchor, relation, tail-anchors)
        hr_tensor: Embeddings (head-anchor, relation)
        entities_tensor: Embeddings of all tail entities
        target: Target entity indices
        examples: Evaluation examples
        top_k: Number of top predictions to save
        batch_size: Batch size for processing

    Returns:
        Tuple of (topk_scores, topk_indices, metrics, ranks)
    """
    assert hr_tensor.size(1) == entities_tensor.size(1), "Embedding dimensions must match"
    assert hr_tensor.size(0) == related_hr_tensor.size(0), "Tensor sizes must match"

    total = hr_tensor.size(0)
    entity_count = entities_tensor.size(0)
    assert entity_count == len(entity_dict), "Entity count mismatch"

    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)

    topk_scores, topk_indices, ranks = [], [], []
    metrics_accumulator = {'mean_rank': 0, 'mrr': 0, 'hit@1': 0,
                           'hit@3': 0, 'hit@10': 0, 'hit@50': 0}

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size

        # Compute scores via matrix multiplication
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        related_batch_score = torch.mm(related_hr_tensor[start:end, :], entities_tensor.t())

        # Re-rank based on topological structure
        rerank_by_graph(
            related_batch_score, batch_score,
            examples[start:end],
            entity_dict=entity_dict,
        )

        # Filter known triplets
        _filter_known_triplets(
            batch_score, examples, start, entity_dict, all_triplet_dict,
        )

        # Rank scores and get target ranks
        batch_sorted_score, batch_sorted_indices = torch.sort(
            batch_score, dim=-1, descending=True,
        )

        batch_target = target[start:end]
        target_rank = torch.nonzero(
            batch_sorted_indices.eq(batch_target).long(), as_tuple=False,
        )
        assert target_rank.size(0) == batch_score.size(0), "Rank size mismatch"

        # Calculate metrics
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx, "Index mismatch in ranks"

            current_rank = idx_rank[1] + 1  # Convert to 1-based ranking

            metrics_accumulator['mean_rank'] += current_rank
            metrics_accumulator['mrr'] += 1.0 / current_rank
            metrics_accumulator['hit@1'] += 1 if current_rank <= 1 else 0
            metrics_accumulator['hit@3'] += 1 if current_rank <= 3 else 0
            metrics_accumulator['hit@10'] += 1 if current_rank <= 10 else 0
            metrics_accumulator['hit@50'] += 1 if current_rank <= 50 else 0

            ranks.append(current_rank)

        # Store top-k predictions
        topk_scores.extend(batch_sorted_score[:, :top_k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :top_k].tolist())

    # Normalize metrics
    metrics = {
        k: round(v / total, 4) for k, v in metrics_accumulator.items()
    }

    assert len(topk_scores) == total, "Top-k scores count mismatch"
    return topk_scores, topk_indices, metrics, ranks


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def eval_single_direction(
        predictor: BertPredictor,
        entity_tensor: torch.Tensor,
        eval_forward: bool = True,
        batch_size: int = 64,
) -> Dict:
    """
    Evaluate model performance in a single direction.

    Args:
        predictor: Trained BERT predictor model
        entity_tensor: Pre-computed entity embeddings
        eval_forward: If True, evaluate forward direction (head->tail)
        batch_size: Batch size for metric computation

    Returns:
        Dictionary of evaluation metrics
    """
    start_time = time()

    # Load evaluation data
    examples = load_data(
        args.valid_path,
        add_forward_triplet=eval_forward,
        add_backward_triplet=not eval_forward,
    )

    # Compute embeddings for head-relation pairs
    hr_tensor, _, related_hr_tensor = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)

    # Get target entities
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]

    logger.info('Predict tensor done, computing metrics...')

    # Compute evaluation metrics
    topk_scores, topk_indices, metrics, ranks = compute_metrics(
        related_hr_tensor=related_hr_tensor,
        hr_tensor=hr_tensor,
        entities_tensor=entity_tensor,
        target=target,
        examples=examples,
        batch_size=batch_size,
    )

    # Log metrics
    direction = 'forward' if eval_forward else 'backward'
    logger.info(f'{direction} metrics: {json.dumps(metrics)}')

    # Save detailed predictions
    _save_prediction_details(
        examples, topk_scores, topk_indices, target, ranks,
        eval_direction=direction,
    )

    logger.info(f'Evaluation took {round(time() - start_time, 3)} seconds')
    return metrics


def _save_prediction_details(
        examples: List[Example],
        topk_scores: List,
        topk_indices: List,
        target: List[int],
        ranks: List[int],
        eval_direction: str,
) -> None:
    """Save detailed prediction information to JSON file."""
    pred_infos = []

    for idx, example in enumerate(examples):
        current_scores = topk_scores[idx]
        current_indices = topk_indices[idx]
        predicted_idx = current_indices[0]

        # Build score info dictionary
        score_info = {
            entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
            for topk_score, topk_idx in zip(current_scores, current_indices)
        }

        pred_info = PredInfo(
            head=example.head,
            relation=example.relation,
            tail=example.tail,
            pred_tail=entity_dict.get_entity_by_idx(predicted_idx).entity,
            pred_score=round(current_scores[0], 4),
            topk_score_info=json.dumps(score_info),
            rank=ranks[idx],
            correct=predicted_idx == target[idx],
        )
        pred_infos.append(pred_info)

    # Save to file
    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)

    output_path = f'{prefix}/task_hrt_{split}_{eval_direction}_{basename}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(info) for info in pred_infos], f, ensure_ascii=False, indent=4)


def predict_by_split() -> None:
    """Run prediction evaluation on train/valid/test splits."""
    assert os.path.exists(args.valid_path), f"Valid path not found: {args.valid_path}"
    assert os.path.exists(args.train_path), f"Train path not found: {args.train_path}"

    # Load pre-trained model
    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)

    # Compute entity embeddings
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    # Evaluate both directions
    forward_metrics = eval_single_direction(
        predictor, entity_tensor=entity_tensor, eval_forward=True,
    )
    backward_metrics = eval_single_direction(
        predictor, entity_tensor=entity_tensor, eval_forward=False,
    )

    # Compute average metrics
    averaged_metrics = {
        k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4)
        for k in forward_metrics
    }
    logger.info(f'Averaged metrics: {averaged_metrics}')

    # Save summary
    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)

    output_path = f'{prefix}/task_hrt_{split}_{basename}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'forward metrics: {json.dumps(forward_metrics)}\n')
        f.write(f'backward metrics: {json.dumps(backward_metrics)}\n')
        f.write(f'average metrics: {json.dumps(averaged_metrics)}\n')


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    predict_by_split()
