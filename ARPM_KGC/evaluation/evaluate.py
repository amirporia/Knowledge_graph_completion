import json
import os
from dataclasses import dataclass, asdict
from time import time
from typing import List, Tuple, Dict

import torch
import tqdm

from ARPM_KGC.evaluation.predict import ARPMPredictor
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict, get_all_triplet_dict
from ..utils.doc import load_data, Example
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
    lambda_p: float
    lambda_s: float


# ---------------------------------------------------------------------------
# Setup Functions
# ---------------------------------------------------------------------------

def _setup_entity_dict() -> EntityDict:
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
    """Find all head entities that connect to the given entity_id as tail."""
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
    """Mask scores for known (filtered) triplets in the batch, in place."""
    for idx in range(batch_score.size(0)):
        example = examples[start_idx + idx]

        gold_neighbor_ids = all_triplet_dicts.get_neighbors(example.head_id, example.relation)

        if len(gold_neighbor_ids) > 10000:
            logger.debug(
                f'{example.head_id} - {example.relation} has {len(gold_neighbor_ids)} neighbors'
            )

        mask_indices = [
            entity_dictionary.entity_to_idx(e_id)
            for e_id in gold_neighbor_ids
            if e_id != example.tail_id
        ]

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
        predictor: ARPMPredictor,
        q_tensor: torch.Tensor,
        prototypes_tensor: torch.Tensor,
        m_struct_tensor: torch.Tensor,
        lambda_p_tensor: torch.Tensor,
        lambda_s_tensor: torch.Tensor,
        entities_tensor: torch.Tensor,
        target: List[int],
        examples: List[Example],
        top_k: int = 20,
        batch_size: int = 256,
        slot_gate_tensor: torch.Tensor = None,
) -> Tuple[List, List, Dict, List]:
    """Compute filtered-ranking evaluation metrics using ARPM-KGC's combined
    score S(t|h,r) = S_q(t) + lambda_p*S_p(t) + lambda_s*S_struct(t) against the
    full entity set. adaptive structural memory, gated by the
    learned lambda_s, IS the graph-aware re-ranking signal now.
    """
    d = q_tensor.size(1)
    assert d == entities_tensor.size(1), "Embedding dimensions must match"

    total = q_tensor.size(0)
    entity_count = entities_tensor.size(0)
    assert entity_count == len(entity_dict), "Entity count mismatch"

    target = torch.LongTensor(target).unsqueeze(-1).to(q_tensor.device)
    model_obj = predictor.model.module if hasattr(predictor.model, 'module') else predictor.model

    topk_scores, topk_indices, ranks = [], [], []
    metrics_accumulator = {'mean_rank': 0, 'mrr': 0, 'hit@1': 0,
                            'hit@3': 0, 'hit@10': 0, 'hit@50': 0}

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size

        S_q = model_obj.score_query(q_tensor[start:end], entities_tensor)
        S_p = model_obj.score_prototypes(
            prototypes_tensor[start:end], entities_tensor,
            slot_gate=slot_gate_tensor[start:end] if slot_gate_tensor is not None else None,
        )
        S_s = model_obj.score_struct(m_struct_tensor[start:end], entities_tensor)

        batch_score = model_obj.combined_score(
            S_q, S_p, S_s, lambda_p_tensor[start:end], lambda_s_tensor[start:end]
        )

        _filter_known_triplets(
            batch_score, examples, start, entity_dict, all_triplet_dict,
        )

        batch_sorted_score, batch_sorted_indices = torch.sort(
            batch_score, dim=-1, descending=True,
        )

        batch_target = target[start:end]
        target_rank = torch.nonzero(
            batch_sorted_indices.eq(batch_target).long(), as_tuple=False,
        )
        assert target_rank.size(0) == batch_score.size(0), "Rank size mismatch"

        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx, "Index mismatch in ranks"

            current_rank = idx_rank[1] + 1

            metrics_accumulator['mean_rank'] += current_rank
            metrics_accumulator['mrr'] += 1.0 / current_rank
            metrics_accumulator['hit@1'] += 1 if current_rank <= 1 else 0
            metrics_accumulator['hit@3'] += 1 if current_rank <= 3 else 0
            metrics_accumulator['hit@10'] += 1 if current_rank <= 10 else 0
            metrics_accumulator['hit@50'] += 1 if current_rank <= 50 else 0

            ranks.append(current_rank)

        topk_scores.extend(batch_sorted_score[:, :top_k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :top_k].tolist())

    metrics = {
        k: round(v / total, 4) for k, v in metrics_accumulator.items()
    }

    assert len(topk_scores) == total, "Top-k scores count mismatch"
    return topk_scores, topk_indices, metrics, ranks


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def eval_single_direction(
        predictor: ARPMPredictor,
        entity_tensor: torch.Tensor,
        eval_forward: bool = True,
        batch_size: int = 64,
        save_details: bool = True,
) -> Dict:
    start_time = time()

    examples = load_data(
        args.valid_path,
        add_forward_triplet=eval_forward,
        add_backward_triplet=not eval_forward,
    )

    memory = predictor.predict_by_examples(examples)
    q_tensor = memory['q'].to(entity_tensor.device)
    prototypes_tensor = memory['prototypes'].to(entity_tensor.device)
    m_struct_tensor = memory['m_struct'].to(entity_tensor.device)
    lambda_p_tensor = memory['lambda_p'].to(entity_tensor.device)
    lambda_s_tensor = memory['lambda_s'].to(entity_tensor.device)
    slot_gate_tensor = memory['slot_gate'].to(entity_tensor.device) if memory['slot_gate'] is not None else None

    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]

    logger.info('Predict tensor done, computing metrics...')

    topk_scores, topk_indices, metrics, ranks = compute_metrics(
        predictor=predictor,
        q_tensor=q_tensor,
        prototypes_tensor=prototypes_tensor,
        m_struct_tensor=m_struct_tensor,
        lambda_p_tensor=lambda_p_tensor,
        lambda_s_tensor=lambda_s_tensor,
        entities_tensor=entity_tensor,
        target=target,
        examples=examples,
        batch_size=batch_size,
        slot_gate_tensor=slot_gate_tensor,
    )

    direction = 'forward' if eval_forward else 'backward'
    logger.info(f'{direction} metrics: {json.dumps(metrics)}')

    if save_details:
        _save_prediction_details(
            examples, topk_scores, topk_indices, target, ranks,
            lambda_p_tensor, lambda_s_tensor,
            eval_direction=direction,
        )

    logger.info(f'Evaluation took {round(time() - start_time, 3)} seconds')
    return metrics


def evaluate_predictor(
        predictor: ARPMPredictor,
        entity_tensor: torch.Tensor,
        batch_size: int = 256,
        save_details: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run the full filtered-ranking protocol (forward + backward, averaged) on
    `args.valid_path` for an already-loaded-or-wrapped predictor.
    """
    forward_metrics = eval_single_direction(
        predictor, entity_tensor=entity_tensor, eval_forward=True,
        batch_size=batch_size, save_details=save_details,
    )
    backward_metrics = eval_single_direction(
        predictor, entity_tensor=entity_tensor, eval_forward=False,
        batch_size=batch_size, save_details=save_details,
    )
    averaged_metrics = {
        k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4)
        for k in forward_metrics
    }
    return {'forward': forward_metrics, 'backward': backward_metrics, 'average': averaged_metrics}


def _save_prediction_details(
        examples: List[Example],
        topk_scores: List,
        topk_indices: List,
        target: List[int],
        ranks: List[int],
        lambda_p_tensor: torch.Tensor,
        lambda_s_tensor: torch.Tensor,
        eval_direction: str,
) -> None:
    """Save detailed predictions, including the per-query memory gates
    lambda_p/lambda_s"""
    pred_infos = []

    for idx, example in enumerate(examples):
        current_scores = topk_scores[idx]
        current_indices = topk_indices[idx]
        predicted_idx = current_indices[0]

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
            lambda_p=round(lambda_p_tensor[idx].item(), 4),
            lambda_s=round(lambda_s_tensor[idx].item(), 4),
        )
        pred_infos.append(pred_info)

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

    predictor = ARPMPredictor()
    predictor.load(ckt_path=args.eval_model_path)

    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    result = evaluate_predictor(predictor, entity_tensor=entity_tensor, save_details=True)
    forward_metrics, backward_metrics, averaged_metrics = (
        result['forward'], result['backward'], result['average']
    )
    logger.info(f'Averaged metrics: {averaged_metrics}')

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
