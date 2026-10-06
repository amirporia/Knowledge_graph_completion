import json
import os
from dataclasses import dataclass, asdict
from time import time
from typing import List, Tuple, Dict

import torch
import tqdm

from .predict import BertPredictor
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict, get_all_triplet_dict
from ..utils.doc import load_data, Example
from ..utils.triplet import EntityDict
from ..utils.utils import rerank_by_graph


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
# Setup
# ---------------------------------------------------------------------------

def _setup_entity_dict() -> EntityDict:
    """Initialize entity dictionary based on task configuration."""
    if args.task == 'wiki5m_ind':
        return EntityDict(
            entity_dict_dir=os.path.dirname(args.valid_path),
            inductive_test_path=args.valid_path,
        )
    return get_entity_dict()


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _filter_known_triplets(
        batch_score: torch.Tensor,
        examples: List[Example],
        start_idx: int,
        mask_value: float,
) -> None:
    """Mask scores of every known true tail (except the gold one) so ranking is "filtered"."""
    for idx in range(batch_score.size(0)):
        cur_ex = examples[start_idx + idx]
        gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)

        if len(gold_neighbor_ids) > 10000:
            logger.debug(
                f'{cur_ex.head_id} - {cur_ex.relation} has {len(gold_neighbor_ids)} neighbors'
            )

        mask_indices = [
            entity_dict.entity_to_idx(e_id)
            for e_id in gold_neighbor_ids
            if e_id != cur_ex.tail_id
        ]
        mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
        batch_score[idx].index_fill_(0, mask_indices, mask_value)


# ---------------------------------------------------------------------------
# Core Computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
        hr_tensor: torch.Tensor,
        entities_tensor: torch.Tensor,
        target: List[int],
        examples: List[Example],
        top_k: int = 3,
        batch_size: int = 256,
        mask_value: float = -1.0,
) -> Tuple[List, List, Dict, List]:
    """Compute link prediction metrics (MR, MRR, Hit@1/3/10/50).

    `mask_value` replaces the scores of known true tails; it must be lower than any real score
    (-1 for cosine scores, a very negative number for unbounded dot-product scores).

    Returns:
        Tuple of (topk_scores, topk_indices, metrics, ranks)
    """
    assert hr_tensor.size(1) == entities_tensor.size(1), "Embedding dimensions must match"
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0), "Entity count mismatch"

    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)

    topk_scores, topk_indices, ranks = [], [], []
    metrics_accumulator = {'mean_rank': 0, 'mrr': 0, 'hit@1': 0,
                           'hit@3': 0, 'hit@10': 0, 'hit@50': 0}

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size

        # batch_size x entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

        # filter known triplets
        _filter_known_triplets(batch_score, examples, start, mask_value)

        batch_sorted_score, batch_sorted_indices = torch.sort(
            batch_score, dim=-1, descending=True,
        )
        target_rank = torch.nonzero(
            batch_sorted_indices.eq(batch_target).long(), as_tuple=False,
        )
        assert target_rank.size(0) == batch_score.size(0), "Rank size mismatch"

        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx, "Index mismatch in ranks"

            current_rank = idx_rank[1] + 1  # 0-based -> 1-based

            metrics_accumulator['mean_rank'] += current_rank
            metrics_accumulator['mrr'] += 1.0 / current_rank
            metrics_accumulator['hit@1'] += 1 if current_rank <= 1 else 0
            metrics_accumulator['hit@3'] += 1 if current_rank <= 3 else 0
            metrics_accumulator['hit@10'] += 1 if current_rank <= 10 else 0
            metrics_accumulator['hit@50'] += 1 if current_rank <= 50 else 0

            ranks.append(current_rank)

        topk_scores.extend(batch_sorted_score[:, :top_k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :top_k].tolist())

    metrics = {k: round(v / total, 4) for k, v in metrics_accumulator.items()}
    assert len(topk_scores) == total, "Top-k scores count mismatch"
    return topk_scores, topk_indices, metrics, ranks


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def eval_single_direction(
        predictor: BertPredictor,
        entity_tensor: torch.Tensor,
        eval_forward: bool = True,
        batch_size: int = 256,
) -> Dict:
    """Evaluate in one direction (forward: head->tail, backward: tail->head via inverse relations)."""
    start_time = time()

    examples = load_data(
        args.valid_path,
        add_forward_triplet=eval_forward,
        add_backward_triplet=not eval_forward,
    )

    hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('Predict tensor done, computing metrics...')

    topk_scores, topk_indices, metrics, ranks = compute_metrics(
        hr_tensor=hr_tensor,
        entities_tensor=entity_tensor,
        target=target,
        examples=examples,
        batch_size=batch_size,
        mask_value=predictor.score_floor,
    )

    direction = 'forward' if eval_forward else 'backward'
    logger.info(f'{direction} metrics: {json.dumps(metrics)}')

    _save_prediction_details(
        examples, topk_scores, topk_indices, target, ranks,
        eval_direction=direction,
    )

    logger.info(f'Evaluation takes {round(time() - start_time, 3)} seconds')
    return metrics


def _save_prediction_details(
        examples: List[Example],
        topk_scores: List,
        topk_indices: List,
        target: List[int],
        ranks: List[int],
        eval_direction: str,
) -> None:
    """Save detailed per-example predictions to a JSON file."""
    pred_infos = []

    for idx, example in enumerate(examples):
        current_scores = topk_scores[idx]
        current_indices = topk_indices[idx]
        predicted_idx = current_indices[0]

        score_info = {
            entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
            for topk_score, topk_idx in zip(current_scores, current_indices)
        }

        pred_infos.append(PredInfo(
            head=example.head,
            relation=example.relation,
            tail=example.tail,
            pred_tail=entity_dict.get_entity_by_idx(predicted_idx).entity,
            pred_score=round(current_scores[0], 4),
            topk_score_info=json.dumps(score_info),
            rank=ranks[idx],
            correct=predicted_idx == target[idx],
        ))

    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)

    output_path = f'{prefix}/eval_{split}_{eval_direction}_{basename}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(info) for info in pred_infos], f, ensure_ascii=False, indent=4)


def predict_by_split() -> None:
    """Run prediction + evaluation on the configured split (forward and backward)."""
    assert os.path.exists(args.valid_path), f"Valid path not found: {args.valid_path}"
    assert os.path.exists(args.train_path), f"Train path not found: {args.train_path}"

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    forward_metrics = eval_single_direction(
        predictor, entity_tensor=entity_tensor, eval_forward=True,
    )
    backward_metrics = eval_single_direction(
        predictor, entity_tensor=entity_tensor, eval_forward=False,
    )
    averaged_metrics = {
        k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4)
        for k in forward_metrics
    }
    logger.info(f'Averaged metrics: {averaged_metrics}')

    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)

    with open(f'{prefix}/metrics_{split}_{basename}.json', 'w', encoding='utf-8') as f:
        f.write(f'forward metrics: {json.dumps(forward_metrics)}\n')
        f.write(f'backward metrics: {json.dumps(backward_metrics)}\n')
        f.write(f'average metrics: {json.dumps(averaged_metrics)}\n')


if __name__ == '__main__':
    predict_by_split()
