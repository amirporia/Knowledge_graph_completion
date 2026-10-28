import json
import os
import random
from dataclasses import dataclass, asdict
from time import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import tqdm

from .predict import BertPredictor
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict, get_all_triplet_dict
from ..utils.doc import load_data, Example
from ..utils.triplet import EntityDict
from ..utils.utils import rerank_by_graph

FILTER_FILL = -1.0          # same fill value as the original implementation
ENTITY_CHUNK = 524288       # entities scored at a time (4.8M x batch fp32 does not fit on a 16 GB GPU)
TIE_EPS = 2e-6              # scores within this tolerance of the gold count as ties (see ARPM evaluate.py)


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

# tail_id -> {head_id}, built lazily and only for the heads of the evaluated queries. The original
# `_collect_mask_indices` scanned ALL (h, r) keys (28M on Wikidata5M) for EVERY query.
_rev_index: Dict[str, set] = {}
_rev_done: set = set()


def _iter_hr2tails():
    d = all_triplet_dict
    parts = [d.base, d.extra] if hasattr(d, 'base') else [d]
    for part in parts:
        yield from part.hr2tails.items()


def _ensure_rev_index(head_ids) -> None:
    missing = set(head_ids) - _rev_done
    if not missing:
        return
    logger.info(f'Building reverse index for {len(missing)} query heads...')
    for (head_id, _), tail_ids in _iter_hr2tails():
        if not missing.isdisjoint(tail_ids):
            for t in missing.intersection(tail_ids):
                _rev_index.setdefault(t, set()).add(head_id)
    _rev_done.update(missing)


def _filter_indices(example: Example) -> np.ndarray:
    """Entities to mask for this query (same semantics as the original): the other known tails of
    (h, r), plus every entity that has the head as a tail of some triple (excluding the gold)."""
    idx = {entity_dict.entity_to_idx(e)
           for e in all_triplet_dict.get_neighbors(example.head_id, example.relation)
           if e != example.tail_id}
    if getattr(args, 'filter_head_neighbors', True):      # --no-filter-head-neighbors = standard protocol
        for h in _rev_index.get(example.head_id, ()):
            if h != example.tail_id:
                idx.add(entity_dict.entity_to_idx(h))
    return np.fromiter(idx, dtype=np.int64, count=len(idx))


@torch.no_grad()
def compute_metrics(
        related_hr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor,
        entities_tensor: torch.Tensor,
        target: List[int],
        examples: List[Example],
        top_k: int = 20,
        batch_size: int = 256,
        entity_chunk: int = ENTITY_CHUNK,
) -> Tuple[List, List, Dict, List]:
    """
    Link-prediction metrics with the RAA-KGC inference rule (paper Eq. 9):

        score(t) = cos(e^avg_hrta, e_t) + cos(e_hr, e_t)   (+ optional graph re-rank bonus)

    The entity axis is processed in chunks (cast to fp32 on the fly, so `entities_tensor` may be fp16).
    rank = 1 + #{entities with a strictly higher score}; the gold is excluded from the count explicitly.
    """
    assert hr_tensor.size(1) == entities_tensor.size(1), "Embedding dimensions must match"
    assert hr_tensor.size(0) == related_hr_tensor.size(0), "Tensor sizes must match"

    total = hr_tensor.size(0)
    n_ent = entities_tensor.size(0)
    assert n_ent == len(entity_dict), "Entity count mismatch"

    device = entities_tensor.device
    hr_tensor = hr_tensor.to(device)
    related_hr_tensor = related_hr_tensor.to(device)
    target_t = torch.LongTensor(target).to(device)
    use_bonus = args.neighbor_weight >= 1e-6

    topk_scores, topk_indices, ranks = [], [], []

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        nb = end - start
        rows = torch.arange(nb, device=device)
        tgt = target_t[start:end]
        hr = hr_tensor[start:end]
        rel = related_hr_tensor[start:end]

        bonus = None
        if use_bonus:
            bonus = torch.zeros(nb, n_ent, device=device)
            rerank_by_graph(bonus, examples[start:end], entity_dict=entity_dict)

        fmask = torch.zeros(nb, n_ent, dtype=torch.bool, device=device)
        for i in range(nb):
            idxs = _filter_indices(examples[start + i])
            if len(idxs):
                fmask[i, torch.as_tensor(idxs, device=device)] = True

        gold = entities_tensor[tgt].float()
        gold_score = (hr * gold).sum(1) + (rel * gold).sum(1)
        if use_bonus:
            gold_score = gold_score + bonus[rows, tgt]
        gold_score = (gold_score + TIE_EPS).unsqueeze(1)

        counts = torch.zeros(nb, dtype=torch.long, device=device)
        best_v = best_i = None
        for s in range(0, n_ent, entity_chunk):
            e = min(s + entity_chunk, n_ent)
            ent_t = entities_tensor[s:e].float().t()

            score = hr.mm(ent_t) + rel.mm(ent_t)
            if bonus is not None:
                score += bonus[:, s:e]
            score.masked_fill_(fmask[:, s:e], FILTER_FILL)

            if top_k > 0:  # top-k BEFORE the gold is removed from the count
                v, ix = score.topk(min(top_k, score.size(1)), dim=-1)
                ix = ix + s
                if best_v is None:
                    best_v, best_i = v, ix
                else:
                    v = torch.cat([best_v, v], dim=1)
                    ix = torch.cat([best_i, ix], dim=1)
                    best_v, sel = v.topk(min(top_k, v.size(1)), dim=-1)
                    best_i = ix.gather(1, sel)

            in_chunk = (tgt >= s) & (tgt < e)
            if in_chunk.any():
                score[rows[in_chunk], tgt[in_chunk] - s] = float('-inf')
            counts += (score > gold_score).sum(dim=1)

        ranks.extend((counts + 1).tolist())
        if top_k > 0:
            topk_scores.extend(best_v.tolist())
            topk_indices.extend(best_i.tolist())

    r = torch.tensor(ranks, dtype=torch.float)
    metrics = {
        'mean_rank': round(r.mean().item(), 4),
        'mrr': round((1.0 / r).mean().item(), 4),
        'hit@1': round((r <= 1).float().mean().item(), 4),
        'hit@3': round((r <= 3).float().mean().item(), 4),
        'hit@10': round((r <= 10).float().mean().item(), 4),
        'hit@50': round((r <= 50).float().mean().item(), 4),
    }
    return topk_scores, topk_indices, metrics, ranks


def eval_single_direction(
        predictor: BertPredictor,
        entity_tensor: torch.Tensor,
        eval_forward: bool = True,
        batch_size: int = 64,
        save_details: bool = True,
        max_queries: Optional[int] = None,
) -> Dict:
    """Evaluate model performance in a single direction (head->tail or tail->head).
    `max_queries` (default --max-eval-queries) evaluates a fixed random subset when > 0."""
    start_time = time()

    examples = load_data(
        args.valid_path,
        add_forward_triplet=eval_forward,
        add_backward_triplet=not eval_forward,
    )
    limit = getattr(args, 'max_eval_queries', 0) if max_queries is None else max_queries
    if limit and 0 < limit < len(examples):
        examples = random.Random(0).sample(examples, limit)
        logger.info(f'Evaluating on a random subset of {limit} queries')

    if getattr(args, 'filter_head_neighbors', True):
        _ensure_rev_index({ex.head_id for ex in examples})

    hr_tensor, _, related_hr_tensor = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)

    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]

    logger.info('Predict tensor done, computing metrics...')

    topk_scores, topk_indices, metrics, ranks = compute_metrics(
        related_hr_tensor=related_hr_tensor,
        hr_tensor=hr_tensor,
        entities_tensor=entity_tensor,
        target=target,
        examples=examples,
        top_k=20 if save_details else 0,
        batch_size=batch_size,
    )

    direction = 'forward' if eval_forward else 'backward'
    logger.info(f'{direction} metrics: {json.dumps(metrics)}')

    if save_details:
        _save_prediction_details(
            examples, topk_scores, topk_indices, target, ranks,
            eval_direction=direction,
        )

    logger.info(f'Evaluation took {round(time() - start_time, 3)} seconds')
    return metrics


def evaluate_predictor(
        predictor: BertPredictor,
        entity_tensor: torch.Tensor,
        batch_size: int = 64,
        save_details: bool = True,
        max_queries: Optional[int] = None,
) -> Dict:
    """Forward + backward filtered ranking; shared by the trainer and the eval scripts."""
    fwd = eval_single_direction(predictor, entity_tensor, True, batch_size, save_details, max_queries)
    bwd = eval_single_direction(predictor, entity_tensor, False, batch_size, save_details, max_queries)
    avg = {k: round((fwd[k] + bwd[k]) / 2, 4) for k in fwd}
    return {'forward': fwd, 'backward': bwd, 'average': avg}


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

    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)

    output_path = f'{prefix}/task_hrt_{split}_{eval_direction}_{basename}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(info) for info in pred_infos], f, ensure_ascii=False, indent=4)


def predict_by_split() -> None:
    """Run prediction evaluation on the valid/test split."""
    assert os.path.exists(args.valid_path), f"Valid path not found: {args.valid_path}"
    assert os.path.exists(args.train_path), f"Train path not found: {args.train_path}"

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)

    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    result = evaluate_predictor(predictor, entity_tensor=entity_tensor)
    forward_metrics, backward_metrics, averaged_metrics = (
        result['forward'], result['backward'], result['average'])
    logger.info(f'Averaged metrics: {averaged_metrics}')

    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)

    output_path = f'{prefix}/task_hrt_{split}_{basename}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'forward metrics: {json.dumps(forward_metrics)}\n')
        f.write(f'backward metrics: {json.dumps(backward_metrics)}\n')
        f.write(f'average metrics: {json.dumps(averaged_metrics)}\n')


if __name__ == '__main__':
    predict_by_split()