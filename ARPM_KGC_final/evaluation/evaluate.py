import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from time import time
from typing import List, Tuple, Dict, Optional

import torch
import tqdm

from .predict import BertPredictor
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict, get_all_triplet_dict
from ..utils.doc import load_data, Example
from ..utils.triplet import EntityDict
from ..utils.utils import rerank_by_graph, get_model_obj

SCALE_GRID = (0.0, 0.25, 0.5, 1.0)
FILTER_FILL = -1e4
BEST_SCALES_NAME = 'best_scales.json'


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
    lambda_p: float = 0.0
    lambda_s: float = 0.0


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

_rev_index: Optional[Dict[str, set]] = None


def _get_rev_index() -> Dict[str, set]:
    """tail_id -> {head_id} (replaces the baseline's O(#(h,r) keys) scan per query; same semantics)."""
    global _rev_index
    if _rev_index is None:
        _rev_index = defaultdict(set)
        for (head_id, _), tail_ids in all_triplet_dict.hr2tails.items():
            for tail_id in tail_ids:
                _rev_index[tail_id].add(head_id)
    return _rev_index


def _filter_indices(example: Example) -> List[int]:
    """Entity indices filtered for this query (baseline semantics): the other known tails of (h, r),
    plus every entity that has the head as a tail of some triple (excluding the gold)."""
    idx = {entity_dict.entity_to_idx(e) for e in all_triplet_dict.get_neighbors(example.head_id, example.relation)
           if e != example.tail_id}
    for h in _get_rev_index().get(example.head_id, ()):
        if h != example.tail_id:
            idx.add(entity_dict.entity_to_idx(h))
    return sorted(idx)


def current_scales() -> Tuple[float, float]:
    return (1.0 if args.scale_p is None else args.scale_p,
            1.0 if args.scale_s is None else args.scale_s)


def _summarise(ranks: torch.Tensor) -> Dict[str, float]:
    r = ranks.float()
    return {
        'mean_rank': round(r.mean().item(), 4),
        'mrr': round((1.0 / r).mean().item(), 4),
        'hit@1': round((r <= 1).float().mean().item(), 4),
        'hit@3': round((r <= 3).float().mean().item(), 4),
        'hit@10': round((r <= 10).float().mean().item(), 4),
        'hit@50': round((r <= 50).float().mean().item(), 4),
    }


@torch.no_grad()
def compute_metrics(
        predictor: BertPredictor,
        memory: Dict[str, torch.Tensor],
        entities_tensor: torch.Tensor,
        target: List[int],
        examples: List[Example],
        scale_pairs: List[Tuple[float, float]],
        top_k: int = 0,
        batch_size: int = 256,
) -> Dict:
    """Filtered ranking.

        baseline (RAA-KGC Eq. 9):  S = cos(e_hr, e_t) + cos(e^avg_hrta, e_t)
        ARPM:                      S = <baseline> + sp*lambda_p*S_p + ss*lambda_s*S_struct

    (+ optional graph re-rank bonus). The baseline part is computed once per batch and re-used for
    every (sp, ss) pair; without the memory branch only (1, 1) is meaningful and the memory terms
    are skipped, which gives exactly the baseline score."""
    model_obj = get_model_obj(predictor.model)
    use_memory = getattr(model_obj, 'use_memory', False)

    total = memory['q'].size(0)
    assert memory['q'].size(1) == entities_tensor.size(1), "Embedding dimensions must match"
    assert entities_tensor.size(0) == len(entity_dict), "Entity count mismatch"

    device = entities_tensor.device
    target_t = torch.LongTensor(target).unsqueeze(-1).to(device)
    filter_cache = [_filter_indices(ex) for ex in examples]

    ranks = [[] for _ in scale_pairs]
    topk_scores, topk_indices = [], []

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = min(start + batch_size, total)

        base = (memory['q'][start:end].mm(entities_tensor.t())
                + memory['q_hrta'][start:end].mm(entities_tensor.t()))

        if use_memory:
            S_p = model_obj.score_prototypes(memory['prototypes'][start:end], entities_tensor)
            S_s = model_obj.score_struct(memory['m_struct'][start:end], entities_tensor)
            lam_p = memory['lambda_p'][start:end].unsqueeze(1)
            lam_s = memory['lambda_s'][start:end].unsqueeze(1)

        # Optional SimKGC graph re-ranking
        bonus = torch.zeros_like(base)
        rerank_by_graph(bonus, examples[start:end], entity_dict=entity_dict)
        base = base + bonus

        fmask = torch.zeros_like(base, dtype=torch.bool)
        for i in range(end - start):
            idxs = filter_cache[start + i]
            if idxs:
                fmask[i, torch.as_tensor(idxs, device=device)] = True

        tgt = target_t[start:end]
        for pi, (sp, ss) in enumerate(scale_pairs):
            score = base
            if use_memory:
                score = base + sp * lam_p * S_p + ss * lam_s * S_s
            score = score.masked_fill(fmask, FILTER_FILL)
            tgt_score = score.gather(1, tgt)
            ranks[pi].append(((score > tgt_score).sum(dim=1) + 1).cpu())
            if top_k > 0 and pi == 0:
                vals, inds = score.topk(top_k, dim=-1)
                topk_scores.extend(vals.tolist())
                topk_indices.extend(inds.tolist())

    all_ranks = [torch.cat(r) for r in ranks]
    return {'metrics': [_summarise(r) for r in all_ranks], 'ranks': all_ranks,
            'topk_scores': topk_scores, 'topk_indices': topk_indices}


def eval_single_direction(
        predictor: BertPredictor,
        entity_tensor: torch.Tensor,
        eval_forward: bool = True,
        batch_size: int = 64,
        save_details: bool = True,
        scale_pairs: Optional[List[Tuple[float, float]]] = None,
) -> List[Dict]:
    """Returns a list of metric dicts, one per (scale_p, scale_s) pair."""
    start_time = time()
    scale_pairs = scale_pairs or [current_scales()]

    examples = load_data(
        args.valid_path,
        add_forward_triplet=eval_forward,
        add_backward_triplet=not eval_forward,
    )

    memory = {k: v.to(entity_tensor.device) for k, v in predictor.predict_by_examples(examples).items()}
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]

    logger.info('Predict tensor done, computing metrics...')
    out = compute_metrics(predictor, memory, entity_tensor, target, examples, scale_pairs,
                          top_k=20 if save_details else 0, batch_size=batch_size)

    direction = 'forward' if eval_forward else 'backward'
    logger.info(f'{direction} metrics (scales={scale_pairs[0]}): {json.dumps(out["metrics"][0])}')

    if save_details:
        zeros = torch.zeros(len(examples))
        _save_prediction_details(examples, out['topk_scores'], out['topk_indices'], target,
                                 out['ranks'][0].tolist(),
                                 memory.get('lambda_p', zeros), memory.get('lambda_s', zeros), direction)

    logger.info(f'Evaluation took {round(time() - start_time, 3)} seconds')
    return out['metrics']


def evaluate_predictor(
        predictor: BertPredictor,
        entity_tensor: torch.Tensor,
        batch_size: int = 256,
        save_details: bool = True,
        scale_pairs: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """Forward + backward filtered ranking. The first pair of `scale_pairs` fills
    'forward'/'backward'/'average'; 'grid' lists every pair."""
    pairs = scale_pairs or [current_scales()]
    fwd = eval_single_direction(predictor, entity_tensor, True, batch_size, save_details, pairs)
    bwd = eval_single_direction(predictor, entity_tensor, False, batch_size, save_details, pairs)
    avg = [{k: round((f[k] + b[k]) / 2, 4) for k in f} for f, b in zip(fwd, bwd)]

    grid = [{'scale_p': p, 'scale_s': s, 'forward': f, 'backward': b, 'average': a}
            for (p, s), f, b, a in zip(pairs, fwd, bwd, avg)]
    return {'forward': fwd[0], 'backward': bwd[0], 'average': avg[0], 'grid': grid}


def _save_prediction_details(examples, topk_scores, topk_indices, target, ranks,
                             lambda_p_tensor, lambda_s_tensor, eval_direction: str) -> None:
    pred_infos = []
    for idx, example in enumerate(examples):
        scores, indices = topk_scores[idx], topk_indices[idx]
        score_info = {entity_dict.get_entity_by_idx(i).entity: round(s, 3) for s, i in zip(scores, indices)}
        pred_infos.append(PredInfo(
            head=example.head, relation=example.relation, tail=example.tail,
            pred_tail=entity_dict.get_entity_by_idx(indices[0]).entity,
            pred_score=round(scores[0], 4),
            topk_score_info=json.dumps(score_info),
            rank=ranks[idx],
            correct=indices[0] == target[idx],
            lambda_p=round(float(lambda_p_tensor[idx]), 4),
            lambda_s=round(float(lambda_s_tensor[idx]), 4),
        ))

    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    output_path = f'{prefix}/task_hrt_{split}_{eval_direction}_{basename}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(info) for info in pred_infos], f, ensure_ascii=False, indent=4)


def _resolve_scales(predictor, entity_tensor, scales_path: str) -> None:
    """tune on --valid-path (--tune-scales) | explicit --scale-p/--scale-s | best_scales.json | 1.0/1.0"""
    if args.tune_scales:
        pairs = [(p, s) for p in SCALE_GRID for s in SCALE_GRID]
        res = evaluate_predictor(predictor, entity_tensor, save_details=False, scale_pairs=pairs)
        for g in res['grid']:
            logger.info(f"scale_p={g['scale_p']:.2f} scale_s={g['scale_s']:.2f} -> "
                        f"MRR {g['average']['mrr']:.4f}  H@1 {g['average']['hit@1']:.4f}  "
                        f"H@10 {g['average']['hit@10']:.4f}")
        best = max(res['grid'], key=lambda g: g['average']['mrr'])
        args.scale_p, args.scale_s = best['scale_p'], best['scale_s']
        with open(scales_path, 'w') as f:
            json.dump({'scale_p': best['scale_p'], 'scale_s': best['scale_s'],
                       'tuned_on': args.valid_path, 'average': best['average']}, f, indent=2)
        logger.info(f'Best scales ({best["scale_p"]}, {best["scale_s"]}) saved to {scales_path}')
    elif args.scale_p is None and args.scale_s is None and os.path.exists(scales_path):
        with open(scales_path) as f:
            saved = json.load(f)
        args.scale_p, args.scale_s = saved['scale_p'], saved['scale_s']
        logger.info(f'Loaded scales ({args.scale_p}, {args.scale_s}) from {scales_path} '
                    f'(tuned on {saved.get("tuned_on")})')


def predict_by_split() -> None:
    """Run prediction evaluation on the valid/test split."""
    assert os.path.exists(args.valid_path), f"Valid path not found: {args.valid_path}"
    assert os.path.exists(args.train_path), f"Train path not found: {args.train_path}"

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)

    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    if predictor.use_memory:
        scales_path = os.path.join(os.path.dirname(args.eval_model_path), BEST_SCALES_NAME)
        _resolve_scales(predictor, entity_tensor, scales_path)

    result = evaluate_predictor(predictor, entity_tensor=entity_tensor, save_details=True)
    forward_metrics, backward_metrics, averaged_metrics = (
        result['forward'], result['backward'], result['average'])
    logger.info(f'Scales used: {current_scales()}  Averaged metrics: {averaged_metrics}')

    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open(f'{prefix}/task_hrt_{split}_{basename}.json', 'w', encoding='utf-8') as f:
        f.write(f'memory enabled: {predictor.use_memory}\n')
        if predictor.use_memory:
            f.write(f'scales (scale_p, scale_s): {current_scales()}\n')
        f.write(f'forward metrics: {json.dumps(forward_metrics)}\n')
        f.write(f'backward metrics: {json.dumps(backward_metrics)}\n')
        f.write(f'average metrics: {json.dumps(averaged_metrics)}\n')


if __name__ == '__main__':
    predict_by_split()
