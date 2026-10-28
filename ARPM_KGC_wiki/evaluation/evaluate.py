import itertools
import json
import os
import random
from dataclasses import dataclass, asdict
from time import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import tqdm

from .predict import ARPMPredictor
from ..setting.config import args, get_coefs
from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict, get_all_triplet_dict
from ..utils.doc import load_data, Example
from ..utils.triplet import EntityDict
from ..utils.utils import rerank_by_graph

FILTER_FILL = -1e4
BEST_COEFS_NAME = 'best_coefs.json'

# Number of entities scored at a time. Wikidata5M has 4.8M entities x 768 dims: scoring against the
# whole matrix at once needs several (batch x 4.8M) fp32 tensors, which does not fit on a 16 GB GPU.
# The ranks / top-k are accumulated chunk by chunk instead (identical result, bounded memory).
ENTITY_CHUNK = 524288

# The gold score is computed by a small separate matmul, the other scores by the chunk matmul; for entities
# with an IDENTICAL embedding (same text) the two can differ by ~1e-7 and would be counted as "higher".
# Scores within this tolerance of the gold are treated as ties (= not higher), as in the exact-equality
# behaviour of the single-matmul version.
TIE_EPS = 2e-6

Coefs = Tuple[float, float, float]  # (c_h, c_p, c_s)


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
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()

# tail_id -> {head_id}, built lazily and ONLY for the entities that are heads of the evaluated queries
# (the old version indexed all 4.8M entities = several GB, and copied the whole hr2tails dict first).
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
    """Entity indices to filter for this query (same semantics as the baseline): the other known
    tails of (h, r), plus every entity that has the head as a tail of some triple (excluding the gold)."""
    idx = {entity_dict.entity_to_idx(e) for e in all_triplet_dict.get_neighbors(example.head_id, example.relation)
           if e != example.tail_id}
    if getattr(args, 'filter_head_neighbors', True):      # --no-filter-head-neighbors = standard protocol
        for h in _rev_index.get(example.head_id, ()):
            if h != example.tail_id:
                idx.add(entity_dict.entity_to_idx(h))
    return np.fromiter(idx, dtype=np.int64, count=len(idx))


def current_coefs() -> Coefs:
    """(c_h, c_p, c_s) currently in `args` (unset values fall back to DEFAULT_COEFS)."""
    return get_coefs(args)


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
        predictor: ARPMPredictor,
        memory: Dict[str, torch.Tensor],
        entities_tensor: torch.Tensor,
        target: List[int],
        examples: List[Example],
        coef_triples: List[Coefs],
        top_k: int = 0,
        batch_size: int = 256,
        entity_chunk: int = ENTITY_CHUNK,
) -> Dict:
    """Filtered ranking with S = S_q + c_h*S_hrta + c_p*S_p + c_s*S_struct (+ graph re-rank bonus).

    The entity axis is processed in chunks of `entity_chunk` rows (cast to fp32 on the fly, so
    `entities_tensor` may be stored in fp16). For every chunk all score components are computed once and
    re-used for every (c_h, c_p, c_s) triple. rank = 1 + #{entities with a strictly higher score}; the gold
    entity is excluded from the count explicitly, so tiny kernel-level differences between its separately
    computed score and its in-chunk score cannot add 1 to the rank."""
    model_obj = predictor.model.module if hasattr(predictor.model, 'module') else predictor.model
    use_raa = getattr(model_obj, 'use_raa', False)

    total = memory['q'].size(0)
    n_ent = entities_tensor.size(0)
    assert memory['q'].size(1) == entities_tensor.size(1), "Embedding dimensions must match"
    assert n_ent == len(entity_dict), "Entity count mismatch"

    device = entities_tensor.device
    target_t = torch.LongTensor(target).to(device)
    use_bonus = args.neighbor_weight >= 1e-6

    ranks = [[] for _ in coef_triples]
    topk_scores, topk_indices = [], []

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        nb = end - start
        rows = torch.arange(nb, device=device)
        tgt = target_t[start:end]

        q = memory['q'][start:end]
        q_h = memory['q_hrta'][start:end] if use_raa else None
        protos = memory['prototypes'][start:end]
        m_s = memory['m_struct'][start:end]

        bonus = None
        if use_bonus:
            bonus = torch.zeros(nb, n_ent, device=device)
            rerank_by_graph(bonus, examples[start:end], entity_dict=entity_dict)

        fmask = torch.zeros(nb, n_ent, dtype=torch.bool, device=device)
        for i in range(nb):
            idxs = _filter_indices(examples[start + i])
            if len(idxs):
                fmask[i, torch.as_tensor(idxs, device=device)] = True

        # ---- gold scores (the gold entity is never filtered) ----
        gold = entities_tensor[tgt].float()
        g_fixed = model_obj.score_query(q, gold).diagonal().clone()
        if use_bonus:
            g_fixed = g_fixed + bonus[rows, tgt]
        g_h = model_obj.score_query(q_h, gold).diagonal() if use_raa else None
        g_p = model_obj.score_prototypes(protos, gold).diagonal()
        g_s = model_obj.score_struct(m_s, gold).diagonal()

        tgt_scores = []
        for ch, cp, cs in coef_triples:
            sc = g_fixed.clone()
            if use_raa and ch != 0.0:
                sc += ch * g_h
            if cp != 0.0:
                sc += cp * g_p
            if cs != 0.0:
                sc += cs * g_s
            tgt_scores.append((sc + TIE_EPS).unsqueeze(1))

        # ---- all entities, chunk by chunk ----
        counts = torch.zeros(len(coef_triples), nb, dtype=torch.long, device=device)
        best_v = best_i = None
        for s in range(0, n_ent, entity_chunk):
            e = min(s + entity_chunk, n_ent)
            ent = entities_tensor[s:e].float()

            S_q = model_obj.score_query(q, ent)
            S_h = model_obj.score_query(q_h, ent) if use_raa else None
            S_p = model_obj.score_prototypes(protos, ent)
            S_s = model_obj.score_struct(m_s, ent)
            fixed = S_q if bonus is None else S_q + bonus[:, s:e]
            fm = fmask[:, s:e]

            in_chunk = (tgt >= s) & (tgt < e)
            c_rows, c_cols = rows[in_chunk], tgt[in_chunk] - s

            for pi, (ch, cp, cs) in enumerate(coef_triples):
                score = fixed.clone()
                if use_raa and ch != 0.0:
                    score += ch * S_h
                if cp != 0.0:
                    score += cp * S_p
                if cs != 0.0:
                    score += cs * S_s
                score.masked_fill_(fm, FILTER_FILL)

                if top_k > 0 and pi == 0:  # top-k is taken BEFORE the gold is removed from the count
                    v, ix = score.topk(min(top_k, score.size(1)), dim=-1)
                    ix = ix + s
                    if best_v is None:
                        best_v, best_i = v, ix
                    else:
                        v = torch.cat([best_v, v], dim=1)
                        ix = torch.cat([best_i, ix], dim=1)
                        best_v, sel = v.topk(min(top_k, v.size(1)), dim=-1)
                        best_i = ix.gather(1, sel)

                if c_rows.numel():
                    score[c_rows, c_cols] = float('-inf')
                counts[pi] += (score > tgt_scores[pi]).sum(dim=1)

        for pi in range(len(coef_triples)):
            ranks[pi].append((counts[pi] + 1).cpu())
        if top_k > 0:
            topk_scores.extend(best_v.tolist())
            topk_indices.extend(best_i.tolist())

    all_ranks = [torch.cat(r) for r in ranks]
    return {'metrics': [_summarise(r) for r in all_ranks], 'ranks': all_ranks,
            'topk_scores': topk_scores, 'topk_indices': topk_indices}


def eval_single_direction(
        predictor: ARPMPredictor,
        entity_tensor: torch.Tensor,
        eval_forward: bool = True,
        batch_size: int = 64,
        save_details: bool = True,
        coef_triples: Optional[List[Coefs]] = None,
        max_queries: Optional[int] = None,
) -> List[Dict]:
    """Returns a list of metric dicts, one per (c_h, c_p, c_s) triple.
    `max_queries` (default: --max-eval-queries) evaluates a fixed random subset (seeded) when > 0."""
    start_time = time()
    coef_triples = coef_triples or [current_coefs()]

    examples = load_data(args.valid_path, add_forward_triplet=eval_forward,
                         add_backward_triplet=not eval_forward)
    limit = getattr(args, 'max_eval_queries', 0) if max_queries is None else max_queries
    if limit and 0 < limit < len(examples):
        examples = random.Random(0).sample(examples, limit)
        logger.info(f'Evaluating on a random subset of {limit} queries')

    if getattr(args, 'filter_head_neighbors', True):
        _ensure_rev_index({ex.head_id for ex in examples})

    memory = {k: v.to(entity_tensor.device) for k, v in predictor.predict_by_examples(examples).items()}
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]

    logger.info('Predict tensor done, computing metrics...')
    out = compute_metrics(predictor, memory, entity_tensor, target, examples, coef_triples,
                          top_k=20 if save_details else 0, batch_size=batch_size)

    direction = 'forward' if eval_forward else 'backward'
    logger.info(f'{direction} metrics (coefs={coef_triples[0]}): {json.dumps(out["metrics"][0])}')

    if save_details:
        _save_prediction_details(examples, out['topk_scores'], out['topk_indices'], target,
                                 out['ranks'][0].tolist(), direction)

    logger.info(f'Evaluation took {round(time() - start_time, 3)} seconds')
    return out['metrics']


def evaluate_predictor(
        predictor: ARPMPredictor,
        entity_tensor: torch.Tensor,
        batch_size: int = 256,
        save_details: bool = True,
        coef_triples: Optional[List[Coefs]] = None,
        max_queries: Optional[int] = None,
) -> Dict:
    """Forward + backward filtered ranking. The first triple of `coef_triples` (default: current
    coefficients) fills 'forward'/'backward'/'average'; 'grid' lists every triple."""
    triples = coef_triples or [current_coefs()]
    fwd = eval_single_direction(predictor, entity_tensor, True, batch_size, save_details, triples, max_queries)
    bwd = eval_single_direction(predictor, entity_tensor, False, batch_size, save_details, triples, max_queries)
    avg = [{k: round((f[k] + b[k]) / 2, 4) for k in f} for f, b in zip(fwd, bwd)]

    grid = [{'coef_h': h, 'coef_p': p, 'coef_s': s, 'forward': f, 'backward': b, 'average': a}
            for (h, p, s), f, b, a in zip(triples, fwd, bwd, avg)]
    return {'forward': fwd[0], 'backward': bwd[0], 'average': avg[0], 'grid': grid}


def _save_prediction_details(examples, topk_scores, topk_indices, target, ranks,
                             eval_direction: str) -> None:
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
        ))

    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    output_path = f'{prefix}/task_hrt_{split}_{eval_direction}_{basename}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(info) for info in pred_infos], f, ensure_ascii=False, indent=4)


def resolve_coefs(predictor, entity_tensor, coefs_path: str, batch_size: int = 256) -> None:
    """Decide (c_h, c_p, c_s), in this priority order, and write them into `args`:
         1. --tune-coefs : grid search on --valid-path by average MRR (on a random subset of
                           --tune-queries queries per direction if > 0), saved to best_coefs.json
         2. explicit --coef-h / --coef-p / --coef-s (unset ones fall back to DEFAULT_COEFS)
         3. best_coefs.json next to the checkpoint (if no coefficient was given)
         4. DEFAULT_COEFS
    Tune on the VALIDATION split only; evaluate the test split afterwards without --tune-coefs."""
    model_obj = predictor.model.module if hasattr(predictor.model, 'module') else predictor.model
    use_raa = getattr(model_obj, 'use_raa', False)

    if args.tune_coefs:
        if 'test' in os.path.basename(args.valid_path).lower():
            raise ValueError(f'--tune-coefs must run on the validation split, got {args.valid_path} '
                             '(tuning on test would leak labels into the reported result).')
        h_grid = args.coef_h_grid if use_raa else [0.0]  # c_h is meaningless without the RAA path
        grid = list(itertools.product(h_grid, args.coef_p_grid, args.coef_s_grid))
        tune_q = getattr(args, 'tune_queries', 0)
        logger.info(f'Tuning (c_h, c_p, c_s) over {len(grid)} combinations on {args.valid_path}'
                    f'{f" ({tune_q} queries per direction)" if tune_q else ""}')
        res = evaluate_predictor(predictor, entity_tensor, batch_size=batch_size,
                                 save_details=False, coef_triples=grid, max_queries=tune_q)
        for g in res['grid']:
            logger.info(f"c_h={g['coef_h']:.2f} c_p={g['coef_p']:.2f} c_s={g['coef_s']:.2f} -> "
                        f"MRR {g['average']['mrr']:.4f}  H@1 {g['average']['hit@1']:.4f}  "
                        f"H@10 {g['average']['hit@10']:.4f}")
        best = max(res['grid'], key=lambda g: g['average']['mrr'])
        args.coef_h, args.coef_p, args.coef_s = best['coef_h'], best['coef_p'], best['coef_s']
        for name, val, vals in (('c_h', best['coef_h'], h_grid), ('c_p', best['coef_p'], args.coef_p_grid),
                                ('c_s', best['coef_s'], args.coef_s_grid)):
            if len(set(vals)) > 1 and val == max(vals):
                logger.warning(f'Best {name}={val} is at the upper edge of its grid; consider extending it.')
        with open(coefs_path, 'w') as f:
            json.dump({'coef_h': best['coef_h'], 'coef_p': best['coef_p'], 'coef_s': best['coef_s'],
                       'tuned_on': args.valid_path, 'tune_queries_per_direction': tune_q,
                       'average': best['average']}, f, indent=2)
        logger.info(f'Best coefficients {current_coefs()} saved to {coefs_path}')
    elif args.coef_h is None and args.coef_p is None and args.coef_s is None and os.path.exists(coefs_path):
        with open(coefs_path) as f:
            saved = json.load(f)
        args.coef_h, args.coef_p, args.coef_s = saved['coef_h'], saved['coef_p'], saved['coef_s']
        logger.info(f'Loaded coefficients {current_coefs()} from {coefs_path} '
                    f'(tuned on {saved.get("tuned_on")})')
        if saved.get('tuned_on') == args.valid_path:
            logger.info('Note: evaluating on the same split the coefficients were tuned on.')


def predict_by_split() -> None:
    assert os.path.exists(args.valid_path), f"Valid path not found: {args.valid_path}"
    assert os.path.exists(args.train_path), f"Train path not found: {args.train_path}"

    predictor = ARPMPredictor()
    predictor.load(ckt_path=args.eval_model_path)
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    coefs_path = os.path.join(os.path.dirname(args.eval_model_path), BEST_COEFS_NAME)
    resolve_coefs(predictor, entity_tensor, coefs_path)

    result = evaluate_predictor(predictor, entity_tensor=entity_tensor, save_details=True)
    forward_metrics, backward_metrics, averaged_metrics = (
        result['forward'], result['backward'], result['average'])
    logger.info(f'Coefficients used (c_h, c_p, c_s): {current_coefs()}  Averaged metrics: {averaged_metrics}')

    prefix = os.path.dirname(args.eval_model_path)
    basename = os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open(f'{prefix}/task_hrt_{split}_{basename}.json', 'w', encoding='utf-8') as f:
        f.write(f'coefficients (c_h, c_p, c_s): {current_coefs()}\n')
        f.write(f'forward metrics: {json.dumps(forward_metrics)}\n')
        f.write(f'backward metrics: {json.dumps(backward_metrics)}\n')
        f.write(f'average metrics: {json.dumps(averaged_metrics)}\n')


if __name__ == '__main__':
    predict_by_split()