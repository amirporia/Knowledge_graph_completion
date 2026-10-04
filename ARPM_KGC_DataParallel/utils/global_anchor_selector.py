"""Fixed global anchors by head-weighted facility location.

For each relation r, pick once a ranked list of (head, tail) training pairs that is
(a) representative of the whole relation and (b) diverse, by greedily maximising

    F(S) = sum_j w_j * max_{s in S} sim(j, s),      w_j = count_{h_j}^(-power)

(monotone submodular -> greedy is a (1 - 1/e) approximation). The list is identical
across queries, epochs, runs and evaluation, and is cached next to train.txt.json.

Only the TRAINING set is read, and the embeddings use no neighbour text, so nothing
about any held-out edge can leak into the selection.
"""
import hashlib
import json
import os
import time
import zlib
from collections import Counter
from typing import Dict, List, Tuple

import torch

from .dict_hub import get_entity_dict, init_tokenizer, get_tokenizer
from ..setting.config import args
from ..setting.logger_config import logger

CACHE_VERSION = 1
_WAIT_TIMEOUT_S = 6 * 3600
_POLL_S = 10


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(table_size: int, cap: int, power: float) -> str:
    st = os.stat(args.train_path)
    key = json.dumps({
        'v': CACHE_VERSION, 'N': table_size, 'cap': cap, 'power': power,
        'model': args.pretrained_model, 'pooling': args.pooling,
        'max_tokens': args.max_num_tokens,
        'train_size': st.st_size, 'train_mtime': int(st.st_mtime),
    }, sort_keys=True)
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()[:10]
    return os.path.join(os.path.dirname(args.train_path), f'global_anchors_fl_{digest}.json')


def _load_table(path: str) -> Dict[str, List[Tuple[str, str]]]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {r: [(h, t) for h, t in pairs] for r, pairs in raw.items()}


def _save_table_atomic(table: Dict[str, List[Tuple[str, str]]], path: str) -> None:
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _entity_text(entity_id: str, entity_dict, memo: Dict[str, str]) -> str:
    if entity_id in memo:
        return memo[entity_id]
    ex = entity_dict.id2entity.get(entity_id)
    if ex is None:
        text = ''
    else:
        name = (ex.entity or '').strip()
        desc = (ex.entity_desc or '').strip()
        if name and desc.startswith(name):
            desc = desc[len(name):].strip()
        text = f'{name}: {desc}' if (name and desc) else (name or desc)
    memo[entity_id] = text
    return text


@torch.no_grad()
def _embed_pairs(model, tokenizer, device, pairs, relation, entity_dict, memo, batch_size) -> torch.Tensor:
    from ..model.models import _pool_output  # lazy: avoid import cycles at module load

    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        heads = [_entity_text(h, entity_dict, memo) for h, _ in chunk]
        rest = [f'{relation} [SEP] {_entity_text(t, entity_dict, memo)}' for _, t in chunk]
        enc = tokenizer(
            heads, rest, add_special_tokens=True, max_length=args.max_num_tokens,
            truncation=True, padding=True, return_token_type_ids=True, return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                       token_type_ids=enc['token_type_ids'], return_dict=True).last_hidden_state
        vec = _pool_output(args.pooling, hidden[:, 0, :], enc['attention_mask'], hidden)
        out.append(vec.float().cpu())
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Selection (one relation)
# ---------------------------------------------------------------------------

def _greedy_facility_location(sim: torch.Tensor, w: torch.Tensor, n_pick: int) -> List[int]:
    n = sim.size(0)
    cur = torch.zeros(n, dtype=sim.dtype)
    chosen: List[int] = []
    taken = torch.zeros(n, dtype=torch.bool)
    for _ in range(min(n_pick, n)):
        gain = (w.unsqueeze(1) * (sim - cur.unsqueeze(1)).clamp(min=0)).sum(dim=0)
        gain = gain.masked_fill(taken, float('-inf'))
        best = gain.max()
        c = int((gain == best).nonzero()[0])  # ties -> smallest index (= smallest sorted (h, t))
        chosen.append(c)
        taken[c] = True
        cur = torch.maximum(cur, sim[:, c])
    return chosen


def _select_for_relation(relation, pairs, table_size, cap, power, model, tokenizer, device,
                         entity_dict, memo, batch_size) -> List[Tuple[str, str]]:
    # Step 1: head weights from the FULL relation.
    counts = Counter(h for h, _ in pairs)

    # Step 2: quality filter (no self-loops, non-empty text on both sides).
    cand = [(h, t) for h, t in pairs
            if h != t and _entity_text(h, entity_dict, memo) and _entity_text(t, entity_dict, memo)]
    if not cand:
        cand = list(pairs)

    # Step 3: deterministic cap (crc32 pseudo-random subsample, then canonical order).
    if len(cand) > cap:
        cand = sorted(cand, key=lambda p: zlib.crc32(f'{p[0]}|{p[1]}'.encode('utf-8')))[:cap]
    cand = sorted(cand)

    w = torch.tensor([counts[h] ** (-power) for h, _ in cand], dtype=torch.float64)

    # Step 4: frozen-encoder embeddings (no neighbour text -> query independent).
    emb = _embed_pairs(model, tokenizer, device, cand, relation, entity_dict, memo, batch_size).double()

    # Step 5: centred, re-normalised cosine mapped to [0, 1].
    if emb.size(0) > 1:
        emb = emb - emb.mean(dim=0, keepdim=True)
        emb = torch.nn.functional.normalize(emb, dim=1)
    sim = ((emb @ emb.t() + 1.0) / 2.0).clamp(0.0, 1.0)

    # Step 6: greedy facility location.
    picked = _greedy_facility_location(sim, w, table_size)
    return [cand[i] for i in picked]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _build_table(relation2pairs, table_size, cap, power, batch_size):
    from transformers import AutoModel

    entity_dict = get_entity_dict()
    init_tokenizer(args)
    tokenizer = get_tokenizer()

    if torch.cuda.is_available():
        idx = args.local_rank if getattr(args, 'distributed', False) else args.gpu
        device = torch.device(f'cuda:{idx}')
    else:
        device = torch.device('cpu')

    model = AutoModel.from_pretrained(args.pretrained_model).to(device).eval()
    memo: Dict[str, str] = {}
    table: Dict[str, List[Tuple[str, str]]] = {}

    relations = sorted(relation2pairs)
    start = time.time()
    for i, r in enumerate(relations):
        table[r] = _select_for_relation(
            r, relation2pairs[r], table_size, cap, power,
            model, tokenizer, device, entity_dict, memo, batch_size)
        if (i + 1) % 50 == 0 or i + 1 == len(relations):
            logger.info(f'Global anchor selection: {i + 1}/{len(relations)} relations '
                        f'({time.time() - start:.0f}s)')

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return table


def load_or_build_global_anchor_table(relation2pairs: Dict[str, List[Tuple[str, str]]],
                                      table_size: int, cap: int, power: float,
                                      batch_size: int) -> Dict[str, List[Tuple[str, str]]]:
    """Return {relation: ranked [(head_id, tail_id), ...]} of length <= table_size."""
    path = _cache_path(table_size, cap, power)

    if os.path.exists(path):
        logger.info(f'Loading fixed global anchors from {path}')
        return _load_table(path)

    if getattr(args, 'distributed', False) and args.rank != 0:
        logger.info(f'Rank {args.rank}: waiting for rank 0 to build {path}')
        waited = 0
        while not os.path.exists(path):
            if waited > _WAIT_TIMEOUT_S:
                raise TimeoutError(f'Timed out waiting for {path}')
            time.sleep(_POLL_S)
            waited += _POLL_S
        time.sleep(2)  # let the atomic rename settle on network filesystems
        return _load_table(path)

    logger.info(f'Building fixed global anchors (facility location): table_size={table_size}, '
                f'cap={cap}, power={power}')
    table = _build_table(relation2pairs, table_size, cap, power, batch_size)
    _save_table_atomic(table, path)
    logger.info(f'Saved fixed global anchors to {path}')
    return table
