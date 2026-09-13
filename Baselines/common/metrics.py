from typing import Dict, Set, Tuple

import torch
import tqdm


@torch.no_grad()
def evaluate_mrr(model, eval_triples: torch.LongTensor,
                  true_tail_filter: Dict[Tuple[int, int], Set[int]],
                  num_entities: int, batch_size: int = 256,
                  device: torch.device = None, desc: str = 'eval') -> Dict[str, float]:
    """Standard filtered ranking evaluation. Because every relation is already
    inverse-augmented (see common/data.py), scoring `(h, r, ?)` for both the original
    and inverse copy of a triple covers tail- and head-prediction in one code path.

    Requires `model.score_all(h_idx, r_idx) -> (B, num_entities)` (higher = more
    plausible), implemented by every model in embedding_models/.
    """
    device = device or next(model.parameters()).device
    model.eval()

    total = eval_triples.size(0)
    agg = {'mrr': 0.0, 'mr': 0.0, 'hits@1': 0.0, 'hits@3': 0.0, 'hits@10': 0.0}

    for start in tqdm.tqdm(range(0, total, batch_size), desc=desc, leave=False):
        batch = eval_triples[start:start + batch_size].to(device)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

        scores = model.score_all(h, r)  # (B, num_entities), higher is better

        # filter out every other known-true tail for (h, r), keep the target's own score
        for i in range(h.size(0)):
            key = (h[i].item(), r[i].item())
            known = true_tail_filter.get(key)
            if known:
                target = t[i].item()
                idx = torch.tensor([e for e in known if e != target],
                                    dtype=torch.long, device=device)
                if idx.numel() > 0:
                    scores[i].index_fill_(0, idx, float('-inf'))

        target_score = scores.gather(1, t.view(-1, 1))
        # rank = 1 + number of candidates strictly better than the target
        ranks = (scores > target_score).sum(dim=1).float() + 1.0

        agg['mrr'] += (1.0 / ranks).sum().item()
        agg['mr'] += ranks.sum().item()
        agg['hits@1'] += (ranks <= 1).sum().item()
        agg['hits@3'] += (ranks <= 3).sum().item()
        agg['hits@10'] += (ranks <= 10).sum().item()

    return {k: round(v / total, 4) for k, v in agg.items()}
