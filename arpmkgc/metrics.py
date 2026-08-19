"""Training-time accuracy and evaluation-time filtered ranking metrics."""

from typing import Dict, List, Tuple

import torch

from .data.entities import EntityDict
from .data.triplets import TripletDict

# Whether a larger value is better, for every metric key that can appear in
# a validation metric_dict -- used by Trainer._check_best to compare
# checkpoints regardless of which metric is configured as the selection
# criterion. mean_rank is the one metric here where *lower* is better.
METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "mrr": True, "hit@1": True, "hit@3": True, "hit@10": True, "hit@50": True,
    "mean_rank": False,
    "Acc@1": True, "Acc@3": True,
}


@torch.no_grad()
def accuracy(scores: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """Top-k accuracy over an in-batch score matrix (diagnostic only)."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = scores.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k * (100.0 / batch_size))
    return results


@torch.no_grad()
def filter_known_answers(scores: torch.Tensor, head_ids: List[str], relations: List[str],
                         target_idx: torch.Tensor, all_triplet_dict: TripletDict,
                         entity_dict: EntityDict) -> None:
    """In-place filtered-ranking mask: for each row i, sets the score of
    every entity that is a *known* correct answer for (head_ids[i],
    relations[i]) -- other than the target itself -- to -inf, following the
    standard filtered evaluation protocol (Sec 7.2)."""
    for i in range(scores.size(0)):
        known = all_triplet_dict.get_neighbor_tails(head_ids[i], relations[i])
        if len(known) <= 1:
            continue
        idx_to_mask = [entity_dict.entity_to_idx(e) for e in known
                       if entity_dict.entity_to_idx(e) != target_idx[i].item()]
        if idx_to_mask:
            scores[i, torch.tensor(idx_to_mask, device=scores.device)] = float("-inf")


@torch.no_grad()
def compute_ranks(scores: torch.Tensor, target_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (ranks [B] (1-based), sorted_scores [B, N], sorted_indices [B, N])."""
    sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
    matches = sorted_indices.eq(target_idx.unsqueeze(-1))
    rank_pos = matches.float().argmax(
        dim=-1)  # first True position (argmax on bool->float is fine, exactly one True per row)
    ranks = rank_pos + 1
    return ranks, sorted_scores, sorted_indices


def summarize_ranks(ranks: List[int]) -> Dict[str, float]:
    n = len(ranks)
    mean_rank = sum(ranks) / n
    mrr = sum(1.0 / r for r in ranks) / n
    hits = {}
    for k in (1, 3, 10, 50):
        hits[f"hit@{k}"] = sum(1 for r in ranks if r <= k) / n
    return {
        "mean_rank": round(mean_rank, 4),
        "mrr": round(mrr, 4),
        **{k: round(v, 4) for k, v in hits.items()},
    }
