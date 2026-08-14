"""
Training objective (Sec 4.11):

    L_final = CE(S, y)
    L_p     = CE(S_p, y)                      (optional, Sec 4.10/4.11)
    L_retr  = L_rel + eta_div * L_div          (optional, Sec 4.4.1)
    L       = L_final + eta_p * L_p + eta_r * L_retr

`build_training_scores` assembles the in-batch score matrices (including
the optional self-negative column and the false-negative batch mask), and
`compute_loss` combines everything per `ARPMConfig`'s weights/toggles.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import ARPMConfig
from .data.triplets import TripletDict
from .masking import build_batch_mask, build_self_negative_mask
from .model import ARPMKGCModel, ModelOutput


@dataclass
class LossOutput:
    total: torch.Tensor
    final: torch.Tensor
    prototype: torch.Tensor
    retrieval: torch.Tensor
    diversity: torch.Tensor
    relevance: torch.Tensor


def build_training_scores(model: ARPMKGCModel, output: ModelOutput, batch: dict,
                           train_triplet_dict: TripletDict) -> Dict[str, torch.Tensor]:
    cfg = model.config
    device = output.qe.device
    batch_size = output.qe.size(0)

    scores = model.score(
        output.qe, output.prototypes, output.lambda_mem, output.tail_vector,
        apply_margin_diagonal=True,
    )
    s_q, s_p, s = scores["Sq"], scores["Sp"], scores["S"]

    valid_mask = build_batch_mask(batch["head_ids"], batch["relations"], batch["tail_ids"],
                                   train_triplet_dict).to(device)

    if cfg.use_self_negative:
        self_neg_score = model.self_negative_score(output.qe, output.head_vector)  # [B]
        self_neg_valid = build_self_negative_mask(batch["head_ids"], batch["relations"],
                                                    train_triplet_dict).to(device)
        s_q = torch.cat([s_q, self_neg_score.unsqueeze(1)], dim=1)
        s = torch.cat([s, self_neg_score.unsqueeze(1)], dim=1)
        valid_mask = torch.cat([valid_mask, self_neg_valid.unsqueeze(1)], dim=1)

    s = s.masked_fill(~valid_mask, float("-1e4"))
    s_q = s_q.masked_fill(~valid_mask, float("-1e4"))

    target = torch.arange(batch_size, device=device)

    result = {"S": s, "Sq": s_q, "target": target}
    if s_p is not None:
        p_valid_mask = valid_mask[:, :batch_size]
        result["Sp"] = s_p.masked_fill(~p_valid_mask, float("-1e4"))
    return result


def compute_loss(model: ARPMKGCModel, output: ModelOutput, batch: dict,
                  train_triplet_dict: TripletDict) -> LossOutput:
    cfg = model.config
    scores = build_training_scores(model, output, batch, train_triplet_dict)

    l_final = F.cross_entropy(scores["S"], scores["target"])

    l_p = torch.zeros((), device=output.qe.device)
    if cfg.use_prototype_loss and "Sp" in scores:
        l_p = F.cross_entropy(scores["Sp"], scores["target"])

    l_div = torch.zeros((), device=output.qe.device)
    if cfg.use_diversity_regularization and output.alpha is not None and output.anchors is not None:
        l_div = model.anchor_selector.diversity_loss(output.alpha, output.anchors, output.anchor_mask)

    l_rel = torch.zeros((), device=output.qe.device)
    if cfg.use_retrieval_supervision and output.raw_scores is not None:
        anchor_tail_ids = batch.get("anchor_tail_ids")
        if anchor_tail_ids is not None:
            l_rel = model.anchor_selector.retrieval_supervision_loss(
                output.raw_scores, output.anchor_mask, anchor_tail_ids, batch["tail_ids"],
            )

    l_retr = l_rel + cfg.diversity_weight * l_div
    total = l_final + cfg.prototype_loss_weight * l_p + cfg.retrieval_loss_weight * l_retr

    return LossOutput(total=total, final=l_final, prototype=l_p, retrieval=l_retr,
                       diversity=l_div, relevance=l_rel)
