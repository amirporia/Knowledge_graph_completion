"""
Filtered ranking evaluation (Sec 7.2 protocol; Sec 4.12 inference procedure).

Mirrors a standard KGC evaluation loop (forward head->tail and backward
tail->head via the "inverse relation" convention, then averaged), but is
independently implemented against `ARPMKGCModel.score` rather than any
baseline evaluation code.

Also persists per-example interpretability signals (lambda_mem, hop weights
beta, prototype weights gamma, top anchors) requested by the proposal's
Sec 8 ("Analysis of When ARPM-KGC Helps") -- e.g. correlating lambda_mem
with candidate availability D(h, r), or hop-weight distributions with
performance on sparse-neighborhood queries.
"""

import json
import os
from dataclasses import dataclass, asdict
from time import time
from typing import Dict, List, Optional

import torch
import tqdm
from torch.utils.data import DataLoader

from .config import ARPMConfig
from .data.dataset import ARPMDataset, Collator, Tokenization, load_examples
from .data.dict_hub import get_all_triplet_dict, get_entity_dict
from .logging_utils import logger
from .metrics import compute_ranks, filter_known_answers, summarize_ranks
from .model import ARPMKGCModel, to_model_output
from .utils import get_model_obj, move_to_cuda


@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    rank: int
    correct: bool
    lambda_mem: Optional[float] = None
    top_hop: Optional[int] = None
    num_active_prototypes: Optional[int] = None
    num_anchors_used: Optional[int] = None


@torch.no_grad()
def compute_entity_embeddings(model: ARPMKGCModel, entity_ids: List[str], tokenization: Tokenization,
                              batch_size: int, use_cuda: bool) -> torch.Tensor:
    model.eval()
    embeddings = []
    for start in tqdm.tqdm(range(0, len(entity_ids), batch_size), desc="Encoding entities"):
        chunk = entity_ids[start:start + batch_size]
        encoded = [tokenization.encode_entity(e) for e in chunk]
        batch = {
            "input_ids": _pad([e["input_ids"] for e in encoded], tokenization.tokenizer.pad_token_id),
            "token_type_ids": _pad([e["token_type_ids"] for e in encoded], 0),
        }
        lengths = [len(e["input_ids"]) for e in encoded]
        mask = torch.zeros_like(batch["input_ids"])
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        batch["attention_mask"] = mask
        if use_cuda:
            batch = move_to_cuda(batch)
        embeddings.append(get_model_obj(model).encoder.encode_entity(batch))
    return torch.cat(embeddings, dim=0)


def _pad(seqs, pad_id):
    max_len = max((len(s) for s in seqs), default=1)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


@torch.no_grad()
def eval_single_direction(config: ARPMConfig, model: ARPMKGCModel, entity_tensor: torch.Tensor,
                          eval_forward: bool, use_cuda: bool, data_path: str,
                          batch_size: int = 64) -> Dict:
    start_time = time()
    model.eval()

    tokenization = Tokenization(config)
    entity_dict = get_entity_dict(config)
    all_triplet_dict = get_all_triplet_dict(config)

    raw_examples = load_examples(data_path, add_forward=eval_forward, add_backward=not eval_forward)
    dataset = ARPMDataset(config, data_path, tokenization=tokenization)
    dataset.examples = raw_examples  # reuse the dataset's candidate pipeline over just this direction

    collator = Collator(config, tokenization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator,
                        num_workers=config.workers)

    all_ranks: List[int] = []
    pred_infos: List[PredInfo] = []

    for batch in tqdm.tqdm(loader, desc=f"Eval ({'fwd' if eval_forward else 'bwd'})"):
        if use_cuda:
            batch = move_to_cuda(batch)
        output = to_model_output(model(batch))

        scores = get_model_obj(model).score(output.qe, output.prototypes, output.lambda_mem, entity_tensor,
                                            apply_margin_diagonal=False)["S"]

        target_idx = torch.tensor(
            [entity_dict.entity_to_idx(t) for t in batch["tail_ids"]], device=scores.device,
        )
        filter_known_answers(scores, batch["head_ids"], batch["relations"], target_idx,
                             all_triplet_dict, entity_dict)

        ranks, sorted_scores, sorted_indices = compute_ranks(scores, target_idx)
        all_ranks.extend(ranks.tolist())

        top1_idx = sorted_indices[:, 0]
        top1_score = sorted_scores[:, 0]
        for i in range(len(batch["head_ids"])):
            pred_infos.append(PredInfo(
                head=batch["head_ids"][i], relation=batch["relations"][i], tail=batch["tail_ids"][i],
                pred_tail=entity_dict.get_by_idx(top1_idx[i].item()).entity_id,
                pred_score=round(top1_score[i].item(), 4),
                rank=int(ranks[i].item()),
                correct=bool(top1_idx[i].item() == target_idx[i].item()),
                lambda_mem=round(output.lambda_mem[i].item(), 4),
                top_hop=int(output.beta[i].argmax().item()) + 1 if output.beta is not None else None,
                num_active_prototypes=int(output.slot_mask[i].sum().item()) if output.slot_mask is not None else None,
                num_anchors_used=int(output.anchor_mask[i].sum().item()) if output.anchor_mask is not None else None,
            ))

    metrics = summarize_ranks(all_ranks)
    direction = "forward" if eval_forward else "backward"
    logger.info(f"{direction} metrics: {json.dumps(metrics)}")
    logger.info(f"Evaluation took {round(time() - start_time, 3)}s")
    return {"metrics": metrics, "pred_infos": pred_infos}


def predict_by_split(config: ARPMConfig) -> Dict:
    use_cuda = torch.cuda.is_available()

    from .predict import ARPMPredictor
    predictor = ARPMPredictor(config)
    predictor.load(config.eval_model_path, use_data_parallel=config.data_parallel)
    model = predictor.model
    # Everything architecture/data-generation sensitive (max_hop,
    # candidate_budget, anchor_selection_mode, ...) must come from the
    # checkpoint's own config, not the raw CLI config `config` -- the two
    # only need to agree on path/system fields, which ARPMPredictor.load
    # already merges in (see `_RUNTIME_OVERRIDE_FIELDS`).
    eval_config = predictor.config
    data_path = eval_config.test_path if config.eval_split == "test" else eval_config.valid_path

    entity_dict = get_entity_dict(eval_config)
    entity_ids = [ex.entity_id for ex in entity_dict.entity_exs]
    entity_tensor = compute_entity_embeddings(model, entity_ids, predictor.tokenization,
                                              batch_size=eval_config.batch_size, use_cuda=use_cuda)
    if use_cuda:
        entity_tensor = entity_tensor.cuda()

    forward_result = eval_single_direction(eval_config, model, entity_tensor, eval_forward=True,
                                           use_cuda=use_cuda, data_path=data_path)
    backward_result = eval_single_direction(eval_config, model, entity_tensor, eval_forward=False,
                                            use_cuda=use_cuda, data_path=data_path)

    averaged = {
        k: round((forward_result["metrics"][k] + backward_result["metrics"][k]) / 2, 4)
        for k in forward_result["metrics"]
    }
    logger.info(f"Averaged metrics ({config.eval_split}): {averaged}")

    prefix = os.path.dirname(eval_config.eval_model_path)
    basename = os.path.basename(eval_config.eval_model_path)
    split = os.path.basename(data_path)

    with open(os.path.join(prefix, f"metrics_{split}_{basename}.json"), "w", encoding="utf-8") as f:
        json.dump({"forward": forward_result["metrics"], "backward": backward_result["metrics"],
                   "average": averaged}, f, indent=2)

    for direction, result in (("forward", forward_result), ("backward", backward_result)):
        out_path = os.path.join(prefix, f"predictions_{split}_{direction}_{basename}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in result["pred_infos"]], f, ensure_ascii=False, indent=2)

    return {"forward": forward_result["metrics"], "backward": backward_result["metrics"], "average": averaged}


if __name__ == "__main__":
    from .config import parse_args

    cfg = parse_args()
    cfg.is_test = True
    predict_by_split(cfg)
