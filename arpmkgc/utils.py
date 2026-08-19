"""
Generic training/inference utilities (checkpointing, meters, device moves).

Independent re-implementation; not imported from the baseline package.
"""

import glob
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .logging_utils import logger


class AttrDict(dict):
    """Dict that also supports attribute-style access (for loading saved configs)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class AverageMeter:
    """Tracks the running average of a scalar."""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Formats a row of AverageMeters for periodic logging."""

    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        logger.info("\t".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_model_obj(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel / DistributedDataParallel if present."""
    return model.module if hasattr(model, "module") else model


def move_to_device(obj: Any, device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, device) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj


def move_to_cuda(sample: Any) -> Any:
    if not sample:
        return {}
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() \
        else torch.device("cpu")
    return move_to_device(sample, device)


def save_checkpoint(state: Dict[str, Any], is_best: bool, filename: str,
                    eval_state: Dict[str, Any] = None) -> None:
    """Persist a full training checkpoint, and mirror to `model_last.mdl` /
    (if `is_best`) `model_best.mdl`. `eval_state` -- when given -- is the
    lightweight subset (config + weights only) written to model_best.mdl so
    evaluation/prediction scripts don't need to load optimizer/scheduler state.
    """
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    # torch.save(state, filename)

    if is_best:
        torch.save(eval_state if eval_state is not None else state,
                   os.path.join(dirname, "model_best.mdl"))
    torch.save(state, os.path.join(dirname, "model_last.mdl"))


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logger.info(f"Loading checkpoint from {path}")
    return torch.load(path, map_location=map_location)


def delete_old_checkpoints(path_pattern: str, keep: int = 4) -> None:
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for file_path in files[keep:]:
        logger.info(f"Deleting old checkpoint: {file_path}")
        try:
            os.remove(file_path)
        except OSError as e:
            logger.error(f"Failed to delete {file_path}: {e}")


def report_num_trainable_parameters(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {total / 1e6:.2f}M")
    return total
