import os
import random
import glob
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """Seed python/numpy/torch RNGs for reproducibility. No-op if seed is None."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Computes and stores the average and current value (same contract as
    Baseline/utils/utils.py::AverageMeter so logging looks familiar)."""

    def __init__(self, name: str, fmt: str = ':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EarlyStopping:
    """Generic early-stopping tracker keyed on a scalar metric (default: MRR, higher-is-better).

    `step(metric)` returns True iff this is a new best. After patience consecutive
    non-improving calls, `should_stop` becomes True.
    """

    def __init__(self, patience: int = 5, mode: str = 'max', min_delta: float = 1e-5):
        assert mode in ('max', 'min')
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.mode == 'max':
            return metric > self.best + self.min_delta
        return metric < self.best - self.min_delta

    def step(self, metric: float) -> bool:
        improved = self._is_improvement(metric)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def save_checkpoint(state: Dict[str, Any], is_best: bool, model_dir: str,
                     filename: str = 'model_last.mdl') -> None:
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, os.path.join(model_dir, filename))
    if is_best:
        torch.save(state, os.path.join(model_dir, 'model_best.mdl'))


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return torch.load(path, map_location=map_location)


def delete_old_checkpoints(path_pattern: str, keep: int = 3) -> None:
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        try:
            os.remove(f)
        except OSError:
            pass


def move_to_device(sample: Any, device: torch.device) -> Any:
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    if isinstance(sample, dict):
        return {k: move_to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, (list, tuple)):
        seq = [move_to_device(v, device) for v in sample]
        return type(sample)(seq) if isinstance(sample, tuple) else seq
    return sample
