import glob
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..setting.logger_config import logger


class AttrDict(dict):
    """Dictionary that allows attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def save_checkpoint(state: Dict[str, Any], is_best: bool, filename: str,
                    eval_state: Dict[str, Any] = None) -> None:
    """Persist a full training checkpoint, and mirror to `model_last.mdl` /
    (if `is_best`) `model_best.mdl`."""
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    if is_best:
        torch.save(eval_state if eval_state is not None else state,
                   os.path.join(dirname, "model_best.mdl"))
    torch.save(state, os.path.join(dirname, "model_last.mdl"))


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    """Load a checkpoint dict (model/optimizer/scheduler/epoch/...) from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    logger.info(f'Loading checkpoint from {path}')
    return torch.load(path, map_location=map_location)


def delete_old_checkpoints(path_pattern: str, keep: int = 5) -> None:
    """Delete old checkpoint files, keeping only the most recent ones."""
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)

    for file_path in files[keep:]:
        logger.info(f'Delete old checkpoint: {file_path}')
        try:
            os.remove(file_path)
        except OSError as e:
            logger.error(f'Failed to delete {file_path}: {e}')


def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    """Report the number of trainable parameters in a model."""
    assert isinstance(model, torch.nn.Module), 'Argument must be nn.Module'

    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            num_parameters += param_count
            logger.info(f'{name}: {param_count}')

    logger.info(f'Number of parameters: {num_parameters // 10 ** 6}M')
    return num_parameters


def get_model_obj(model: nn.Module) -> nn.Module:
    """Get the underlying model, unwrapping DataParallel/DDP if needed."""
    return model.module if hasattr(model, "module") else model


def move_to_device(obj: Any, device) -> Any:
    """Recursively move tensors to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    return obj


def move_to_cuda(sample: Any) -> Any:
    """Recursively move sample to the current process's CUDA device (or CPU if
    unavailable)."""
    if not sample:
        return {}

    device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')
    return move_to_device(sample, device)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ':f'):
        self.count = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.val = 0.0
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
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress across batches."""

    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
