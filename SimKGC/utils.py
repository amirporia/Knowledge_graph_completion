import os
import glob
import torch
import shutil

import numpy as np
import torch.nn as nn

from logger_config import logger


class AttrDict:
    pass


def save_checkpoint(state: dict, is_best: bool, filename: str, eval_state: dict = None) -> None:
    """Persist a full training checkpoint (model + optimizer + scheduler + AMP
    scaler + epoch + best-metric/early-stopping bookkeeping) to `filename`, and
    mirror it to `model_last.mdl` so training can always be resumed from the
    most recent state via `--resume`/`--resume-path`.

    If `is_best`, also write `model_best.mdl` -- using the lighter `eval_state`
    (just epoch/args/state_dict, all `predict.py`/`evaluate.py` ever read) when
    one is given, so the checkpoint used for evaluation/deployment doesn't carry
    around optimizer/scheduler state it doesn't need.
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    torch.save(state, filename)
    shutil.copyfile(filename, os.path.join(dirname, 'model_last.mdl'))
    if is_best:
        torch.save(eval_state if eval_state is not None else state,
                  os.path.join(dirname, 'model_best.mdl'))


def delete_old_ckt(path_pattern: str, keep: int = 5) -> None:
    """Delete old checkpoint files, keeping only the most recent `keep`.

    BUGFIX: this previously ran `os.system('rm -f {}'.format(f))` -- a
    Unix-only shell command. On Windows (outside WSL/git-bash) there is no
    `rm` on PATH, so `os.system` just returns a non-zero exit code that this
    function never checked: old checkpoints silently piled up forever instead
    of being deleted. `os.remove` is cross-platform and matches what
    ARPM_KGC's own `utils/utils.py::delete_old_checkpoints` already does.
    """
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        logger.info('Delete old checkpoint {}'.format(f))
        try:
            os.remove(f)
        except OSError as e:
            logger.error('Failed to delete {}: {}'.format(f, e))


def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    assert isinstance(model, torch.nn.Module), 'Argument must be nn.Module'

    num_parameters = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_parameters += np.prod(list(p.size()))
            logger.info('{}: {}'.format(name, np.prod(list(p.size()))))

    logger.info('Number of parameters: {}M'.format(num_parameters // 10**6))
    return num_parameters


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
