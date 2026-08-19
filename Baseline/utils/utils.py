import glob
import os
import shutil
from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..setting.logger_config import logger
from ..setting.config import args
from .dict_hub import get_link_graph
from .doc import Example
from .triplet import EntityDict


class AttrDict(dict):
    """Dictionary that allows attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def save_checkpoint(state: Dict[str, Any], is_best: bool, filename: str,
                    eval_state: Dict[str, Any] = None) -> None:
    """Save model checkpoint and maintain best/last model copies.

    `state` (which may include optimizer/scheduler/scaler tensors for resuming) is
    written to `filename` and mirrored to model_last.mdl. If `is_best`, model_best.mdl
    is written from `eval_state` when given (falling back to `state` otherwise) --
    this keeps the best-model file, which is what evaluation/prediction scripts load,
    free of optimizer/scheduler state it has no use for.
    """
    # torch.save(state, filename)
    dirname = os.path.dirname(filename)

    if is_best:
        torch.save(eval_state if eval_state is not None else state,
                   os.path.join(dirname, 'model_best.mdl'))
    shutil.copyfile(filename, os.path.join(dirname, 'model_last.mdl'))


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
    """Get the underlying model, unwrapping DataParallel if needed."""
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


def split_and_move_to_device(tensor_list: List, device_ids: List[int]) -> Dict[int, List]:
    """Distribute tensors across multiple devices."""
    num_gpus = len(device_ids)
    split_batch_dict = {device_id: [] for device_id in device_ids}

    for i, tensor in enumerate(tensor_list):
        device_id = device_ids[i % num_gpus]
        split_batch_dict[device_id].append(move_to_device(tensor, device_id))

    return split_batch_dict


def move_to_cuda(sample: Any) -> Any:
    """Recursively move sample to the current process's CUDA device (or CPU if unavailable).

    Uses torch.cuda.current_device() rather than a hardcoded index so this is correct
    both for single-GPU runs (whatever device Trainer/BertPredictor selected) and for
    each rank of a multi-GPU DDP run (after torch.cuda.set_device(local_rank)).
    """
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


def rerank_by_graph(related_batch_score: torch.tensor,
                    batch_score: torch.tensor,
                    examples: List[Example],
                    entity_dict: EntityDict) -> None:
    """
    Rerank batch scores using graph neighborhood information.

    Args:
        related_batch_score: Scores from related batch
        batch_score: Batch scores to be reranked
        examples: List of examples to process
        entity_dict: Entity dictionary for mapping
    """
    # Validate inductive setting
    if args.task == 'wiki5m_ind':
        assert args.neighbor_weight < 1e-6, 'Inductive setting cannot use re-rank strategy'

    # Early return if neighbor weight is negligible
    if args.neighbor_weight < 1e-6:
        return

    # Process each example in the batch
    for idx in range(batch_score.size(0)):
        current_example = examples[idx]

        # Get n-hop neighbor indices for the current head entity
        neighbor_indices = get_link_graph().get_n_hop_entity_indices(
            current_example.head_id,
            entity_dict=entity_dict,
            n_hop=args.rerank_n_hop
        )

        # Add neighbor weight to batch scores
        if neighbor_indices:
            delta_weights = torch.tensor(
                [args.neighbor_weight for _ in neighbor_indices],
                device=batch_score.device
            )
            neighbor_tensor = torch.LongTensor(list(neighbor_indices)).to(batch_score.device)
            batch_score[idx].index_add_(0, neighbor_tensor, delta_weights)

            # Sort related batch scores
            _, related_sorted_indices = torch.sort(
                related_batch_score, dim=-1, descending=True
            )
            # Weight adjustment for related neighbors (currently zero - placeholder)
            top_related_indices = related_sorted_indices[idx][:len(neighbor_indices)]
            zero_weights = torch.zeros(len(top_related_indices), device=batch_score.device)
            batch_score[idx].index_add_(0, top_related_indices, zero_weights)
