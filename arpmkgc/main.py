"""
CLI entry point.

    python -m arpmkgc.main train --ablation full --task wn18rr
    python -m arpmkgc.main train --ablation A9 --task fb15k237 --epochs 10
    python -m arpmkgc.main evaluate --ablation full --task wn18rr \
        --eval-model-path data/WN18RR/checkpoint/full/model_best.mdl
    python -m arpmkgc.main list-ablations

    # multi-GPU (DDP):
    torchrun --nproc_per_node=2 -m arpmkgc.main train --ablation full --task wn18rr

    # multi-GPU (DDP) on Kaggle's 2x T4 notebooks -- see KAGGLE_DDP.md at the
    # repo root for the full recipe. The two env-var/version fixes below
    # (`_configure_nccl_for_notebook_gpus`, `_init_process_group`) exist
    # specifically so this launch line doesn't hang or crash on Kaggle:
    #
    #   !torchrun --nproc_per_node=2 -m arpmkgc.main train --ablation full \
    #       --task wn18rr --batch-size 16 --deduplicate-anchors \
    #       --use-gradient-checkpointing --use-amp
"""

import os
import sys

import torch
import torch.distributed as dist

from .ablations import ABLATIONS
from .config import parse_args
from .logging_utils import logger


def _configure_nccl_for_notebook_gpus() -> None:
    """Kaggle's (and several other notebook/container) dual-GPU hosts run in
    a virtualized environment that does not support real GPU peer-to-peer
    (P2P/NVLink) or InfiniBand transports between the two T4s. NCCL probes
    for these by default and, when they're advertised but don't actually
    work, the process group either hangs indefinitely at
    `init_process_group()` or at the very first cross-GPU all-reduce/
    broadcast -- which on Kaggle shows up as `torchrun` appearing to freeze
    right after the "Building ARPM-KGC model" log line, with no error and no
    traceback.

    Forcing NCCL to fall back to a transport that *does* work in that
    environment (shared-memory/socket) avoids the hang. Every setting here
    uses `setdefault`, so anyone who has already exported a specific value
    (e.g. because they're on real multi-GPU hardware with working NVLink)
    is left untouched -- this only fills in a safe default for the common
    "single Kaggle notebook, 2x T4" case.
    """
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    # Quiets NCCL's normal init chatter but still surfaces real errors;
    # override with NCCL_DEBUG=INFO if you need to debug a hang.
    os.environ.setdefault("NCCL_DEBUG", "WARN")


def _init_process_group(config) -> None:
    """Sets the current process's GPU and creates the NCCL process group.

    `device_id=` (eager communicator init) is only accepted by
    `dist.init_process_group` on newer torch releases (~2.3+); Kaggle's
    preinstalled torch version varies by image, so we try the modern call
    first and transparently fall back to the older signature instead of
    crashing with a `TypeError: unexpected keyword argument 'device_id'`
    on an older image. Functionally this argument is only a performance
    hint -- dropping it does not change correctness.
    """
    torch.cuda.set_device(config.local_rank)
    device = torch.device(f"cuda:{config.local_rank}")
    try:
        dist.init_process_group(
            backend=config.dist_backend, init_method="env://", device_id=device,
        )
    except TypeError:
        dist.init_process_group(backend=config.dist_backend, init_method="env://")


def _dispatch() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "evaluate", "list-ablations"):
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    del sys.argv[1]

    if command == "list-ablations":
        for name in sorted(ABLATIONS):
            print(name)
        return

    config = parse_args()

    if config.distributed:
        _configure_nccl_for_notebook_gpus()
        _init_process_group(config)
        logger.info(
            f"Distributed training: rank {config.rank}/{config.world_size} "
            f"(local GPU {config.local_rank}, backend={config.dist_backend})"
        )

    if config.rank == 0:
        logger.info(f"Config ({config.ablation_name}):\n{config.to_dict()}")
        config.save(f"{config.model_dir}/config.json")

    try:
        if command == "train":
            from .trainer import Trainer
            Trainer(config).train_loop()
        elif command == "evaluate":
            from .evaluate import predict_by_split
            config.is_test = True
            predict_by_split(config)
    finally:
        if config.distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    _dispatch()
