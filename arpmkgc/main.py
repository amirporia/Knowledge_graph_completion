"""
CLI entry point.

    python -m arpmkgc.main train --ablation full --task wn18rr
    python -m arpmkgc.main train --ablation A9 --task fb15k237 --epochs 10
    python -m arpmkgc.main evaluate --ablation full --task wn18rr \
        --eval-model-path data/WN18RR/checkpoint/full/model_best.mdl
    python -m arpmkgc.main list-ablations
"""

import sys

from .ablations import ABLATIONS
from .config import parse_args
from .logging_utils import logger


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
    logger.info(f"Config ({config.ablation_name}):\n{config.to_dict()}")
    config.save(f"{config.model_dir}/config.json")

    if command == "train":
        from .trainer import Trainer
        Trainer(config).train_loop()
    elif command == "evaluate":
        from .evaluate import predict_by_split
        config.is_test = True
        predict_by_split(config)


if __name__ == "__main__":
    _dispatch()
