"""
`ARPMPredictor`: loads a trained checkpoint (architecture config + weights +
relation vocabulary) and prepares the model/tokenizer for inference,
independent of the baseline's `BertPredictor`.
"""

import os
from collections import OrderedDict
from typing import Optional

import torch

from .config import ARPMConfig
from .data.dataset import Tokenization
from .data.dict_hub import get_relation_vocab
from .logging_utils import logger
from .model import ARPMKGCModel
from .utils import load_checkpoint, move_to_cuda

_RUNTIME_OVERRIDE_FIELDS = (
    "valid_path", "test_path", "train_path", "eval_model_path", "model_dir",
    "batch_size", "workers", "gpu", "use_amp", "is_test",
)


def clean_state_dict(state_dict: dict) -> OrderedDict:
    """Strips a leading 'module.' (from DataParallel/DDP) if present."""
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k[len("module."):] if k.startswith("module.") else k] = v
    return cleaned


class ARPMPredictor:
    def __init__(self, runtime_config: ARPMConfig):
        self.runtime_config = runtime_config
        self.config: Optional[ARPMConfig] = None
        self.model: Optional[ARPMKGCModel] = None
        self.tokenization: Optional[Tokenization] = None
        self.use_cuda = False

    def load(self, ckpt_path: str, use_data_parallel: bool = False) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        trained_config = ARPMConfig(**ckpt["config"])

        for field in _RUNTIME_OVERRIDE_FIELDS:
            setattr(trained_config, field, getattr(self.runtime_config, field))
        trained_config.is_test = True
        self.config = trained_config

        self.tokenization = Tokenization(self.config)
        relation_vocab = get_relation_vocab(self.config)
        num_relations_now = len(relation_vocab)
        num_relations_ckpt = ckpt["num_relations"]
        if num_relations_now != num_relations_ckpt:
            logger.warning(
                f"Relation vocabulary size mismatch: checkpoint was trained with "
                f"{num_relations_ckpt} relations, current data implies {num_relations_now}. "
                f"Using the checkpoint's size; unseen relations fall back to UNK."
            )
            num_relations = num_relations_ckpt
        else:
            num_relations = num_relations_now

        self.model = ARPMKGCModel(self.config, num_relations=num_relations)
        state_dict = clean_state_dict(ckpt["state_dict"])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self._setup_device(use_data_parallel)
        logger.info(f"Model loaded from {ckpt_path}")

    def _setup_device(self, use_data_parallel: bool) -> None:
        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info("Using DataParallel for inference")
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model = self.model.cuda()
            self.use_cuda = True
        else:
            logger.info("Using CPU for inference")

    def move_batch(self, batch: dict) -> dict:
        return move_to_cuda(batch) if self.use_cuda else batch
