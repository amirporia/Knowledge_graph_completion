import json
import os
from collections import OrderedDict
from typing import List

import torch
import torch.utils.data
import tqdm

from ..model.models import build_model
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import init_tokenizer
from ..utils.doc import collate, Example, Dataset
from ..utils.utils import AttrDict, move_to_cuda

# Settings that change how examples are tokenized: they must follow the checkpoint, not the CLI defaults.
_SYNCED_ARGS = ('use_link_graph', 'method', 'entity_text', 'max_num_tokens')


def clean_state_dict(state_dict: dict) -> OrderedDict:
    """Remove 'module.' prefix from DataParallel / DDP state dict."""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[len('module.'):] if key.startswith('module.') else key
        new_state_dict[clean_key] = value
    return new_state_dict


class BertPredictor:
    """Predictor class for model inference (SimKGC and HaSa checkpoints)."""

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False
        self.device = None
        # value used to mask known true tails in filtered ranking: must be below every real score
        self.score_floor = -1.0

    def load(self, ckt_path: str, use_data_parallel: bool = False) -> None:
        """Load model from checkpoint."""
        if not os.path.exists(ckt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckt_path}")

        ckt_dict = torch.load(ckt_path, map_location='cpu')
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        init_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        # DataParallel introduces a 'module.' prefix
        self.model.load_state_dict(clean_state_dict(ckt_dict['state_dict']), strict=True)
        self.model.eval()

        # cosine scores live in [-1, 1]; HaSa dot products are unbounded
        self.score_floor = -1.0 if getattr(self.model, 'score_normalized', True) else -1e9

        self._setup_device(use_data_parallel)

        logger.info(f'Model loaded successfully from {ckt_path} (method={self.train_args.method})')

    def _setup_device(self, use_data_parallel: bool) -> None:
        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Using DataParallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
            self.device = torch.device('cuda')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.model.to(self.device)
            self.use_cuda = True
            logger.info(f'Using device: {self.device}')
        else:
            self.device = torch.device('cpu')
            logger.info('Using CPU for inference')

    def _setup_args(self) -> None:
        """Fill missing training args from the current config and switch the global config to test mode."""
        for key, value in args.__dict__.items():
            if key not in self.train_args.__dict__:
                logger.info(f'Set default attribute: {key}={value}')
                self.train_args.__dict__[key] = value

        logger.info(
            'Args used in training:\n' +
            json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)
        )

        for key in _SYNCED_ARGS:
            setattr(args, key, getattr(self.train_args, key))
        args.is_test = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        """Predict (head, relation) and tail embeddings for the given examples."""
        data_loader = self._create_dataloader(
            examples, batch_size=max(args.batch_size, 512), num_workers=1
        )

        hr_tensor_list, tail_tensor_list = [], []
        for batch_dict in data_loader:
            batch_dict = self._move_to_device(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            tail_tensor_list.append(outputs['tail_vector'])

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

    @torch.no_grad()
    def predict_by_entities(self, entity_exs: List) -> torch.Tensor:
        """Predict embeddings for entities."""
        examples = [
            Example(head_id='', relation='', tail_id=entity_ex.entity_id)
            for entity_ex in entity_exs
        ]

        data_loader = self._create_dataloader(
            examples, batch_size=max(args.batch_size, 1024), num_workers=2
        )

        ent_tensor_list = []
        for batch_dict in tqdm.tqdm(data_loader):
            batch_dict['only_ent_embedding'] = True
            batch_dict = self._move_to_device(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)

    @staticmethod
    def _create_dataloader(examples: List[Example], batch_size: int,
                           num_workers: int) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            Dataset(path='', examples=examples),
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate,
            shuffle=False
        )

    def _move_to_device(self, batch_dict: dict) -> dict:
        if self.use_cuda:
            batch_dict = move_to_cuda(batch_dict)
        return batch_dict
