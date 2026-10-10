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
from ..utils.doc import collate, collate_entity, Example, Dataset
from ..utils.utils import AttrDict, move_to_cuda

_NON_MODEL_KEYS = ('triplet_mask', 'self_negative_mask', 'batch_data', 'test_forward')


def clean_state_dict(state_dict: dict) -> OrderedDict:
    """Remove 'module.' prefix from DataParallel/DDP state dict."""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[len('module.'):] if key.startswith('module.') else key
        new_state_dict[clean_key] = value
    return new_state_dict


class ARPMPredictor:
    """Inference wrapper. `predict_by_examples` returns the memory bundle needed to compute
    S = cos(q,e) + cos(q_hrta,e) + lambda_p S_p + lambda_s S_struct against the full entity set."""

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False
        self.device = None
        self.batch_size = args.batch_size

    @classmethod
    def from_model(cls, model: torch.nn.Module, device: torch.device,
                   use_cuda: bool, batch_size: int = None) -> 'ARPMPredictor':
        predictor = cls()
        predictor.model = model
        predictor.device = device
        predictor.use_cuda = use_cuda
        predictor.batch_size = batch_size or args.batch_size
        return predictor

    def load(self, ckt_path: str, use_data_parallel: bool = False) -> None:
        if not os.path.exists(ckt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckt_path}")

        ckt_dict = torch.load(ckt_path, map_location='cpu')
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        init_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        self.model.load_state_dict(clean_state_dict(ckt_dict['state_dict']), strict=True)
        self.model.eval()
        self._setup_device(use_data_parallel)
        logger.info(f'Model loaded successfully from {ckt_path}')

    def _setup_device(self, use_data_parallel: bool) -> None:
        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Using DataParallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
            self.device = torch.device('cuda')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            self.use_cuda = True
            logger.info(f'Using device: {self.device}')
        else:
            self.device = torch.device('cpu')
            logger.info('Using CPU for inference')

    def _setup_args(self) -> None:
        for key, value in args.__dict__.items():
            if key not in self.train_args.__dict__:
                logger.info(f'Setting default attribute: {key}={value}')
                self.train_args.__dict__[key] = value

        logger.info('Training arguments:\n' +
                    json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4))

        if hasattr(self.train_args, 'use_link_graph'):
            args.__dict__['use_link_graph'] = self.train_args.use_link_graph
        # the candidate pool / RAA path must be built exactly as in training
        for key in ('anchor_num', 'num_hops', 'local_per_hop_budget', 'global_budget', 'anchor_budget'):
            if hasattr(self.train_args, key):
                args.__dict__[key] = getattr(self.train_args, key)
        args.__dict__['is_test'] = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]) -> dict:
        """Returns concatenated tensors:
           q (N,d), q_hrta (N,d), prototypes (N,K,d), m_struct (N,d), lambda_p (N,), lambda_s (N,)"""
        data_loader = self._create_dataloader(examples, is_test=False)
        keys = ('q', 'q_hrta', 'prototypes', 'm_struct', 'lambda_p', 'lambda_s')
        collected = {k: [] for k in keys}

        for batch_dict in tqdm.tqdm(data_loader, desc='Predicting query memory'):
            model_kwargs = {k: v for k, v in batch_dict.items() if k not in _NON_MODEL_KEYS}
            outputs = self.model(**self._move_to_device(model_kwargs))
            for k in keys:
                collected[k].append(outputs[k].float())

        return {k: torch.cat(v, dim=0) for k, v in collected.items()}

    @torch.no_grad()
    def predict_by_entities(self, entity_exs: List) -> torch.Tensor:
        examples = [Example(head_id='', relation='', tail_id=e.entity_id) for e in entity_exs]
        data_loader = self._create_dataloader(examples, is_test=True)
        ent_tensors = []
        for batch_dict in tqdm.tqdm(data_loader, desc='Predicting entities'):
            outputs = self.model(**self._move_to_device(batch_dict))
            ent_tensors.append(outputs['ent_vectors'])
        return torch.cat(ent_tensors, dim=0)

    def _create_dataloader(self, examples: List[Example], is_test: bool) -> torch.utils.data.DataLoader:
        dataset = Dataset(path='', examples=examples, test_set=is_test)
        collate_fn = collate_entity if is_test else collate
        return torch.utils.data.DataLoader(
            dataset, num_workers=4, batch_size=self.batch_size,
            collate_fn=collate_fn, shuffle=False, pin_memory=self.use_cuda
        )

    def _move_to_device(self, batch_dict: dict) -> dict:
        if self.use_cuda:
            batch_dict = move_to_cuda(batch_dict)
        return batch_dict
