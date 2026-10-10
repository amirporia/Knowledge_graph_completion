import json
import os
from collections import OrderedDict
from typing import List, Dict

import torch
import torch.utils.data
import tqdm

from ..model.models import build_model
from ..setting.config import args
from ..setting.logger_config import logger
from ..utils.dict_hub import init_tokenizer
from ..utils.doc import collate, Example, Dataset, collate_test
from ..utils.utils import AttrDict, move_to_cuda, get_model_obj

# settings that must be identical to training for the candidate pool / RAA path
_TRAIN_TIME_KEYS = ('use_link_graph', 'use_memory', 'anchor_num', 'num_hops', 'local_per_hop_budget',
                    'global_budget', 'anchor_budget', 'disable_hop_anchors')
_MEMORY_KEYS = ('prototypes', 'm_struct', 'lambda_p', 'lambda_s')


def clean_state_dict(state_dict: dict) -> OrderedDict:
    """Remove 'module.' prefix from DataParallel state dict."""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[len('module.'):] if key.startswith('module.') else key
        new_state_dict[clean_key] = value
    return new_state_dict


class BertPredictor:
    """Predictor class for model inference (baseline + optional ARPM memory)."""

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False
        self.device = None
        self.batch_size = args.batch_size

    @classmethod
    def from_model(cls, model: torch.nn.Module, device: torch.device,
                   use_cuda: bool, batch_size: int = None) -> 'BertPredictor':
        """Wrap an in-memory (already trained) model, e.g. for in-training validation."""
        predictor = cls()
        predictor.model = model
        predictor.device = device
        predictor.use_cuda = use_cuda
        predictor.batch_size = batch_size or args.batch_size
        return predictor

    @property
    def use_memory(self) -> bool:
        return bool(getattr(get_model_obj(self.model), 'use_memory', False))

    def load(self, ckt_path: str, use_data_parallel: bool = False) -> None:
        """Load model from checkpoint."""
        if not os.path.exists(ckt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckt_path}")

        ckt_dict = torch.load(ckt_path, map_location='cpu')
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        init_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        state_dict = ckt_dict['state_dict']
        new_state_dict = clean_state_dict(state_dict)
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        self._setup_device(use_data_parallel)

        logger.info(f'Model loaded successfully from {ckt_path} (memory={self.use_memory})')

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
        """Fill missing attributes with defaults and push train-time settings into the global config."""
        for key, value in args.__dict__.items():
            if key not in self.train_args.__dict__:
                logger.info(f'Setting default attribute: {key}={value}')
                self.train_args.__dict__[key] = value

        logger.info(
            'Training arguments:\n' +
            json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)
        )

        for key in _TRAIN_TIME_KEYS:
            if hasattr(self.train_args, key):
                args.__dict__[key] = getattr(self.train_args, key)
        args.__dict__['is_test'] = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]) -> Dict[str, torch.Tensor]:
        """Returns a dict of concatenated tensors:
            q (N,d)        classic query e_hr
            q_hrta (N,d)   anchor-enhanced query e^avg_hrta (anchors sampled from the TRAIN graph only)
          and, if the model has the ARPM memory branch:
            prototypes (N,K,d), m_struct (N,d), lambda_p (N,), lambda_s (N,)
        """
        data_loader = self._create_dataloader(examples, is_test=False)

        # Anchor tensors are batch-first (B, K, L); run the unwrapped module for simplicity.
        model = get_model_obj(self.model)
        use_memory = self.use_memory

        collected = {'q': [], 'q_hrta': []}
        if use_memory:
            collected.update({k: [] for k in _MEMORY_KEYS})

        for batch_dict in tqdm.tqdm(data_loader, desc='Predicting queries'):
            batch_dict = self._move_to_device(batch_dict)
            outputs = model(**batch_dict)

            collected['q'].append(outputs['hr_vector'].float())
            collected['q_hrta'].append(outputs['related_hr_vector'].float())
            if use_memory:
                for k in _MEMORY_KEYS:
                    collected[k].append(outputs[k].float())

        return {k: torch.cat(v, dim=0) for k, v in collected.items()}

    @torch.no_grad()
    def predict_by_entities(self, entity_exs: List) -> torch.Tensor:
        """Predict embeddings (encoder g2) for candidate entities."""
        examples = [
            Example(head_id='', relation='', tail_id=entity_ex.entity_id)
            for entity_ex in entity_exs
        ]

        data_loader = self._create_dataloader(examples, is_test=True)
        ent_tensors = []

        for batch_dict in tqdm.tqdm(data_loader, desc='Predicting entities'):
            batch_dict['only_ent_embedding'] = True
            batch_dict = self._move_to_device(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensors.append(outputs['ent_vectors'])

        return torch.cat(ent_tensors, dim=0)

    def _create_dataloader(self, examples: List[Example], is_test: bool) -> torch.utils.data.DataLoader:
        dataset = Dataset(path='', examples=examples, test_set=is_test)
        collate_fn = collate_test if is_test else collate

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=self.use_cuda
        )

    def _move_to_device(self, batch_dict: dict) -> dict:
        if self.use_cuda:
            batch_dict = move_to_cuda(batch_dict)
        return batch_dict
