import json
import os
from collections import OrderedDict
from typing import List

import torch
import torch.utils.data
import tqdm

from Baseline.model.models import build_model
from Baseline.setting.config import args
from Baseline.setting.logger_config import logger
from Baseline.utils.dict_hub import init_tokenizer
from Baseline.utils.doc import collate, Example, Dataset, collate_test
from Baseline.utils.utils import AttrDict, move_to_cuda, get_model_obj


def clean_state_dict(state_dict: dict) -> OrderedDict:
    """Remove 'module.' prefix from DataParallel state dict."""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[len('module.'):] if key.startswith('module.') else key
        new_state_dict[clean_key] = value
    return new_state_dict


class BertPredictor:
    """Predictor class for BERT-based model inference."""

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False
        self.device = None
        self.batch_size = args.batch_size

    @classmethod
    def from_model(cls, model: torch.nn.Module, device: torch.device,
                   use_cuda: bool, batch_size: int = None) -> 'BertPredictor':
        predictor = cls()
        predictor.model = model
        predictor.device = device
        predictor.use_cuda = use_cuda
        predictor.batch_size = batch_size or args.batch_size
        return predictor

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

        logger.info(f'Model loaded successfully from {ckt_path}')

    def _setup_device(self, use_data_parallel: bool) -> None:
        """Configure model device placement."""
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
        """Configure arguments with defaults and update global config."""
        for key, value in args.__dict__.items():
            if key not in self.train_args.__dict__:
                logger.info(f'Setting default attribute: {key}={value}')
                self.train_args.__dict__[key] = value

        logger.info(
            'Training arguments:\n' +
            json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)
        )

        if hasattr(self.train_args, 'use_link_graph'):
            args.__dict__['use_link_graph'] = self.train_args.use_link_graph
        args.__dict__['is_test'] = True

    def _autocast(self):
        """fp16 autocast for the encoders when --eval-amp is given (training also used AMP)."""
        return torch.autocast('cuda', dtype=torch.float16,
                              enabled=bool(self.use_cuda and getattr(args, 'eval_amp', False)))

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]) -> tuple:
        """
        Predict embeddings for query examples.

        Returns:
            (hr_vectors e_hr, tail_vectors e_t, related_hr_vectors e^avg_hrta)
        """
        data_loader = self._create_dataloader(examples, is_test=False)

        # Anchor tensors are batch-first but we always run the unwrapped module here (single GPU).
        model = get_model_obj(self.model)

        hr_tensors, tail_tensors, related_hr_tensors = [], [], []

        for batch_dict in data_loader:
            batch_dict = self._move_to_device(batch_dict)
            with self._autocast():
                outputs = model(**batch_dict)

            hr_tensors.append(outputs['hr_vector'].float())
            tail_tensors.append(outputs['tail_vector'].float())
            related_hr_tensors.append(outputs['related_hr_vector'].float())

        return (
            torch.cat(hr_tensors, dim=0),
            torch.cat(tail_tensors, dim=0),
            torch.cat(related_hr_tensors, dim=0)
        )

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
            with self._autocast():
                outputs = self.model(**batch_dict)
            ent_tensors.append(outputs['ent_vectors'].float())

        return torch.cat(ent_tensors, dim=0)

    def _create_dataloader(
            self,
            examples: List[Example],
            is_test: bool
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader from examples."""
        dataset = Dataset(path='', examples=examples, test_set=is_test)
        collate_fn = collate_test if is_test else collate

        # Entity encoding is cheap per item and can use a big batch (--entity-batch-size per GPU);
        # query encoding keeps the small --batch-size.
        batch_size = self.batch_size
        entity_bs = getattr(args, 'entity_batch_size', 0)
        if is_test and entity_bs > 0:
            batch_size = entity_bs * max(1, torch.cuda.device_count())

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=self.use_cuda
        )

    def _move_to_device(self, batch_dict: dict) -> dict:
        """Move batch dictionary to appropriate device."""
        if self.use_cuda:
            batch_dict = move_to_cuda(batch_dict)
        return batch_dict