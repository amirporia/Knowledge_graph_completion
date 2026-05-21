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
from ..utils.doc import collate, Example, Dataset, collate_test
from ..utils.utils import AttrDict, move_to_cuda


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

    def load(self, ckt_path: str, use_data_parallel: bool = False) -> None:
        """
        Load model from checkpoint.

        Args:
            ckt_path: Path to checkpoint file
            use_data_parallel: Whether to use DataParallel for multi-GPU
        """
        if not os.path.exists(ckt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckt_path}")

        ckt_dict = torch.load(ckt_path, map_location='cpu')
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        init_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        # Handle DataParallel state dict prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = clean_state_dict(state_dict)
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        # Setup device and distributed training
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
        # Add missing default arguments from global config
        for key, value in args.__dict__.items():
            if key not in self.train_args.__dict__:
                logger.info(f'Setting default attribute: {key}={value}')
                self.train_args.__dict__[key] = value

        logger.info(
            'Training arguments:\n' +
            json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)
        )

        # Update global config attributes for test mode
        if hasattr(self.train_args, 'use_link_graph'):
            args.__dict__['use_link_graph'] = self.train_args.use_link_graph
        args.__dict__['is_test'] = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]) -> tuple:
        """
        Predict embeddings for relation examples.

        Args:
            examples: List of Example objects

        Returns:
            Tuple of (hr_vectors, tail_vectors, related_hr_vectors)
        """
        data_loader = self._create_dataloader(examples, is_test=False)

        hr_tensors, tail_tensors, related_hr_tensors = [], [], []

        for batch_dict in data_loader:
            batch_dict = self._move_to_device(batch_dict)
            outputs = self.model(**batch_dict)

            hr_tensors.append(outputs['hr_vector'])
            tail_tensors.append(outputs['tail_vector'])
            related_hr_tensors.append(outputs['related_hr_vector'])

        return (
            torch.cat(hr_tensors, dim=0),
            torch.cat(tail_tensors, dim=0),
            torch.cat(related_hr_tensors, dim=0)
        )

    @torch.no_grad()
    def predict_by_entities(self, entity_exs: List) -> torch.Tensor:
        """
        Predict embeddings for entities.

        Args:
            entity_exs: List of entity examples

        Returns:
            Tensor of entity embeddings
        """
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

    def _create_dataloader(
            self,
            examples: List[Example],
            is_test: bool
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader from examples."""
        dataset = Dataset(path='', examples=examples, test_set=is_test)
        collate_fn = collate_test if is_test else collate

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=4,  # Consistent number of workers
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=self.use_cuda
        )

    def _move_to_device(self, batch_dict: dict) -> dict:
        """Move batch dictionary to appropriate device."""
        if self.use_cuda:
            batch_dict = move_to_cuda(batch_dict)
        return batch_dict
