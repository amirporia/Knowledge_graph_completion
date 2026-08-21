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


def clean_state_dict(state_dict: dict) -> OrderedDict:
    """Remove 'module.' prefix from DataParallel/DDP state dict."""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[len('module.'):] if key.startswith('module.') else key
        new_state_dict[clean_key] = value
    return new_state_dict


class ARPMPredictor:
    """Predictor class for ARPM-KGC inference.

    Mirrors Baseline/evaluation/predict.py::BertPredictor, but `predict_by_examples`
    returns the full memory bundle (query embedding, prototypes, structural memory,
    gates) needed to compute S(t|h,r) = S_q + lambda_p*S_p + lambda_s*S_struct
    against the full entity set in evaluation/evaluate.py, instead of just a
    single hr_vector.
    """

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False
        self.device = None

    def load(self, ckt_path: str, use_data_parallel: bool = False) -> None:
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

        logger.info(
            'Training arguments:\n' +
            json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)
        )

        if hasattr(self.train_args, 'use_link_graph'):
            args.__dict__['use_link_graph'] = self.train_args.use_link_graph
        args.__dict__['is_test'] = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]) -> dict:
        """Predict the ARPM-KGC memory bundle for a list of query examples.

        Returns a dict of concatenated tensors:
          q:          (N, d)
          prototypes: (N, K, d)
          m_struct:   (N, d)
          lambda_p:   (N,)
          lambda_s:   (N,)
          slot_gate:  (N, K) or None (only when --use-gumbel-proto)
        """
        data_loader = self._create_dataloader(examples, is_test=False)

        q_list, proto_list, struct_list = [], [], []
        lambda_p_list, lambda_s_list = [], []
        slot_gate_list = []
        has_slot_gate = False

        for batch_dict in tqdm.tqdm(data_loader, desc='Predicting query memory'):
            model_kwargs = {k: v for k, v in batch_dict.items()
                             if k not in ('triplet_mask', 'self_negative_mask', 'batch_data', 'test_forward')}
            batch_dict_dev = self._move_to_device(model_kwargs)
            outputs = self.model(**batch_dict_dev)

            q_list.append(outputs['q'])
            proto_list.append(outputs['prototypes'])
            struct_list.append(outputs['m_struct'])
            lambda_p_list.append(outputs['lambda_p'])
            lambda_s_list.append(outputs['lambda_s'])
            if 'slot_gate' in outputs:
                has_slot_gate = True
                slot_gate_list.append(outputs['slot_gate'])

        result = {
            'q': torch.cat(q_list, dim=0),
            'prototypes': torch.cat(proto_list, dim=0),
            'm_struct': torch.cat(struct_list, dim=0),
            'lambda_p': torch.cat(lambda_p_list, dim=0),
            'lambda_s': torch.cat(lambda_s_list, dim=0),
            'slot_gate': torch.cat(slot_gate_list, dim=0) if has_slot_gate else None,
        }
        return result

    @torch.no_grad()
    def predict_by_entities(self, entity_exs: List) -> torch.Tensor:
        """Predict E_1(t) embeddings for every entity in the dictionary."""
        examples = [
            Example(head_id='', relation='', tail_id=entity_ex.entity_id)
            for entity_ex in entity_exs
        ]

        data_loader = self._create_dataloader(examples, is_test=True)
        ent_tensors = []

        for batch_dict in tqdm.tqdm(data_loader, desc='Predicting entities'):
            batch_dict = self._move_to_device(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensors.append(outputs['ent_vectors'])

        return torch.cat(ent_tensors, dim=0)

    def _create_dataloader(self, examples: List[Example], is_test: bool) -> torch.utils.data.DataLoader:
        dataset = Dataset(path='', examples=examples, test_set=is_test)
        collate_fn = collate_entity if is_test else collate

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=self.use_cuda
        )

    def _move_to_device(self, batch_dict: dict) -> dict:
        if self.use_cuda:
            batch_dict = move_to_cuda(batch_dict)
        return batch_dict
