import glob
import os
from typing import Optional

from transformers import AutoTokenizer

from .triplet import TripletDict, EntityDict, LinkGraph
from ..setting.config import args
from ..setting.logger_config import logger

# Module-level singletons
_train_triplet_dict: Optional[TripletDict] = None
_all_triplet_dict = None
_link_graph: Optional[LinkGraph] = None
_entity_dict: Optional[EntityDict] = None
_tokenizer: Optional[AutoTokenizer] = None


class _OverlayTripletDict:
    """train + valid/test triplets WITHOUT a second copy of the (huge) training graph.

    The old `_init_all_triplet_dict` loaded every `*.txt.json` of the data folder, i.e. the training file
    a second time (another ~10 GB on Wikidata5M). `get_neighbors` returns the union of both parts."""

    def __init__(self, base: TripletDict, extra: TripletDict) -> None:
        self.base = base
        self.extra = extra

    def get_neighbors(self, head_id: str, relation: str):
        a = self.base.get_neighbors(head_id, relation)
        b = self.extra.get_neighbors(head_id, relation)
        return (a | b) if b else a


def _init_entity_dict() -> None:
    global _entity_dict
    if _entity_dict is None:
        _entity_dict = EntityDict(
            entity_dict_dir=os.path.dirname(args.valid_path)
        )


def _init_train_triplet_dict() -> None:
    global _train_triplet_dict
    if _train_triplet_dict is None:
        _train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict() -> None:
    global _all_triplet_dict
    if _all_triplet_dict is None:
        path_pattern = os.path.join(
            os.path.dirname(args.train_path), '*.txt.json'
        )
        train_abs = os.path.abspath(args.train_path)
        extra_paths = [p for p in glob.glob(path_pattern) if os.path.abspath(p) != train_abs]
        _all_triplet_dict = _OverlayTripletDict(
            get_train_triplet_dict(), TripletDict(path_list=extra_paths)
        )


def _init_link_graph() -> None:
    global _link_graph
    if _link_graph is None:
        _link_graph = LinkGraph(train_path=args.train_path)


def init_tokenizer(arguments) -> None:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(arguments.pretrained_model)
        logger.info(f'Build tokenizer from {arguments.pretrained_model}')


def get_entity_dict() -> EntityDict:
    _init_entity_dict()
    return _entity_dict


def get_train_triplet_dict() -> TripletDict:
    _init_train_triplet_dict()
    return _train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return _all_triplet_dict


def get_link_graph() -> LinkGraph:
    _init_link_graph()
    return _link_graph


def get_tokenizer():
    if _tokenizer is None:
        init_tokenizer(args)
    return _tokenizer