"""
Lazily-built, process-wide singletons for the entity dictionary, triplet
indices, link graph, relation vocabulary, and tokenizer.

Mirrors the "load once, reuse everywhere" pattern common to KGC codebases,
but implemented independently (no import from the baseline `ours` package)
and parameterized explicitly by `ARPMConfig` rather than a global config
object, so multiple configs can coexist cleanly (e.g. in tests).
"""

import glob
import os
from typing import Optional

from .entities import EntityDict
from .graph import LinkGraph
from .triplets import RelationVocab, TripletDict
from ..logging_utils import logger

_entity_dict: Optional[EntityDict] = None
_train_triplet_dict: Optional[TripletDict] = None
_all_triplet_dict: Optional[TripletDict] = None
_link_graph: Optional[LinkGraph] = None
_relation_vocab: Optional[RelationVocab] = None
_tokenizer = None


def get_entity_dict(config) -> EntityDict:
    global _entity_dict
    if _entity_dict is None:
        entity_dir = os.path.dirname(config.valid_path)
        _entity_dict = EntityDict(entity_dict_dir=entity_dir)
        logger.info(f"Loaded {len(_entity_dict)} entities from {entity_dir}")
    return _entity_dict


def get_train_triplet_dict(config) -> TripletDict:
    global _train_triplet_dict
    if _train_triplet_dict is None:
        _train_triplet_dict = TripletDict(path_list=[config.train_path])
        logger.info(f"Train triplet dict: {_train_triplet_dict.triplet_cnt} directed triples, "
                    f"{len(_train_triplet_dict.relations)} relations")
    return _train_triplet_dict


def get_all_triplet_dict(config) -> TripletDict:
    """Union of train/valid/test triples, used only for the *filtered*
    ranking protocol at evaluation time (Sec 7.2) -- never for candidate
    retrieval or memory construction during training (that would leak
    validation/test triples into the training-time candidate pool)."""
    global _all_triplet_dict
    if _all_triplet_dict is None:
        pattern = os.path.join(os.path.dirname(config.train_path), "*.txt.json")
        paths = sorted(glob.glob(pattern))
        if not paths:
            paths = [config.train_path, config.valid_path, config.test_path]
        _all_triplet_dict = TripletDict(path_list=paths)
        logger.info(f"All-split triplet dict built from {paths}")
    return _all_triplet_dict


def get_link_graph(config) -> LinkGraph:
    global _link_graph
    if _link_graph is None:
        _link_graph = LinkGraph(train_path=config.train_path)
        logger.info(f"Link graph built with {len(_link_graph.graph)} nodes "
                    f"(train split only -- Sec 10.6 structural-leakage constraint)")
    return _link_graph


def get_relation_vocab(config) -> RelationVocab:
    global _relation_vocab
    if _relation_vocab is None:
        relations_path = os.path.join(os.path.dirname(config.train_path), "relations.json")
        if os.path.exists(relations_path):
            _relation_vocab = RelationVocab.load(relations_path)
        else:
            _relation_vocab = RelationVocab.from_triplet_dict(get_train_triplet_dict(config))
        logger.info(f"Relation vocabulary size: {len(_relation_vocab)}")
    return _relation_vocab


def init_tokenizer(config) -> None:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        logger.info(f"Built tokenizer from {config.pretrained_model}")


def get_tokenizer(config=None):
    if _tokenizer is None:
        assert config is not None, "Tokenizer not yet initialized; pass a config the first time."
        init_tokenizer(config)
    return _tokenizer


def reset_singletons() -> None:
    """Test helper: clears all module-level singletons."""
    global _entity_dict, _train_triplet_dict, _all_triplet_dict, _link_graph, _relation_vocab, _tokenizer
    _entity_dict = None
    _train_triplet_dict = None
    _all_triplet_dict = None
    _link_graph = None
    _relation_vocab = None
    _tokenizer = None
