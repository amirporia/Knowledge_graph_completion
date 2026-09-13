from .transe import TransE
from .distmult import DistMult
from .complex import ComplEx
from .rotate import RotatE
from .quate import QuatE
from .qiqekgc import QIQEKGC
from .conve import ConvE
from .rgcn import RGCN

MODEL_REGISTRY = {
    'transe': TransE,
    'distmult': DistMult,
    'complex': ComplEx,
    'rotate': RotatE,
    'quate': QuatE,
    'qiqekgc': QIQEKGC,
    'conve': ConvE,
    'rgcn': RGCN,
}


def build_model(name: str, num_entities: int, num_relations: int, args):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model {name}. Choices: {list(MODEL_REGISTRY.keys())}')
    return MODEL_REGISTRY[name](num_entities, num_relations, args)
