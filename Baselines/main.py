import json
import logging
import os

from Baselines.config import parse_args
from Baselines.common.data import build_kg_index, load_triples, build_true_tail_filter
from Baselines.common.trainer import EmbeddingTrainer
from Baselines.common.utils import set_seed
from Baselines.embedding_models.factory import build_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    logger.info(f'Args: {json.dumps(vars(args), indent=2)}')

    entities_path = os.path.join(args.data_dir, 'entities.json')
    train_path = os.path.join(args.data_dir, 'train.txt.json')
    valid_path = os.path.join(args.data_dir, 'valid.txt.json')
    test_path = os.path.join(args.data_dir, 'test.txt.json')
    # test.txt.json may not exist for every task layout; fall back to valid for the
    # "all known triples" filter set used at eval time.
    triple_paths = [p for p in (train_path, valid_path, test_path) if os.path.exists(p)]

    kg_index = build_kg_index(entities_path, triple_paths)
    train_triples = load_triples(train_path, kg_index, add_inverse=True)
    valid_triples = load_triples(valid_path, kg_index, add_inverse=True)
    test_triples = load_triples(test_path, kg_index, add_inverse=True) if os.path.exists(test_path) else None

    all_splits = [train_triples, valid_triples] + ([test_triples] if test_triples is not None else [])
    true_tail_filter = build_true_tail_filter(*all_splits)

    model = build_model(args.model, kg_index.num_entities, kg_index.num_relations, args)
    logger.info(f'{args.model}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} '
                f'trainable parameters')

    trainer = EmbeddingTrainer(model, kg_index, train_triples, valid_triples, true_tail_filter, args)
    best_valid_metrics = trainer.fit()
    logger.info(f'Best validation metrics ({args.model}, {args.task}): {best_valid_metrics}')

    if test_triples is not None:
        from Baselines.common.utils import load_checkpoint
        ckpt = load_checkpoint(os.path.join(args.model_dir, 'model_best.mdl'),
                                map_location=trainer.device)
        model.load_state_dict(ckpt['state_dict'])
        test_metrics = trainer.evaluate(test_triples, desc='test')
        logger.info(f'Test metrics ({args.model}, {args.task}): {test_metrics}')
        with open(os.path.join(args.model_dir, f'test_metrics_{args.model}.json'), 'w') as f:
            json.dump({'valid': best_valid_metrics, 'test': test_metrics}, f, indent=2)


if __name__ == '__main__':
    main()
