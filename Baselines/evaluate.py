import argparse
import json
import logging
import os

import torch

from Baselines.common.data import build_kg_index, load_triples, build_true_tail_filter
from Baselines.common.metrics import evaluate_mrr
from Baselines.common.utils import load_checkpoint
from Baselines.embedding_models.factory import build_model
from Baselines.common.data import build_message_passing_graph

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=[
        'transe', 'distmult', 'complex', 'rotate', 'quate', 'conve', 'rgcn'])
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--split', default='test', choices=['valid', 'test'])
    parser.add_argument('--eval-batch-size', default=64, type=int)
    cli_args = parser.parse_args()

    ckpt = load_checkpoint(cli_args.checkpoint,
                           map_location='cuda' if torch.cuda.is_available() else 'cpu')
    train_args = argparse.Namespace(**ckpt['args'])
    train_args.eval_batch_size = cli_args.eval_batch_size

    entities_path = os.path.join(cli_args.data_dir, 'entities.json')
    train_path = os.path.join(cli_args.data_dir, 'train.txt.json')
    valid_path = os.path.join(cli_args.data_dir, 'valid.txt.json')
    test_path = os.path.join(cli_args.data_dir, 'test.txt.json')
    triple_paths = [p for p in (train_path, valid_path, test_path) if os.path.exists(p)]

    kg_index = build_kg_index(entities_path, triple_paths)
    train_triples = load_triples(train_path, kg_index, add_inverse=True)
    valid_triples = load_triples(valid_path, kg_index, add_inverse=True)
    test_triples = load_triples(test_path, kg_index, add_inverse=True) if os.path.exists(test_path) else None
    true_tail_filter = build_true_tail_filter(
        *[t for t in (train_triples, valid_triples, test_triples) if t is not None])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cli_args.model, kg_index.num_entities, kg_index.num_relations, train_args).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    if getattr(model, 'requires_graph_encode', False):
        edge_index, edge_type, edge_norm = build_message_passing_graph(
            train_triples, kg_index.num_entities, kg_index.num_relations)
        with torch.no_grad():
            model.encode_graph(edge_index.to(device), edge_type.to(device), edge_norm.to(device))

    eval_triples = test_triples if cli_args.split == 'test' else valid_triples
    metrics = evaluate_mrr(model, eval_triples, true_tail_filter, kg_index.num_entities,
                           batch_size=cli_args.eval_batch_size, device=device, desc=cli_args.split)
    logger.info(f'{cli_args.model} {cli_args.split} metrics: {json.dumps(metrics, indent=2)}')


if __name__ == '__main__':
    main()
