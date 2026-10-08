import argparse
import os
import random
import warnings
from pathlib import Path

import torch.backends.cudnn as cudnn

SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()
CURRENT_TASK_NAME = "wn18rr"


def _add_toggle(group, name: str, default: bool, help_text: str) -> None:
    """Register `--name` / `--no-name` flags sharing one boolean destination."""
    dest = name.replace('-', '_')
    group.add_argument(f'--{name}', dest=dest, action='store_true', default=default, help=help_text)
    group.add_argument(f'--no-{name}', dest=dest, action='store_false', default=default,
                       help=f'Disable: {help_text}')


def parse_args():
    parser = argparse.ArgumentParser(description='ARPM-KGC v2 arguments')

    # Model
    g = parser.add_argument_group('Model')
    g.add_argument('--pretrained-model', default='bert-base-uncased', type=str)
    g.add_argument('--pooling', default='mean', type=str, choices=['cls', 'mean', 'max'])
    g.add_argument('--dropout', default=0.1, type=float)
    g.add_argument('--max-num-tokens', default=50, type=int)

    # Data
    g = parser.add_argument_group('Data')
    g.add_argument('--task', default=CURRENT_TASK_NAME, type=str,
                   choices=['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans'])
    g.add_argument('--train-path', default=None, type=str)
    g.add_argument('--valid-path', default=None, type=str)

    # Training
    g = parser.add_argument_group('Training')
    g.add_argument('--epochs', default=20, type=int)
    g.add_argument('-b', '--batch-size', default=32, type=int)
    g.add_argument('--lr', '--learning-rate', default=5e-5, type=float, dest='lr',
                   help='LR of the two BERT encoders (and the temperature)')
    g.add_argument('--memory-lr', default=1e-3, type=float, dest='memory_lr',
                   help='LR of the small memory modules (attention, prototypes, hop scorer, gate). '
                        'They are trained from scratch, so 5e-5 is far too small for them.')
    g.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
    g.add_argument('--lr-scheduler', default='linear', type=str, choices=['linear', 'cosine'])
    g.add_argument('--warmup', default=400, type=int)
    g.add_argument('--grad-clip', default=10.0, type=float)
    g.add_argument('--seed', default=None, type=int)

    # Model management
    g = parser.add_argument_group('Model Management')
    g.add_argument('--model-dir', default=None, type=str)
    g.add_argument('--max-to-keep', default=4, type=int)
    g.add_argument('--eval-every-n-step', default=0, type=int)
    g.add_argument('--eval-model-path', default=None, type=str)
    g.add_argument('--resume', action='store_true')
    g.add_argument('--resume-path', default=None, type=str)
    g.add_argument('--checkpoint-metric', default='mrr', type=str,
                   choices=['mrr', 'hit@1', 'hit@3', 'hit@10', 'hit@50'])
    g.add_argument('--full-eval-every-n-epochs', default=1, type=int)
    g.add_argument('--full-eval-batch-size', default=8, type=int)

    # Loss
    g = parser.add_argument_group('Loss and Optimization')
    g.add_argument('--t', default=0.05, type=float)
    g.add_argument('--additive-margin', default=0.02, type=float)
    # same default as the baseline (temperature is learnable, no clamping)
    _add_toggle(g, 'finetune-t', True, 'Make temperature a trainable parameter')
    g.add_argument('--alpha', default=0.2, type=float,
                   help='Weight of L_hrta (anchor-enhanced query loss, same as RAA-KGC)')
    g.add_argument('--eta-combined', default=0.1, type=float, dest='eta_combined')

    # Graph
    g = parser.add_argument_group('Graph')
    g.add_argument('--disable-link-graph', dest='use_link_graph', action='store_false', default=True,
                   help='Disable the link graph: no neighbour text augmentation and no hop>=1 anchors '
                        '(hop-0 anchors, which need no graph traversal, are kept).')
    g.add_argument('--rerank-n-hop', default=5, type=int, dest='rerank_n_hop')
    g.add_argument('--neighbor-weight', default=0.05, type=float, dest='neighbor_weight')

    # Memory
    g = parser.add_argument_group('ARPM-KGC Memory')
    g.add_argument('--anchor-num', default=4, type=int, dest='anchor_num',
                   help='Number of hop-0 anchors used by the RAA-KGC anchor-enhanced query path '
                        '(0 disables it; the model then reduces to the S_q + memory design)')
    g.add_argument('--num-hops', default=5, type=int, dest='num_hops')
    g.add_argument('--num-prototypes', default=4, type=int, dest='num_prototypes')
    g.add_argument('--local-per-hop-budget', default=5, type=int, dest='local_per_hop_budget')
    g.add_argument('--global-budget', default=5, type=int, dest='global_budget')
    g.add_argument('--anchor-budget', default=20, type=int, dest='anchor_budget')
    g.add_argument('--retrieval-temperature', default=0.1, type=float, dest='retrieval_temperature')
    g.add_argument('--proto-temperature', default=0.1, type=float, dest='proto_temperature')
    g.add_argument('--proto-attn-temperature', default=0.5, type=float, dest='proto_attn_temperature')
    g.add_argument('--eps-struct', default=1e-6, type=float, dest='eps_struct')
    g.add_argument('--gate-init-bias', default=-2.0, type=float, dest='gate_init_bias',
                   help='Initial bias of the memory gate (sigmoid(-2)=0.12: memory starts as a small '
                        'perturbation of the baseline score)')
    g.add_argument('--eta-proto', default=0.1, type=float, dest='eta_proto')
    g.add_argument('--eta-struct', default=0.1, type=float, dest='eta_struct')
    g.add_argument('--eta-div', default=0.0, type=float, dest='eta_div')
    g.add_argument('--eta-pdiv', default=0.01, type=float, dest='eta_pdiv',
                   help='Weight of the prototype-specialisation loss (cosine between prototypes)')
    _add_toggle(g, 'use-self-negative', True, 'Use head entity as an additional negative for S_q')

    # Ablations
    g = parser.add_argument_group('Ablation Overrides')
    g.add_argument('--random-anchor-selection', action='store_true', dest='random_anchor_selection')
    g.add_argument('--uniform-hop-weighting', action='store_true', dest='uniform_hop_weighting')
    g.add_argument('--fixed-lambda-p', default=None, type=float, dest='fixed_lambda_p')
    g.add_argument('--fixed-lambda-s', default=None, type=float, dest='fixed_lambda_s')

    # System
    g = parser.add_argument_group('System')
    g.add_argument('-j', '--workers', default=4, type=int)
    g.add_argument('-p', '--print-freq', default=20, type=int)
    g.add_argument('--use-amp', action='store_true')
    g.add_argument('--gpu', default=0, type=int)
    g.add_argument('--dist-backend', default='nccl', type=str, choices=['nccl', 'gloo'])

    # Evaluation
    g = parser.add_argument_group('Evaluation')
    g.add_argument('--is-test', action='store_true', default=False)
    g.add_argument('--scale-p', default=None, type=float, dest='scale_p',
                   help='Global post-hoc multiplier on lambda_p at evaluation (default: best_scales.json '
                        'if present, else 1.0)')
    g.add_argument('--scale-s', default=None, type=float, dest='scale_s',
                   help='Global post-hoc multiplier on lambda_s at evaluation')
    g.add_argument('--tune-scales', action='store_true', dest='tune_scales',
                   help='Grid-search (scale_p, scale_s) on --valid-path and save best_scales.json. '
                        '(0,0) is in the grid, so tuning can always fall back to the RAA baseline score.')

    return parser.parse_args()


def generate_paths_from_task(arguments):
    task = arguments.task.lower()
    if arguments.train_path is None:
        arguments.train_path = str(SCRIPT_DIR / 'data' / task / 'train.txt.json')
    if arguments.valid_path is None:
        arguments.valid_path = str(SCRIPT_DIR / 'data' / task / 'valid.txt.json')
    if arguments.model_dir is None:
        arguments.model_dir = str(SCRIPT_DIR / 'data' / task / 'checkpoint_arpm_v2')
    if arguments.eval_model_path is None:
        arguments.eval_model_path = str(SCRIPT_DIR / 'data' / task / 'checkpoint_arpm_v2' / 'model_best.mdl')
    return arguments


def validate_args(arguments):
    arguments = generate_paths_from_task(arguments)

    if arguments.train_path and not os.path.exists(arguments.train_path):
        raise FileNotFoundError(f"Training data not found: {arguments.train_path}")
    if arguments.valid_path and not os.path.exists(arguments.valid_path):
        raise FileNotFoundError(f"Validation data not found: {arguments.valid_path}")

    if arguments.model_dir:
        os.makedirs(arguments.model_dir, exist_ok=True)
    elif os.path.exists(arguments.eval_model_path):
        arguments.model_dir = os.path.dirname(arguments.eval_model_path)
    else:
        raise ValueError('Either --model-dir or a valid --eval-model-path must be provided')

    if arguments.resume:
        if arguments.resume_path is None:
            arguments.resume_path = os.path.join(arguments.model_dir, 'model_last.mdl')
        if not os.path.exists(arguments.resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {arguments.resume_path}")

    if arguments.anchor_num < 0 or arguments.alpha < 0:
        raise ValueError('--anchor-num and --alpha must be non-negative')
    if arguments.num_prototypes not in (1, 2, 4, 8):
        warnings.warn(f'--num-prototypes={arguments.num_prototypes} is outside {{1,2,4,8}}')
    return arguments


def setup_environment(arguments):
    import torch
    if arguments.seed is not None:
        random.seed(arguments.seed)
        torch.manual_seed(arguments.seed)
        cudnn.deterministic = True

    if arguments.use_amp:
        try:
            import torch.cuda.amp  # noqa: F401
        except ImportError:
            arguments.use_amp = False
            warnings.warn('AMP training is not available, set use_amp=False')

    if not torch.cuda.is_available():
        arguments.use_amp = False
        arguments.print_freq = 1
        warnings.warn('GPU is not available, set use_amp=False and print_freq=1')
    return arguments


def setup_distributed(arguments):
    arguments.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    arguments.rank = int(os.environ.get('RANK', '0'))
    arguments.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    arguments.distributed = arguments.world_size > 1
    return arguments


args = parse_args()
args = validate_args(args)
args = setup_environment(args)
args = setup_distributed(args)
