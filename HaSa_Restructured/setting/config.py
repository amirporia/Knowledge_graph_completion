import argparse
import os
import random
import warnings
from pathlib import Path

import torch.backends.cudnn as cudnn

# Project root directory (the folder that contains this package and `data/`)
SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()

# Current task name
CURRENT_TASK_NAME = "wn18rr"

# All methods share the same data / dataloader / evaluation pipeline.
#   simkgc           : SimKGC baseline (two BERT encoders, cosine InfoNCE)  -- identical to SimKGC_Restructured
#   hasa             : HaSa (hard + false negatives, debiased hardness-aware InfoNCE)
#   hasa_hard_bias   : Hard InfoNCE   (hardness-aware weights, no false-negative term)
#   hasa_wohard_bias : simple InfoNCE (random extra negatives, no hardness weights)
METHODS = ['simkgc', 'hasa', 'hasa_hard_bias', 'hasa_wohard_bias']


def uses_hasa(method: str) -> bool:
    return method.startswith('hasa')


def parse_args():
    """Parse command line arguments (SimKGC defaults are kept; HaSa options are additive)."""
    parser = argparse.ArgumentParser(description='SimKGC / HaSa arguments')

    # Method
    method_group = parser.add_argument_group('Method')
    method_group.add_argument('--method', default='simkgc', type=str.lower, choices=METHODS,
                              help='Which model/loss to train and evaluate')

    # Model settings
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--pretrained-model', default='bert-base-uncased', type=str,
                             help='Path to pretrained model')
    model_group.add_argument('--pooling', default='cls', type=str, choices=['cls', 'mean', 'max', 'pooler'],
                             help="Pooling method ('pooler' = HF pooler_output, used by the original HaSa)")
    model_group.add_argument('--dropout', default=0.1, type=float,
                             help='Dropout rate on final linear layer')
    model_group.add_argument('--max-num-tokens', default=50, type=int,
                             help='Maximum number of tokens')
    model_group.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=None,
                             help='Default: on for simkgc (as in SimKGC), off for HaSa (as in HaSa)')
    model_group.add_argument('--entity-text', default='name_desc', type=str, choices=['name_desc', 'name'],
                             help='Entity input text: "name: description" (SimKGC) or name only')

    # HaSa settings
    hasa_group = parser.add_argument_group('HaSa')
    hasa_group.add_argument('--em-dim', default=500, type=int,
                            help='Embedding dim of the HaSa projection head')
    hasa_group.add_argument('--num-hard-neg', default=3, type=int,
                            help='Hard negatives mined per example')
    hasa_group.add_argument('--num-false-neg', default=3, type=int,
                            help='Likely false negatives (graph neighbours) per example (method=hasa)')
    hasa_group.add_argument('--debias-tau', default=1e-4, type=float,
                            help='Class prior tau of the debiased InfoNCE (original HaSa: 1e-4 FB, 2e-5 WN)')
    hasa_group.add_argument('--hasa-margin', default=1.0, type=float,
                            help='Margin subtracted from positive logits (original HaSa: 1.0)')
    hasa_group.add_argument('--plus', action='store_true',
                            help='HaSa+: add the symmetric tail->(h,r) cross-entropy term')
    hasa_group.add_argument('--neg-count-mode', default='dim', type=str, choices=['dim', 'count'],
                            help='"dim": N in the debiased loss is the embedding dim (what the original code '
                                 'computes); "count": the real number of negatives')

    # Data settings
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--task', default=CURRENT_TASK_NAME, type=str.lower,
                            choices=['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans'],
                            help='Dataset name')
    data_group.add_argument('--train-path', default=None, type=str,
                            help='Path to training data (auto-generated from task if not specified)')
    data_group.add_argument('--valid-path', default=None, type=str,
                            help='Path to validation data (auto-generated from task if not specified)')

    # Training settings
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', default=20, type=int,
                             help='Number of total epochs to run')
    train_group.add_argument('-b', '--batch-size', default=32, type=int,
                             help='Mini-batch size (per process when running with torchrun)')
    train_group.add_argument('--lr', '--learning-rate', default=2e-5, type=float, dest='lr',
                             help='Initial learning rate')
    train_group.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay',
                             help='Weight decay')
    train_group.add_argument('--lr-scheduler', default='linear', type=str, choices=['linear', 'cosine', 'constant'],
                             help='Learning rate scheduler')
    train_group.add_argument('--warmup', default=400, type=int,
                             help='Number of warmup steps')
    train_group.add_argument('--grad-clip', default=10.0, type=float,
                             help='Gradient clipping value')
    train_group.add_argument('--seed', default=None, type=int,
                             help='Seed for training initialization')

    # Model management
    mgmt_group = parser.add_argument_group('Model Management')
    mgmt_group.add_argument('--model-dir', default=None, type=str,
                            help='Path to save model checkpoints (auto-generated from task if not specified)')
    mgmt_group.add_argument('--max-to-keep', default=5, type=int,
                            help='Max number of checkpoints to keep')
    mgmt_group.add_argument('--eval-every-n-step', default=10000, type=int,
                            help='Evaluate every n steps (0 disables mid-epoch evaluation)')
    mgmt_group.add_argument('--eval-model-path', default=None, type=str,
                            help='Path to model for evaluation (auto-generated from task if not specified)')
    mgmt_group.add_argument('--resume', action='store_true',
                            help='Resume training from a checkpoint (model, optimizer, scheduler, epoch)')
    mgmt_group.add_argument('--resume-path', default=None, type=str,
                            help='Checkpoint to resume from (default: <model-dir>/model_last.mdl)')

    # Loss and optimization (SimKGC)
    loss_group = parser.add_argument_group('Loss and Optimization (SimKGC)')
    loss_group.add_argument('--t', default=0.05, type=float,
                            help='Temperature parameter')
    loss_group.add_argument('--additive-margin', default=0.0, type=float,
                            help='Additive margin for InfoNCE loss')
    loss_group.add_argument('--finetune-t', action='store_true',
                            help='Make temperature a trainable parameter')
    loss_group.add_argument('--pre-batch', default=0, type=int,
                            help='Number of pre-batch used for negatives')
    loss_group.add_argument('--pre-batch-weight', default=0.5, type=float,
                            help='Weight for logits from pre-batch negatives')
    loss_group.add_argument('--use-self-negative', action=argparse.BooleanOptionalAction, default=True,
                            help='Use head entity as negative sample (disable with --no-use-self-negative)')

    # Graph and context settings
    graph_group = parser.add_argument_group('Graph')
    graph_group.add_argument('--use-link-graph', action='store_true',
                             help='Use neighbors from link graph as context')
    graph_group.add_argument('--rerank-n-hop', default=2, type=int,
                             help='Use n-hop nodes for re-ranking entities (evaluation only)')
    graph_group.add_argument('--neighbor-weight', default=0.05, type=float,
                             help='Weight for re-ranking entities (evaluation only; use 0 for HaSa, '
                                  'its scores are not cosine-bounded)')

    # System settings
    system_group = parser.add_argument_group('System')
    system_group.add_argument('-j', '--workers', default=4, type=int,
                              help='Number of data loading workers')
    system_group.add_argument('-p', '--print-freq', default=50, type=int,
                              help='Print frequency')
    system_group.add_argument('--use-amp', action='store_true',
                              help='Use automatic mixed precision if available')
    system_group.add_argument('--gpu', default=0, type=int,
                              help='GPU index to use when not running distributed (ignored under torchrun)')
    system_group.add_argument('--dist-backend', default='nccl', type=str, choices=['nccl', 'gloo'],
                              help='torch.distributed backend for multi-GPU training')

    # Evaluation flags
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--is-test', action='store_true', default=False,
                            help='Run in test mode')

    return parser.parse_args()


def generate_paths_from_task(arguments):
    """Generate dynamic paths based on task name."""
    task = arguments.task.lower()

    if arguments.train_path is None:
        arguments.train_path = str(SCRIPT_DIR / 'data' / task / 'train.txt.json')

    if arguments.valid_path is None:
        arguments.valid_path = str(SCRIPT_DIR / 'data' / task / 'valid.txt.json')

    if arguments.model_dir is None:
        arguments.model_dir = str(SCRIPT_DIR / 'data' / task / 'checkpoint')

    if arguments.eval_model_path is None:
        arguments.eval_model_path = str(SCRIPT_DIR / 'data' / task / 'checkpoint' / 'model_best.mdl')

    return arguments


def validate_args(arguments):
    """Validate parsed arguments."""
    arguments = generate_paths_from_task(arguments)

    if arguments.train_path and not os.path.exists(arguments.train_path):
        raise FileNotFoundError(f"Training data not found: {arguments.train_path}")
    if arguments.valid_path and not os.path.exists(arguments.valid_path):
        raise FileNotFoundError(f"Validation data not found: {arguments.valid_path}")

    if uses_hasa(arguments.method) and arguments.task.startswith('wiki5m'):
        raise ValueError('HaSa keeps an embedding bank of all entities and is not meant for wiki5m; '
                         'use --method simkgc there.')

    if arguments.gradient_checkpointing is None:
        arguments.gradient_checkpointing = (arguments.method == 'simkgc')

    os.makedirs(arguments.model_dir, exist_ok=True)

    if arguments.resume:
        if arguments.resume_path is None:
            arguments.resume_path = os.path.join(arguments.model_dir, 'model_last.mdl')
        if not os.path.exists(arguments.resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {arguments.resume_path}")

    return arguments


def setup_environment(arguments):
    """Setup random seeds and check system capabilities."""
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
    """Detect single-node multi-GPU (DDP) settings from torchrun's environment variables."""
    arguments.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    arguments.rank = int(os.environ.get('RANK', '0'))
    arguments.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    arguments.distributed = arguments.world_size > 1
    return arguments


"""entry point for argument parsing and setup."""
args = parse_args()
args = validate_args(args)
args = setup_environment(args)
args = setup_distributed(args)
