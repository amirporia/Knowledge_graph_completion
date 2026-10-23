import argparse
import os
import random
import warnings
from pathlib import Path

import torch.backends.cudnn as cudnn

SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()
CURRENT_TASK_NAME = "wn18rr"

# (c_p, c_s): weights of S_p and S_struct in  S = S_q + c_p S_p + c_s S_struct.
# Used whenever a coefficient is not given on the command line and no best_coefs.json is found.
# c_p defaults to 1.0 because S_p now takes over the role the anchor-enhanced score S_hrta had (c_h = 1).
DEFAULT_COEFS = (1.0, 0.1)


def _add_toggle(group, name: str, default: bool, help_text: str) -> None:
    """Register `--name` / `--no-name` flags sharing one boolean destination."""
    dest = name.replace('-', '_')
    group.add_argument(f'--{name}', dest=dest, action='store_true', default=default, help=help_text)
    group.add_argument(f'--no-{name}', dest=dest, action='store_false', default=default,
                       help=f'Disable: {help_text}')


def parse_args():
    parser = argparse.ArgumentParser(description='ARPM-KGC v3 arguments')

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
    g.add_argument('--memory-lr', default=1e-4, type=float, dest='memory_lr',
                   help='LR of the small memory modules (attention, prototypes, hop scorer). '
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
    g.add_argument('--t-mem', default=0.05, type=float, dest='t_mem',
                   help='Initial temperature of the memory losses L_proto / L_struct '
                        '(separate from --t, which only scales L_query)')
    _add_toggle(g, 'finetune-t-mem', True,
                'Make the memory-loss temperatures (proto, struct) trainable parameters')
    # same default as the baseline (temperature is learnable, no clamping)
    _add_toggle(g, 'finetune-t', True, 'Make temperature a trainable parameter')

    # Graph
    g = parser.add_argument_group('Graph')
    g.add_argument('--disable-link-graph', dest='use_link_graph', action='store_false', default=True,
                   help='Disable the link graph: no neighbour text augmentation and no hop>=1 anchors '
                        '(hop-0 candidates, which need no graph traversal, are kept).')
    g.add_argument('--rerank-n-hop', default=5, type=int, dest='rerank_n_hop')
    g.add_argument('--neighbor-weight', default=0.05, type=float, dest='neighbor_weight')

    # Memory
    g = parser.add_argument_group('ARPM-KGC Memory')
    g.add_argument('--num-hops', default=2, type=int, dest='num_hops')
    g.add_argument('--num-prototypes', default=4, type=int, dest='num_prototypes')
    g.add_argument('--local-per-hop-budget', default=5, type=int, dest='local_per_hop_budget')
    g.add_argument('--global-budget', default=5, type=int, dest='global_budget')
    g.add_argument('--anchor-budget', default=20, type=int, dest='anchor_budget')
    g.add_argument('--retrieval-temperature', default=0.1, type=float, dest='retrieval_temperature')
    g.add_argument('--proto-temperature', default=0.1, type=float, dest='proto_temperature')
    g.add_argument('--proto-attn-temperature', default=0.5, type=float, dest='proto_attn_temperature')
    g.add_argument('--eps-struct', default=1e-6, type=float, dest='eps_struct')

    g.add_argument('--eta-proto', default=0.2, type=float, dest='eta_proto')
    g.add_argument('--eta-struct', default=0.1, type=float, dest='eta_struct')
    g.add_argument('--eta-div', default=0.0, type=float, dest='eta_div')
    g.add_argument('--eta-pdiv', default=0.01, type=float, dest='eta_pdiv',
                   help='Weight of the prototype-specialisation loss (cosine between prototypes)')
    _add_toggle(g, 'use-self-negative', True, 'Use head entity as an additional negative for S_q')

    # Gradient policy of the semantic score S_p (S_struct is always fully detached from the encoders)
    g = parser.add_argument_group('Semantic-memory gradient policy (S_p)')
    _add_toggle(g, 'memory-grad-query', True,
                'Do not detach q in the semantic branch: L_proto also trains the query encoder E_0')
    _add_toggle(g, 'memory-grad-tail', True,
                'Do not detach the in-batch tail vectors in L_proto: L_proto also trains E_1')
    _add_toggle(g, 'memory-grad-cand', True,
                'Let candidate embeddings E_1(t_i) carry gradient from L_proto/L_div (extra memory/compute)')

    # Score coefficients (constants, no learned gate)
    g = parser.add_argument_group('Score coefficients: S = S_q + c_p*S_p + c_s*S_struct')
    g.add_argument('--coef-p', default=None, type=float, dest='coef_p',
                   help=f'c_p: weight of the prototype score S_p (default {DEFAULT_COEFS[0]}; '
                        'overridden by best_coefs.json at evaluation if present and no coef is given)')
    g.add_argument('--coef-s', default=None, type=float, dest='coef_s',
                   help=f'c_s: weight of the structural score S_struct (default {DEFAULT_COEFS[1]})')
    g.add_argument('--tune-coefs', action='store_true', dest='tune_coefs',
                   help='Grid-search (c_p, c_s) on --valid-path by average MRR and save '
                        'best_coefs.json next to the checkpoint. Always run this on the VALIDATION '
                        'split; then evaluate the test split without --tune-coefs.')
    g.add_argument('--coef-p-grid', nargs='+', type=float, default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0], dest='coef_p_grid')
    g.add_argument('--coef-s-grid', nargs='+', type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0], dest='coef_s_grid')

    # Ablations
    g = parser.add_argument_group('Ablation Overrides')
    g.add_argument('--random-anchor-selection', action='store_true', dest='random_anchor_selection')
    g.add_argument('--uniform-hop-weighting', action='store_true', dest='uniform_hop_weighting')

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

    return parser.parse_args()


def get_coefs(arguments):
    """(c_p, c_s) with unset values replaced by DEFAULT_COEFS."""
    given = (arguments.coef_p, arguments.coef_s)
    return tuple(d if v is None else float(v) for v, d in zip(given, DEFAULT_COEFS))


def generate_paths_from_task(arguments):
    task = arguments.task.lower()
    if arguments.train_path is None:
        arguments.train_path = str(SCRIPT_DIR / 'data' / task / 'train.txt.json')
    if arguments.valid_path is None:
        arguments.valid_path = str(SCRIPT_DIR / 'data' / task / 'valid.txt.json')
    if arguments.model_dir is None:
        arguments.model_dir = str(SCRIPT_DIR / 'data' / task / 'checkpoint_arpm_v3')
    if arguments.eval_model_path is None:
        arguments.eval_model_path = str(SCRIPT_DIR / 'data' / task / 'checkpoint_arpm_v3' / 'model_best.mdl')
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