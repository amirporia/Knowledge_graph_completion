import argparse
import os
import random
import warnings
from pathlib import Path

import torch.backends.cudnn as cudnn

# Define the project root directory dynamically
SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()

# Current task name
CURRENT_TASK_NAME = "wn18rr"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ARPM-KGC arguments')

    # Model settings
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--pretrained-model', default='bert-base-uncased', type=str,
                             help='Path to pretrained model')
    model_group.add_argument('--pooling', default='mean', type=str, choices=['cls', 'mean', 'max'],
                             help='BERT pooling method')
    model_group.add_argument('--dropout', default=0.1, type=float,
                             help='Dropout rate on final linear layer')
    model_group.add_argument('--max-num-tokens', default=50, type=int,
                             help='Maximum number of tokens')

    # Data settings
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--task', default=CURRENT_TASK_NAME, type=str,
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
                             help='Mini-batch size')
    train_group.add_argument('--lr', '--learning-rate', default=5e-5, type=float, dest='lr',
                             help='Initial learning rate')
    train_group.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay',
                             help='Weight decay')
    train_group.add_argument('--lr-scheduler', default='linear', type=str, choices=['linear', 'cosine'],
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
    mgmt_group.add_argument('--max-to-keep', default=4, type=int,
                            help='Max number of checkpoints to keep')
    mgmt_group.add_argument('--eval-every-n-step', default=10000, type=int,
                            help='Evaluate every n steps')
    mgmt_group.add_argument('--eval-model-path', default=None, type=str,
                            help='Path to model for evaluation (auto-generated from task if not specified)')
    mgmt_group.add_argument('--resume', action='store_true',
                            help='Resume training from a checkpoint (model, optimizer, scheduler, epoch)')
    mgmt_group.add_argument('--resume-path', default=None, type=str,
                            help='Checkpoint to resume from (default: <model-dir>/model_last.mdl)')
    mgmt_group.add_argument('--checkpoint-metric', default='mrr', type=str,
                            choices=['mrr', 'hit@1', 'hit@3', 'hit@10', 'hit@50'],
                            help='Metric used to pick the best checkpoint. Computed via the SAME '
                                 'filtered-ranking protocol as final evaluation (forward+backward '
                                 'averaged S(t|h,r) ranking against the full entity set '
                                 '/ evaluation/evaluate.py) -- NOT the cheap in-batch accuracy that '
                                 'is also logged every epoch purely as a training diagnostic.')
    mgmt_group.add_argument('--full-eval-every-n-epochs', default=1, type=int,
                            help='Run the full filtered-ranking validation (used for checkpoint '
                                 'selection) every N epochs. Increase this for very large entity '
                                 'dictionaries (e.g. wiki5m) where encoding every entity each '
                                 'epoch is too expensive; other epochs still checkpoint '
                                 '(model_last.mdl) but never overwrite model_best.mdl.')
    mgmt_group.add_argument('--full-eval-batch-size', default=256, type=int,
                            help='Batch size used only for the full filtered-ranking validation '
                                 '(query-vs-full-entity-set scoring), independent of --batch-size')

    # Loss and optimization (shared InfoNCE machinery)
    loss_group = parser.add_argument_group('Loss and Optimization')
    loss_group.add_argument('--t', default=0.05, type=float,
                            help='Temperature parameter (used only to scale CE logits; the '
                                 'reported/ranked S_q, S_p, S_struct, S remain the raw '
                                 'paper-defined similarities, see model/trainer.py)')
    loss_group.add_argument('--additive-margin', default=0.02, type=float,
                            help='Additive margin for the S_q InfoNCE loss only')
    loss_group.add_argument('--finetune-t', action='store_true',
                            help='Make temperature a trainable parameter')

    # Graph and context settings
    graph_group = parser.add_argument_group('Graph')
    graph_group.add_argument('--disable-link-graph', dest='use_link_graph', action='store_false',
                             default=True,
                             help='By default the link graph is BUILT and USED, both for entity-'
                                  'description text augmentation and, more '
                                  'importantly for ARPM-KGC, as the source of local structural '
                                  'candidate anchors A_local(h,r) and adaptive structural memory '
                                  ', here it is core to RQ3. Pass this flag to '
                                  'disable it, which reduces the candidate pool to A_global(r) '
                                  'only and is equivalent to ablation A7 (no local structural '
                                  'memory).')

    # ------------------------------------------------------------------
    # ARPM-KGC Memory (core model)
    # ------------------------------------------------------------------
    arpm_group = parser.add_argument_group('ARPM-KGC Memory')
    arpm_group.add_argument('--num-hops', default=2, type=int, dest='num_hops',
                            help='N: max graph distance for local candidate retrieval and '
                                 'adaptive structural memory, beyond the same-head '
                                 '& relation category. Yields N+1 local hop categories total: '
                                 'hop-0 (same head & relation, graph distance 0), hop-1 '
                                 '(graph distance 1), ..., hop-N (graph distance N) -- e.g. '
                                 '--num-hops 2 gives hop-0, hop-1, AND hop-2.')
    arpm_group.add_argument('--num-prototypes', default=4, type=int, dest='num_prototypes',
                            help='K: number of semantic prototypes, one of {1,2,4,8}')
    arpm_group.add_argument('--local-per-hop-budget', default=5, type=int, dest='local_per_hop_budget',
                            help='Max local candidate anchors sampled per hop slot, including '
                                 'hop 0 (same head & relation, different tail)')
    arpm_group.add_argument('--global-budget', default=10, type=int, dest='global_budget',
                            help='Max global candidate anchors sampled from T_r before capping')
    arpm_group.add_argument('--anchor-budget', default=20, type=int, dest='anchor_budget',
                            help='M: total candidate-pool budget after merging local+global')
    arpm_group.add_argument('--retrieval-temperature', default=0.1, type=float, dest='retrieval_temperature',
                            help='tau_r: softmax temperature for query-conditioned anchor weights alpha_i')
    arpm_group.add_argument('--proto-temperature', default=0.1, type=float, dest='proto_temperature',
                            help='tau_p: log-sum-exp pooling temperature for S_p(t)')
    arpm_group.add_argument('--eps-struct', default=1e-6, type=float, dest='eps_struct',
                            help='epsilon: numerical-stability constant for empty-hop structural memory')
    arpm_group.add_argument('--eta-proto', default=0.1, type=float, dest='eta_proto',
                            help='eta_p: weight of the auxiliary prototype-memory loss L_proto')
    arpm_group.add_argument('--eta-struct', default=0.1, type=float, dest='eta_struct',
                            help='eta_s: weight of the auxiliary structural-memory loss L_struct')
    arpm_group.add_argument('--eta-div', default=0.01, type=float, dest='eta_div',
                            help='eta_div: weight of the anchor diversity regularizer L_div')
    arpm_group.add_argument('--use-self-negative', action='store_true', default=True,
                            help='Use head entity as an additional negative for S_q only')

    # ------------------------------------------------------------------
    # Ablation overrides. A1-A13 are
    # each reachable from the core model via one of these flags (or a plain
    # hyperparameter already listed above):
    #   A1  Random anchors           --random-anchor-selection
    #   A2  No diversity reg.        --eta-div 0
    #   A3  Single prototype         --num-prototypes 1
    #   A4  Fixed vs. adaptive K     compare default (fixed K) against --use-gumbel-proto (A13)
    #   A5  Fixed/uniform hop weight --uniform-hop-weighting
    #   A6  No global relation mem.  --global-budget 0
    #   A7  No local structural mem. --disable-link-graph (removes local anchors entirely,
    #                                 including from prototype construction -- see README)
    #   A8  Fixed lambda_p, lambda_s --fixed-lambda-p X --fixed-lambda-s Y (X,Y in (0,1))
    #   A9  No S_p term              --fixed-lambda-p 0
    #   A10 No S_struct term         --fixed-lambda-s 0
    #   A11 Gumbel anchor gate       --use-gumbel-anchor
    #   A12 Gumbel hop selection     --use-gumbel-hop
    #   A13 Gumbel prototype gate    --use-gumbel-proto
    #   Full                         all defaults
    # ------------------------------------------------------------------
    ablation_group = parser.add_argument_group('Ablation Overrides')
    ablation_group.add_argument('--random-anchor-selection', action='store_true', dest='random_anchor_selection',
                                help='A1: replace query-conditioned alpha_i with a uniform '
                                     'distribution over valid candidates')
    ablation_group.add_argument('--uniform-hop-weighting', action='store_true', dest='uniform_hop_weighting',
                                help='A5: replace the learned beta_l = softmax(G_hop(q)) '
                                     'with a fixed uniform beta_l = 1/L')
    ablation_group.add_argument('--fixed-lambda-p', default=None, type=float, dest='fixed_lambda_p',
                                help='A8/A9: override the learned lambda_p with this '
                                     'constant for every query; 0.0 drops the S_p term entirely (A9)')
    ablation_group.add_argument('--fixed-lambda-s', default=None, type=float, dest='fixed_lambda_s',
                                help='A8/A10: override the learned lambda_s with this '
                                     'constant for every query; 0.0 drops the S_struct term entirely (A10)')

    # ------------------------------------------------------------------
    # Optional discrete (Gumbel-Softmax / Gumbel-Sigmoid) extensions,
    # (ablations A11-A13). Off by default;
    # the paper explicitly treats the dense mechanisms above as the core model.
    # ------------------------------------------------------------------
    gumbel_group = parser.add_argument_group('Gumbel Extensions (optional, ablations A11-A13)')
    gumbel_group.add_argument('--use-gumbel-anchor', action='store_true', dest='use_gumbel_anchor',
                              help='A11: Gumbel-Sigmoid hard keep/drop gate per anchor')
    gumbel_group.add_argument('--gumbel-tau-sel', default=0.5, type=float, dest='gumbel_tau_sel',
                              help='tau_sel: temperature for the anchor keep/drop gate')
    gumbel_group.add_argument('--use-gumbel-hop', action='store_true', dest='use_gumbel_hop',
                              help='A12: Gumbel-Softmax (top-k) hard hop selection')
    gumbel_group.add_argument('--gumbel-tau-hop', default=0.5, type=float, dest='gumbel_tau_hop',
                              help='tau_hop: temperature for hop selection')
    gumbel_group.add_argument('--gumbel-topk-hop', default=1, type=int, dest='gumbel_topk_hop',
                              help='k_hop: number of hop distances kept active per query')
    gumbel_group.add_argument('--use-gumbel-proto', action='store_true', dest='use_gumbel_proto',
                              help='A13: Gumbel-Sigmoid per-slot prototype activation gates')
    gumbel_group.add_argument('--gumbel-tau-proto', default=0.5, type=float, dest='gumbel_tau_proto',
                              help='tau_proto: temperature for prototype slot activation')

    # System settings
    system_group = parser.add_argument_group('System')
    system_group.add_argument('-j', '--workers', default=4, type=int,
                              help='Number of data loading workers')
    system_group.add_argument('-p', '--print-freq', default=20, type=int,
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
    """Generate dynamic paths based on task name. Reuses the exact same data
    directory layout so both pipelines can share preprocessed data."""
    task = arguments.task.upper()

    if arguments.train_path is None:
        arguments.train_path = str(SCRIPT_DIR / 'data' / task / 'train.txt.json')

    if arguments.valid_path is None:
        arguments.valid_path = str(SCRIPT_DIR / 'data' / task / 'valid.txt.json')

    if arguments.model_dir is None:
        arguments.model_dir = str(SCRIPT_DIR / 'data' / task / 'checkpoint_arpm')

    if arguments.eval_model_path is None:
        arguments.eval_model_path = str(SCRIPT_DIR / 'data' / task / 'checkpoint_arpm' / 'model_best.mdl')

    return arguments


def validate_args(arguments):
    """Validate parsed arguments."""
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
        raise ValueError(
            'Either --model-dir or a valid --eval-model-path must be provided'
        )

    if arguments.resume:
        if arguments.resume_path is None:
            arguments.resume_path = os.path.join(arguments.model_dir, 'model_last.mdl')
        if not os.path.exists(arguments.resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {arguments.resume_path}")

    if arguments.num_prototypes not in (1, 2, 4, 8):
        warnings.warn(
            f'--num-prototypes={arguments.num_prototypes} is outside the sweep {{1,2,4,8}} '
            f'used in the core model; proceeding anyway.'
        )

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
            import torch.cuda.amp
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
