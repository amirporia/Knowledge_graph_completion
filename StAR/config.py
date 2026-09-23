import os
import random
import torch
import argparse
import warnings

from pathlib import Path

import torch.backends.cudnn as cudnn

# Repository root shared by every pipeline (ARPM_KGC, HaSa, SimKGC, StAR): this
# file lives at <repo_root>/StAR/config.py, so its parent's parent is
# <repo_root>. All four pipelines read/write the SAME preprocessed data under
# <repo_root>/data/<task>/ (e.g. F:\KGC\Knowledge_graph_completion\data\wn18rr)
# -- there is no separate StAR/data copy.
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description='StAR arguments')
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--task', default='wn18rr', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default=None, type=str, metavar='N',
                    help='path to training data (default: <repo_root>/data/<task>/train.txt.json)')
parser.add_argument('--valid-path', default=None, type=str, metavar='N',
                    help='path to valid data (default: <repo_root>/data/<task>/valid.txt.json)')
parser.add_argument('--model-dir', default=None, type=str, metavar='N',
                    help='path to model dir (default: <this_pipeline>/checkpoint_runtime/<task>)')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=5, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--pooling', default='cls', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--dropout', default=0.1, type=float, metavar='N',
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true',
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=0, type=int,
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.0, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature as a trainable parameter or not')
parser.add_argument('--max-num-tokens', default=50, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', dest='use_self_negative', action='store_true', default=True,
                    help='use head entity as negative')
parser.add_argument('--no-self-negative', dest='use_self_negative', action='store_false',
                    help='BUGFIX: disable head-entity-as-negative. The old --use-self-negative '
                         'flag was declared as action=\'store_true\' with default=True, which '
                         'means it could never actually be turned off from the command line '
                         '(passing it or not passing it both leave it True) -- this flag is '
                         'the fix, and --use-self-negative keeps working exactly as before. '
                         '(Not actually read anywhere in StAR\'s own loss -- kept only for '
                         'CLI/config compatibility with SimKGC/HaSa.)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# only used for evaluation
parser.add_argument('--is-test', default=False, action='store_true',
                    help='is in test mode or not')
parser.add_argument('--rerank-n-hop', default=2, type=int,
                    help='use n-hops node for re-ranking entities, only used during evaluation')
parser.add_argument('--neighbor-weight', default=0.05, type=float,
                    help='weight for re-ranking entities')
parser.add_argument('--eval-model-path', default=None, type=str, metavar='N',
                    help='path to model, only used for evaluation (default: <model-dir>/model_best.mdl)')

# --- StAR-specific (Wang et al. 2021, WWW'21) — matches paper's §3.1-3.3 -------
# u = Pool(Enc([h;r])), v = Pool(Enc([t])) with a *tied* (Siamese) encoder (§3.1).
# Two training objectives (Eq. 12 & 13), combined as L = L^c + gamma * L^d (Eq. 14):
#   L^c: BCE classification over c=[u; u*v; u-v; v] -> MLP -> binary logit (Eq. 8-9,12)
#   L^d: margin hinge loss on s^d = -||u-v||_2 (Eq. 11, 13)
parser.add_argument('--interaction-hidden-dim', default=256, type=int,
                    help='hidden size of the classification MLP over [u, u*v, u-v, v] (Eq. 8-9)')
parser.add_argument('--num-negatives', default=5, type=int,
                    help='|N(tp)| in the paper: negatives sampled per positive triple by '
                         'corrupting head or tail uniformly at random (Appendix A: default 5)')
parser.add_argument('--margin', default=1.0, type=float,
                    help='lambda in Eq. 13, the hinge margin for the spatial/contrastive objective')
parser.add_argument('--structure-loss-weight', default=1.0, type=float,
                    help='gamma in Eq. 14: weight of the spatial structure-learning loss L^d '
                         'relative to the classification loss L^c')
# ----------------------------------------------------------------------------------

# --- early stopping / MRR-based best-model selection -----------------------------
parser.add_argument('--early-stop-patience', default=5, type=int,
                    help='stop training after this many full-MRR evals with no improvement')
parser.add_argument('--full-eval-every-n-epoch', default=1, type=int,
                    help='run the expensive full-corpus filtered-MRR eval (used for '
                         'early stopping / best-checkpoint selection) every N epochs. '
                         'Increase this on large graphs like wiki5m to control cost.')
parser.add_argument('--mrr-eval-batch-size', default=256, type=int,
                    help='batch size used only for the full-corpus MRR eval')
# ----------------------------------------------------------------------------------

# --- NEW: resume training ---------------------------------------------------------
parser.add_argument('--resume', action='store_true',
                    help='Resume training from a checkpoint (model, optimizer, scheduler, '
                         'AMP scaler, epoch, best-metric and early-stopping state)')
parser.add_argument('--resume-path', default=None, type=str,
                    help='Checkpoint to resume from (default: <model-dir>/model_last.mdl)')
# ----------------------------------------------------------------------------------

args = parser.parse_args()

# ------------------------------------------------------------------------------
# Resolve the shared data directory. HaSa, SimKGC, StAR and ARPM_KGC all read the
# SAME preprocessed files -- there is one `data/` folder at the repo root
# (e.g. F:\KGC\Knowledge_graph_completion\data), laid out as data/<task>/{train,
# valid,test}.txt.json + entities.json, with <task> lower-cased (wn18rr,
# fb15k237, wiki5m_trans, wiki5m_ind) to match ARPM_KGC/Baselines' convention.
# --train-path/--valid-path only fall back to this shared location if not given
# explicitly (scripts/*.sh normally pass them explicitly, and have been fixed to
# point at this same shared folder instead of a StAR-local data/ copy).
# ------------------------------------------------------------------------------
_task_lower = args.task.lower()
if args.train_path is None:
    args.train_path = str(REPO_ROOT / 'data' / _task_lower / 'train.txt.json')
if args.valid_path is None:
    args.valid_path = str(REPO_ROOT / 'data' / _task_lower / 'valid.txt.json')

assert not args.train_path or os.path.exists(args.train_path), \
    'Training data not found: {}'.format(args.train_path)
assert args.pooling in ['cls', 'mean', 'max']
assert args.task.lower() in ['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans']
assert args.lr_scheduler in ['linear', 'cosine']

if args.model_dir is None:
    args.model_dir = str(PIPELINE_DIR / 'checkpoint_runtime' / _task_lower)
if args.eval_model_path is None:
    args.eval_model_path = str(Path(args.model_dir) / 'model_best.mdl')

if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)
else:
    assert os.path.exists(args.eval_model_path), 'One of args.model_dir and args.eval_model_path should be valid path'
    args.model_dir = os.path.dirname(args.eval_model_path)

if args.resume:
    if args.resume_path is None:
        args.resume_path = os.path.join(args.model_dir, 'model_last.mdl')
    if not os.path.exists(args.resume_path):
        raise FileNotFoundError('Resume checkpoint not found: {}'.format(args.resume_path))

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn('GPU is not available, set use_amp=False and print_freq=1')
