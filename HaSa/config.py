import os
import random
import torch
import argparse
import warnings

import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='SimKGC arguments')
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--task', default='wn18rr', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default='WN18RR/train.txt.json', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='WN18RR/test.txt.json', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--model-dir', default='WN18RR/checkpoint_runtime/', type=str, metavar='N',
                    help='path to model dir')
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
parser.add_argument('--eval-every-n-step', default=10000, type=int,
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
parser.add_argument('--use-self-negative', default=True, action='store_true',
                    help='use head entity as negative')

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
parser.add_argument('--eval-model-path', default='WN18RR/checkpoint_runtime/model_best.mdl', type=str, metavar='N',
                    help='path to model, only used for evaluation')

# --- HaSa-specific (Zhang, Zhang & Molybog, WWW'24) — matches Algorithm 1 / Eq. 6-9 -
# L_HaSa(h,r,t) = -log( Pos / (Pos + NegHasa) ), where
#   Pos      = exp(e_hr . e_t)
#   Neg      = mean over in-batch tails t' != t of exp(e_hr . e_t')      (Eq. 5, "hard")
#   FalseNeg = mean over M structure-sampled 2-hop-neighbour candidates
#              of exp(e_hr . e_s), approximating E_{t~p-(t|e_hr,fact)}   (Eq. 8-13)
#   NegHasa  = K * ( Neg / (1 - tau) - tau * FalseNeg )                  (Algorithm 1)
parser.add_argument('--tau', default=1e-4, type=float,
                    help='tau = p(l=fact|e_hr) in Eq. 7-9: estimated probability that a '
                         'hard in-batch negative is actually a false negative. Paper\'s best '
                         'values: 2e-05 on WN18RR, 1e-04 on FB15k-237 (Section 7.5)')
parser.add_argument('--num-false-neg-samples', default=4, type=int,
                    help='M in Eq. 13: number of Monte-Carlo candidates sampled per training '
                         'example from alpha(t|e_hr), i.e. uniformly from the head entity\'s '
                         '<=2-hop link-graph neighbourhood (Eq. 9), to estimate FalseNeg')
# ----------------------------------------------------------------------------------

# --- NEW: early stopping / MRR-based best-model selection -----------------------
parser.add_argument('--early-stop-patience', default=5, type=int,
                    help='stop training after this many full-MRR evals with no improvement')
parser.add_argument('--full-eval-every-n-epoch', default=1, type=int,
                    help='run the expensive full-corpus filtered-MRR eval (used for '
                         'early stopping / best-checkpoint selection) every N epochs. '
                         'Increase this on large graphs like wiki5m to control cost.')
parser.add_argument('--mrr-eval-batch-size', default=256, type=int,
                    help='batch size used only for the full-corpus MRR eval')
# ----------------------------------------------------------------------------------

args = parser.parse_args()

assert not args.train_path or os.path.exists(args.train_path)
assert args.pooling in ['cls', 'mean', 'max']
assert args.task.lower() in ['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans']
assert args.lr_scheduler in ['linear', 'cosine']

if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)
else:
    assert os.path.exists(args.eval_model_path), 'One of args.model_dir and args.eval_model_path should be valid path'
    args.model_dir = os.path.dirname(args.eval_model_path)

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
