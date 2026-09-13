import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Triple-based KGE baselines (RAA-KGC repo)')

    parser.add_argument('--model', required=True, type=str,
                        choices=['transe', 'distmult', 'complex', 'rotate', 'quate',
                                'qiqekgc', 'conve', 'rgcn'],
                        help='Which baseline to train/evaluate. Note: "qiqekgc" is the model '
                             'actually cited for the "QuatE" baseline in the original request '
                             '(Li et al. 2023, Information Sciences) -- see its module '
                             'docstring. "quate" is the plain Zhang et al. 2019 QuatE.')
    parser.add_argument('--task', default='wn18rr', type=str,
                        choices=['wn18rr', 'fb15k237', 'wiki5m_trans', 'wiki5m_ind'])

    # Paths — default to the same preprocessed data/<task>/ layout used by
    # Baseline/preprocess and SimKGC/preprocess (entities.json + *.txt.json triples).
    parser.add_argument('--data-dir', default=None, type=str,
                        help='Directory containing entities.json, train.txt.json, valid.txt.json, '
                             'test.txt.json (default: data/<task>/)')
    parser.add_argument('--model-dir', default=None, type=str,
                        help='Where to write checkpoints (default: data/<task>/checkpoint_<model>/)')

    # Model
    parser.add_argument('--embedding-dim', default=200, type=int)
    parser.add_argument('--margin', default=9.0, type=float, help='gamma for TransE/RotatE')
    parser.add_argument('--p-norm', default=1, type=int, help='TransE distance norm')
    parser.add_argument('--adv-temperature', default=1.0, type=float,
                        help='Self-adversarial negative sampling temperature')
    parser.add_argument('--dropout', default=0.2, type=float)
    # ConvE-specific
    parser.add_argument('--conve-height', default=10, type=int)
    parser.add_argument('--conve-filters', default=32, type=int)
    parser.add_argument('--conve-kernel', default=3, type=int)
    parser.add_argument('--conve-input-dropout', default=0.2, type=float,
                        help='"embedding dropout" in the paper, applied to the stacked (h,r) image')
    parser.add_argument('--conve-feature-dropout', default=0.2, type=float,
                        help='dropout applied to the conv output feature maps')
    parser.add_argument('--conve-hidden-dropout', default=0.3, type=float,
                        help='"projection layer dropout" in the paper, applied after the FC layer')
    # RGCN-specific
    parser.add_argument('--rgcn-in-dim', default=200, type=int)
    parser.add_argument('--rgcn-hidden-dim', default=200, type=int)
    parser.add_argument('--rgcn-num-bases', default=30, type=int)
    # QIQE-KGC-specific (Eq. 18-19, 24)
    parser.add_argument('--qiqe-alpha', default=0.5, type=float,
                        help='alpha: weight of the quantum-embedding score/loss component. '
                             'Paper found ~1:1 alpha:beta best via sensitivity sweep (Fig. 4).')
    parser.add_argument('--qiqe-beta', default=0.5, type=float,
                        help='beta: weight of the quaternion score/loss component')
    parser.add_argument('--qiqe-reg-weight', default=0.01, type=float,
                        help='weight of the Eq. 23-26 entity/relation regularization term '
                             '(this repo\'s substitute for the paper\'s own quantum-module '
                             'training loss -- see qiqekgc.py\'s module docstring)')

    # Training
    parser.add_argument('--neg-size', default=256, type=int, help='negatives per positive triple')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('-b', '--batch-size', default=1024, type=int)
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('--eval-entity-chunk', default=5000, type=int,
                        help='Chunk size when scoring against the full entity set (memory control). '
                             'Distance-based models (TransE/RotatE) do an O(batch*chunk*dim) broadcast '
                             'per chunk at eval time, so lower this further for large embedding-dim '
                             'RotatE runs or large entity vocabularies (e.g. wiki5m) if you hit OOM.')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--grad-clip', default=5.0, type=float)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-p', '--print-freq', default=100, type=int)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', default=42, type=int)

    # Early stopping / checkpointing (best model = highest validation MRR)
    parser.add_argument('--eval-every', default=1, type=int, help='evaluate every N epochs')
    parser.add_argument('--patience', default=10, type=int,
                        help='stop after this many evals with no MRR improvement')
    parser.add_argument('--max-to-keep', default=1, type=int)

    args = parser.parse_args()

    if args.data_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.data_dir = os.path.join(repo_root, 'data', args.task)
    if args.model_dir is None:
        args.model_dir = os.path.join(args.data_dir, f'checkpoint_{args.model}')

    return args
