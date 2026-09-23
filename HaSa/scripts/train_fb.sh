#!/usr/bin/env bash
# Hyperparameters loosely follow the paper's Section 7.1 (FB15k-237, sentence-BERT init).
# Paper reports BERT-base and sentence-BERT initializations; swap --pretrained-model to
# e.g. sentence-transformers/bert-base-nli-mean-tokens for the sentence-BERT variant.
#
# Resuming: append --resume (and, if not resuming from the default
# <model-dir>/model_last.mdl, --resume-path /path/to/checkpoint.mdl) to continue
# training from the last saved epoch.
set -x
set -e

TASK="fb15k237"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
REPO_ROOT="$( cd "$DIR/.." && pwd )"
cd "$DIR"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  # Shared with SimKGC, StAR and ARPM_KGC
  DATA_DIR="${REPO_ROOT}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--lr 2e-5 \
--wd 1e-4 \
--embedding-dim 500 \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--task ${TASK} \
--batch-size 16 \
--print-freq 20 \
--tau 1e-4 \
--num-false-neg-samples 4 \
--epochs 10 \
--workers 4 \
--max-to-keep 3 \
--early-stop-patience 5 \
--full-eval-every-n-epoch 1 "$@"
