#!/usr/bin/env bash
# Hyperparameters loosely follow the paper's Section 7.1 (WN18RR, sentence-BERT init).
# Paper reports BERT-base and sentence-BERT initializations; swap --pretrained-model to
# e.g. sentence-transformers/bert-base-nli-mean-tokens for the sentence-BERT variant.
set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--lr 2e-5 \
--wd 1e-4 \
--embedding-dim 500 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 8 \
--print-freq 20 \
--tau 2e-5 \
--num-false-neg-samples 4 \
--epochs 10 \
--workers 4 \
--max-to-keep 3 \
--early-stop-patience 5 \
--full-eval-every-n-epoch 1 "$@"
