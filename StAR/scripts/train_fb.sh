#!/usr/bin/env bash
# Hyperparameters follow the paper's Appendix A / Table 11 (RoBERTa init, FB15k-237).
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
  DATA_DIR="${REPO_ROOT}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model roberta-base \
--lr 1e-5 \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--task ${TASK} \
--batch-size 16 \
--print-freq 50 \
--num-negatives 5 \
--margin 1.0 \
--structure-loss-weight 1.0 \
--epochs 7 \
--workers 4 \
--max-to-keep 3 \
--early-stop-patience 5 \
--full-eval-every-n-epoch 1 "$@"
