#!/usr/bin/env bash

set -x
set -e

TASK="wn18rr"

ROOT="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
cd "${ROOT}"
echo "working directory: ${ROOT}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${ROOT}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${ROOT}/data/${TASK}"
fi

# Single GPU:  python3 -u -m SimKGC_Restructured.main ...
# Multi GPU:   torchrun --nproc_per_node=N -m SimKGC_Restructured.main ...  (--batch-size is per GPU)
python3 -u -m SimKGC_Restructured.main \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 4 \
--max-to-keep 3 "$@"
