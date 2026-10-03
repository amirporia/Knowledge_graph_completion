#!/usr/bin/env bash

set -x
set -e

TASK="wiki5m_ind"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

ROOT="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
cd "${ROOT}"
echo "working directory: ${ROOT}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${ROOT}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${ROOT}/data/${TASK}"
fi

python3 -u -m SimKGC_Restructured.main \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 3e-5 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task "${TASK}" \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 0 \
--epochs 1 \
--workers 3 \
--max-to-keep 10 "$@"
