#!/usr/bin/env bash
# HaSa on FB15k-237 (hyper-parameters of the original main_FB.py: constant LR, no warm-up).
# METHOD=hasa | hasa_hard_bias | hasa_wohard_bias   (default hasa).  Add --plus for HaSa+.
set -x
set -e

TASK="fb15k237"
METHOD="${METHOD:-hasa}"

ROOT="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
cd "${ROOT}"
echo "working directory: ${ROOT}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${ROOT}/checkpoint/${METHOD}_${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${ROOT}/data/${TASK}"
fi

python3 -u -m HaSa_Restructured.main \
--method "${METHOD}" \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model sentence-transformers/all-mpnet-base-v2 \
--pooling pooler \
--em-dim 500 \
--num-hard-neg 3 \
--num-false-neg 3 \
--debias-tau 1e-4 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 256 \
--lr 2e-5 \
--wd 0.01 \
--lr-scheduler constant \
--warmup 0 \
--grad-clip 5 \
--epochs 5 \
--seed 40 \
--print-freq 20 \
--workers 4 \
--max-to-keep 5 "$@"
