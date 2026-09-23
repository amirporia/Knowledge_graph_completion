#!/usr/bin/env bash
#
# Resuming: append --resume (and, if not resuming from the default
# <model-dir>/model_last.mdl, --resume-path /path/to/checkpoint.mdl) to continue
# training from the last saved epoch.
set -x
set -e

TASK="wn18rr"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"             # this pipeline's own root, e.g. .../SimKGC
REPO_ROOT="$( cd "$DIR/.." && pwd )"                           # shared repo root, e.g. .../Knowledge_graph_completion
cd "$DIR"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  # Shared with HaSa, StAR and ARPM_KGC -- one preprocessed copy of the data
  # for every pipeline (previously this pointed at "${DIR}/data/${TASK}", a
  # SimKGC-local copy that doesn't exist and used mixed-case "WN18RR").
  DATA_DIR="${REPO_ROOT}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 8 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 4 \
--max-to-keep 3 "$@"
