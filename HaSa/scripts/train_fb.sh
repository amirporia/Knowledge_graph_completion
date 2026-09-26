#!/usr/bin/env bash
# Hyperparameters loosely follow the paper's Section 7.1 (FB15k-237, sentence-BERT init).
# Paper reports BERT-base and sentence-BERT initializations; swap --pretrained-model to
# e.g. sentence-transformers/bert-base-nli-mean-tokens for the sentence-BERT variant.
#
# Multi-GPU: this script auto-detects every CUDA device visible to this process
# (e.g. Kaggle's T4 x2) and launches via `torchrun` when more than one is found,
# so training uses DistributedDataParallel across all of them -- the same way
# ARPM_KGC is launched (`torchrun --nproc_per_node=N -m ARPM_KGC.main ...`, see
# the top-level README). On a single-GPU (or CPU) machine this is a no-op: it
# falls back to a plain `python3 -u main.py` exactly as before.
# Set NPROC_PER_NODE explicitly to override auto-detection, e.g.
# `NPROC_PER_NODE=1 ./scripts/train_fb.sh` to force single-GPU training even on
# a multi-GPU machine.
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

NPROC_PER_NODE="${NPROC_PER_NODE:-$(python3 -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
  echo "Detected ${NPROC_PER_NODE} GPUs -> launching with torchrun (DistributedDataParallel)"
  LAUNCHER=(torchrun --nproc_per_node="${NPROC_PER_NODE}")
else
  LAUNCHER=(python3 -u)
fi

"${LAUNCHER[@]}" main.py \
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
