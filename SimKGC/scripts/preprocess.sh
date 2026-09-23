#!/usr/bin/env bash
#
# BUGFIX: previously used bare relative paths ("./data/${TASK}/train.txt"),
# which only worked if you happened to invoke this script with your shell's
# CWD set to SimKGC/ -- and even then wrote into a SimKGC-local data/ folder
# instead of the one shared data/ directory every pipeline (HaSa, SimKGC, StAR,
# ARPM_KGC) actually reads from. Now resolved from the script's own location,
# regardless of caller's CWD, and pointed at that shared folder.
set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"     # this pipeline's own root, e.g. .../SimKGC
REPO_ROOT="$( cd "$DIR/.." && pwd )"                   # shared repo root, e.g. .../Knowledge_graph_completion
cd "$DIR"

TASK="wn18rr"
if [[ $# -ge 1 ]]; then
    TASK=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    shift
fi

if [ -z "$DATA_DIR" ]; then
  # Shared with HaSa, StAR and ARPM_KGC -- preprocessing writes its output
  # (train/valid/test.txt.json + entities.json) into this same shared folder,
  # e.g. F:\KGC\Knowledge_graph_completion\data\wn18rr, so every pipeline can
  # reuse it without a separate copy.
  DATA_DIR="${REPO_ROOT}/data/${TASK}"
fi

python3 -u preprocess.py \
--task "${TASK}" \
--train-path "${DATA_DIR}/train.txt" \
--valid-path "${DATA_DIR}/valid.txt" \
--test-path "${DATA_DIR}/test.txt"
