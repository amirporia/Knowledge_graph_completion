#!/usr/bin/env bash
# Usage: eval.sh <checkpoint> [task] [test_path]
# The method (simkgc / hasa*) is read from the checkpoint. Graph re-ranking is on for SimKGC
# (neighbor_weight=0.05) and must be off for HaSa:  NEIGHBOR_WEIGHT=0 bash eval.sh ...  (or use eval_hasa.sh)
set -x
set -e

model_path="bert"
task="wn18rr"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    task=$1
    shift
fi

ROOT="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
cd "${ROOT}"
echo "working directory: ${ROOT}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${ROOT}/data/${task}"
fi

test_path="${DATA_DIR}/test.txt.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    test_path=$1
    shift
fi

neighbor_weight=0.05
rerank_n_hop=2
if [ "${task}" = "wn18rr" ]; then
# WordNet is a sparse graph, use more neighbors for re-rank
  rerank_n_hop=5
fi
if [ "${task}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
  neighbor_weight=0.0
fi
if [ -n "${NEIGHBOR_WEIGHT}" ]; then
  neighbor_weight="${NEIGHBOR_WEIGHT}"
fi

python3 -u -m HaSa_Restructured.evaluation.evaluate \
--task "${task}" \
--is-test \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${test_path}" "$@"
