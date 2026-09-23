#!/usr/bin/env bash
set -x
set -e

model_path="bert"
task="wn18rr"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    task=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
REPO_ROOT="$( cd "$DIR/.." && pwd )"
cd "$DIR"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  # Shared with the other pipelines and ARPM_KGC (previously this pointed at
  # "${DIR}/data/${task}", a pipeline-local copy, and compared `task` against
  # the mixed-case literal "WN18RR" below, which never matched once `task`
  # comes from user input in whatever case they typed it).
  DATA_DIR="${REPO_ROOT}/data/${task}"
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

python3 -u evaluate.py \
--task "${task}" \
--is-test \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${test_path}" "$@"
