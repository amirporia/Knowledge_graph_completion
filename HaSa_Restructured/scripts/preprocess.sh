#!/usr/bin/env bash

set -x
set -e

TASK="wn18rr"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

# The package is run as a module from the project root (the folder containing HaSa_Restructured/ and data/)
ROOT="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
cd "${ROOT}"

python3 -u -m HaSa_Restructured.preprocess.preprocess \
--task "${TASK}" \
--train-path "${ROOT}/data/${TASK}/train.txt" \
--valid-path "${ROOT}/data/${TASK}/valid.txt" \
--test-path "${ROOT}/data/${TASK}/test.txt" "$@"
