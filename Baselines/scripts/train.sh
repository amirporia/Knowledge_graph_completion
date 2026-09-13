#!/usr/bin/env bash
# Generic launcher for any of the 7 triple-based baselines.
# Usage: scripts/train.sh <model> <task> [extra python args...]
#   e.g. scripts/train.sh transe wn18rr --embedding-dim 200 --margin 9 --neg-size 256
#        scripts/train.sh rotate fb15k237 --embedding-dim 500 --margin 9 --adv-temperature 1.0
#        scripts/train.sh rgcn wn18rr --rgcn-in-dim 200 --rgcn-hidden-dim 200 --embedding-dim 200

set -x
set -e

MODEL=$1
TASK=$2
shift 2 || true

DIR="$( cd "$( dirname "$0" )" && cd .. && cd .. && pwd )"   # repo root
cd "$DIR"

python3 -u -m Baselines.main \
  --model "${MODEL}" \
  --task "${TASK}" \
  "$@"
