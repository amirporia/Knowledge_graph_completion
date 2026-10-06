#!/usr/bin/env bash
# Evaluate a HaSa checkpoint (no graph re-ranking; HaSa scores are raw dot products).
# Usage: eval_hasa.sh <checkpoint> [task] [test_path]
export NEIGHBOR_WEIGHT=0.0
exec bash "$( dirname "$0" )/eval.sh" "$@"
