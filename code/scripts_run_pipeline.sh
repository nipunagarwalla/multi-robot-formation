#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${ITERATIONS:-30}"
NUM_ENVS="${NUM_ENVS:-4}"
MAX_STEPS="${MAX_STEPS:-300}"
EPISODES="${EPISODES:-10}"
TAG="${TAG:-e2e}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"

$PYTHON_BIN code/train_hallway.py \
  --iterations "$ITERATIONS" \
  --num-envs "$NUM_ENVS" \
  --max-steps "$MAX_STEPS" \
  --tag "$TAG" \
  --device "$DEVICE"

RUN_DIR="$(ls -dt runs/*_${TAG} 2>/dev/null | head -n1)"
if [[ -z "$RUN_DIR" ]]; then
  echo "No run directory found for tag: $TAG" >&2
  exit 1
fi

$PYTHON_BIN code/eval_hallway.py \
  --weights "$RUN_DIR/weights/latest.pt" \
  --episodes "$EPISODES" \
  --max-steps "$MAX_STEPS" \
  --device "$DEVICE"

$PYTHON_BIN code/scripts/compare_runs.py "$RUN_DIR" --no-plot

echo "Pipeline complete. Results:"
echo "  Config:      $RUN_DIR/config.json"
echo "  Iterations:  $RUN_DIR/iterations.csv"
echo "  Episodes:    $RUN_DIR/episodes.jsonl"
echo "  Eval:        $RUN_DIR/eval.json"
