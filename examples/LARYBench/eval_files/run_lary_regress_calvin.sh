#!/bin/bash
set -euo pipefail

# Required:
# - LARY_ROOT: path to cloned LARYBench repo
#
# Optional:
# - MODEL_TAG (default: starvla_qwen_last_hidden)
# - STRIDE (default: 5)
# - MODEL_TYPE (default: mlp)
#
# This script copies generated CSVs from starVLA examples/LARYBench/artifacts/
# to ${LARY_ROOT}/data and runs LARY regression.

: "${LARY_ROOT:?please set LARY_ROOT}"

MODEL_TAG="${MODEL_TAG:-starvla_qwen_last_hidden}"
STRIDE="${STRIDE:-5}"
MODEL_TYPE="${MODEL_TYPE:-mlp}"

TRAIN_SRC="examples/LARYBench/artifacts/train_la_calvin_${STRIDE}_${MODEL_TAG}.csv"
VAL_SRC="examples/LARYBench/artifacts/val_la_calvin_${STRIDE}_${MODEL_TAG}.csv"

TRAIN_DST="${LARY_ROOT}/data/train_la_calvin_${STRIDE}_${MODEL_TAG}.csv"
VAL_DST="${LARY_ROOT}/data/val_la_calvin_${STRIDE}_${MODEL_TAG}.csv"

if [[ ! -f "${TRAIN_SRC}" ]]; then
  echo "Missing ${TRAIN_SRC}"
  exit 1
fi
if [[ ! -f "${VAL_SRC}" ]]; then
  echo "Missing ${VAL_SRC}"
  exit 1
fi

cp "${TRAIN_SRC}" "${TRAIN_DST}"
cp "${VAL_SRC}" "${VAL_DST}"

cd "${LARY_ROOT}"
python -m lary.cli regress \
  --model "${MODEL_TAG}" \
  --dataset calvin \
  --stride "${STRIDE}" \
  --model-type "${MODEL_TYPE}"
