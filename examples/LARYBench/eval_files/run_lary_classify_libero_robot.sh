#!/bin/bash
set -euo pipefail

# Required:
# - LARY_ROOT
#
# Optional:
# - MODEL_TAG (default: starvla_qwen_last_hidden)
# - DATASETS (default: "robot_1st libero")
# - CLASSIFY_BATCH_SIZE (default: 256)
# - GPUS (default: 0)
# - FORCE_DIM (if set, skip auto dim detection)
# - SMOKE_SECONDS (if > 0, run classify under timeout for smoke)

: "${LARY_ROOT:?please set LARY_ROOT}"

MODEL_TAG="${MODEL_TAG:-starvla_qwen_last_hidden}"
DATASETS="${DATASETS:-robot_1st libero}"
CLASSIFY_BATCH_SIZE="${CLASSIFY_BATCH_SIZE:-256}"
GPUS="${GPUS:-0}"
FORCE_DIM="${FORCE_DIM:-}"
SMOKE_SECONDS="${SMOKE_SECONDS:-0}"

count_classes() {
  local csv_path="$1"
  python - "$csv_path" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
actions = set()
with open(csv_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        actions.add(str(row.get('action', '')))
print(len(actions))
PY
}

infer_dim_from_npz() {
  local csv_path="$1"
  python - "$csv_path" <<'PY'
import csv
import os
import sys
import numpy as np

csv_path = sys.argv[1]
first = None
with open(csv_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = str(row.get('la_path', '')).strip()
        if p:
            first = p
            break
if first is None:
    raise RuntimeError(f'No la_path found in {csv_path}')
if not os.path.isabs(first):
    first = os.path.abspath(first)
arr = np.load(first, allow_pickle=True)['tokens']
if arr.ndim == 0:
    raise RuntimeError(f'Invalid tokens shape: {arr.shape}')
print(int(arr.shape[-1]))
PY
}

for DATASET in ${DATASETS}; do
  TRAIN_SRC="examples/LARYBench/artifacts/train_la_${DATASET}_${MODEL_TAG}.csv"
  VAL_SRC="examples/LARYBench/artifacts/val_la_${DATASET}_${MODEL_TAG}.csv"
  TRAIN_DST="${LARY_ROOT}/data/train_la_${DATASET}_${MODEL_TAG}.csv"
  VAL_DST="${LARY_ROOT}/data/val_la_${DATASET}_${MODEL_TAG}.csv"

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

  CLASSES="$(count_classes "${TRAIN_DST}")"
  if [[ -n "${FORCE_DIM}" ]]; then
    DIM="${FORCE_DIM}"
  else
    DIM="$(infer_dim_from_npz "${TRAIN_DST}")"
  fi

  echo "[${DATASET}] classes=${CLASSES}, dim=${DIM}"

  if [[ "${SMOKE_SECONDS}" -gt 0 ]]; then
    set +e
    (
      cd "${LARY_ROOT}"
      timeout "${SMOKE_SECONDS}" python -m lary.cli classify \
        --model "${MODEL_TAG}" \
        --dataset "${DATASET}" \
        --dim "${DIM}" \
        --classes "${CLASSES}" \
        --batch-size "${CLASSIFY_BATCH_SIZE}" \
        --gpus "${GPUS}"
    )
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 && "${rc}" -ne 124 ]]; then
      echo "[${DATASET}] classify failed (exit=${rc})"
      exit "${rc}"
    fi
    echo "[${DATASET}] classify smoke passed (exit=${rc}, 124=timeout expected)"
  else
    (
      cd "${LARY_ROOT}"
      python -m lary.cli classify \
        --model "${MODEL_TAG}" \
        --dataset "${DATASET}" \
        --dim "${DIM}" \
        --classes "${CLASSES}" \
        --batch-size "${CLASSIFY_BATCH_SIZE}" \
        --gpus "${GPUS}"
    )
  fi
done
