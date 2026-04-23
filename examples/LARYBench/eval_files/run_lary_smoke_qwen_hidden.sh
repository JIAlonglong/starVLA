#!/bin/bash
set -euo pipefail

# Smoke check for qwen-last-hidden integration:
# 1) check la_path resolvability / file existence
# 2) start regress/classify and verify they can launch into training loop
#
# Required:
# - LARY_ROOT
#
# Optional:
# - MODEL_TAG (default: starvla_qwen_last_hidden)
# - STRIDE (default: 5)
# - MODEL_TYPE (default: mlp)
# - GPUS (default: 0)
# - CLASSIFY_BATCH_SIZE (default: 256)
# - SMOKE_SECONDS (default: 120)

: "${LARY_ROOT:?please set LARY_ROOT}"

MODEL_TAG="${MODEL_TAG:-starvla_qwen_last_hidden}"
STRIDE="${STRIDE:-5}"
MODEL_TYPE="${MODEL_TYPE:-mlp}"
GPUS="${GPUS:-0}"
CLASSIFY_BATCH_SIZE="${CLASSIFY_BATCH_SIZE:-256}"
SMOKE_SECONDS="${SMOKE_SECONDS:-120}"

python - "${LARY_ROOT}" "${MODEL_TAG}" "${STRIDE}" <<'PY'
import csv
import os
import sys

lary_root, model, stride = sys.argv[1], sys.argv[2], sys.argv[3]
specs = [
    (os.path.join(lary_root, 'data', f'train_la_calvin_{stride}_{model}.csv'), 'calvin', 'train', model),
    (os.path.join(lary_root, 'data', f'val_la_calvin_{stride}_{model}.csv'), 'calvin', 'val', model),
    (os.path.join(lary_root, 'data', f'train_la_robot_1st_{model}.csv'), 'robot_1st', 'all', model),
    (os.path.join(lary_root, 'data', f'val_la_robot_1st_{model}.csv'), 'robot_1st', 'all', model),
    (os.path.join(lary_root, 'data', f'train_la_libero_{model}.csv'), 'libero', 'all', model),
    (os.path.join(lary_root, 'data', f'val_la_libero_{model}.csv'), 'libero', 'all', model),
]

from lary.path_resolver import resolve_la_path

for csv_path, dataset, split, model_name in specs:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'missing csv: {csv_path}')

    checked = 0
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = str(row.get('la_path', '')).strip()
            if not p:
                raise RuntimeError(f'empty la_path in {csv_path}')
            resolved = p if os.path.isabs(p) else resolve_la_path(p, dataset, split, model_name)
            if not resolved or not os.path.isfile(resolved):
                raise FileNotFoundError(f'la_path not found: csv={csv_path}, value={p}, resolved={resolved}')
            checked += 1
    print(f'[OK] {csv_path}: {checked} entries')
PY

set +e
(
  cd "${LARY_ROOT}"
  timeout "${SMOKE_SECONDS}" python -m lary.cli regress \
    --model "${MODEL_TAG}" \
    --dataset calvin \
    --stride "${STRIDE}" \
    --model-type "${MODEL_TYPE}"
)
REGRESS_RC=$?
set -e
if [[ "${REGRESS_RC}" -ne 0 && "${REGRESS_RC}" -ne 124 ]]; then
  echo "regress smoke failed (exit=${REGRESS_RC})"
  exit "${REGRESS_RC}"
fi

echo "regress smoke passed (exit=${REGRESS_RC}, 124=timeout expected)"

SMOKE_SECONDS="${SMOKE_SECONDS}" GPUS="${GPUS}" CLASSIFY_BATCH_SIZE="${CLASSIFY_BATCH_SIZE}" \
  bash examples/LARYBench/eval_files/run_lary_classify_libero_robot.sh

echo "classification smoke finished."
