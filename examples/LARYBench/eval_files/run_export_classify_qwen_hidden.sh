#!/bin/bash
set -euo pipefail

# Required:
# - STARVLA_CKPT
# - LARY_ROOT
# - DATA_DIR
# - LARY_LA_DIR
#
# Optional:
# - MODEL_TAG (default: starvla_qwen_last_hidden)
# - BATCH_SIZE (default: 16)
# - DATASETS (default: "robot_1st libero")

: "${STARVLA_CKPT:?please set STARVLA_CKPT}"
: "${LARY_ROOT:?please set LARY_ROOT}"
: "${DATA_DIR:?please set DATA_DIR (LARYBench dataset root)}"
: "${LARY_LA_DIR:?please set LARY_LA_DIR}"

MODEL_TAG="${MODEL_TAG:-starvla_qwen_last_hidden}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATASETS="${DATASETS:-robot_1st libero}"

for DATASET in ${DATASETS}; do
  case "${DATASET}" in
    robot_1st)
      DATA_ROOT="${DATA_DIR}/classification"
      ;;
    libero)
      DATA_ROOT="${DATA_DIR}/classification/LIBERO"
      ;;
    *)
      echo "Unsupported classification dataset: ${DATASET}"
      exit 1
      ;;
  esac

  for SPLIT in train val; do
    INPUT_CSV="${LARY_ROOT}/data/${DATASET}_metadata_${SPLIT}.csv"

    python examples/LARYBench/eval_files/export_qwen_last_hidden.py \
      --ckpt_path "${STARVLA_CKPT}" \
      --input_csv "${INPUT_CSV}" \
      --data_root "${DATA_ROOT}" \
      --la_root "${LARY_LA_DIR}" \
      --dataset "${DATASET}" \
      --split "${SPLIT}" \
      --model_tag "${MODEL_TAG}" \
      --batch_size "${BATCH_SIZE}"
  done

done

echo "Done. Classification CSV artifacts are under examples/LARYBench/artifacts/."
