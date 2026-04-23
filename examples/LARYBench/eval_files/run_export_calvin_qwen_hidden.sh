#!/bin/bash
set -euo pipefail

# Required:
# - STARVLA_CKPT
# - LARY_ROOT
# - DATA_DIR
# - LARY_LA_DIR
#
# Example:
# STARVLA_CKPT=/path/to/steps_50000_pytorch_model.pt \
# LARY_ROOT=/path/to/LARYBench \
# DATA_DIR=/path/to/LARYBench \
# LARY_LA_DIR=/path/to/latent_actions \
# bash examples/LARYBench/eval_files/run_export_calvin_qwen_hidden.sh

: "${STARVLA_CKPT:?please set STARVLA_CKPT}"
: "${LARY_ROOT:?please set LARY_ROOT}"
: "${DATA_DIR:?please set DATA_DIR (LARYBench dataset root)}"
: "${LARY_LA_DIR:?please set LARY_LA_DIR}"

MODEL_TAG="${MODEL_TAG:-starvla_qwen_last_hidden}"
STRIDE="${STRIDE:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"

TRAIN_CSV="${LARY_ROOT}/data/calvin_metadata_train.csv"
VAL_CSV="${LARY_ROOT}/data/calvin_metadata_val.csv"
TRAIN_ROOT="${DATA_DIR}/regression/calvin/train_stride5"
VAL_ROOT="${DATA_DIR}/regression/calvin/val_stride5"

python examples/LARYBench/eval_files/export_qwen_last_hidden.py \
  --ckpt_path "${STARVLA_CKPT}" \
  --input_csv "${TRAIN_CSV}" \
  --data_root "${TRAIN_ROOT}" \
  --la_root "${LARY_LA_DIR}" \
  --dataset calvin \
  --split train \
  --stride "${STRIDE}" \
  --model_tag "${MODEL_TAG}" \
  --batch_size "${BATCH_SIZE}"

python examples/LARYBench/eval_files/export_qwen_last_hidden.py \
  --ckpt_path "${STARVLA_CKPT}" \
  --input_csv "${VAL_CSV}" \
  --data_root "${VAL_ROOT}" \
  --la_root "${LARY_LA_DIR}" \
  --dataset calvin \
  --split val \
  --stride "${STRIDE}" \
  --model_tag "${MODEL_TAG}" \
  --batch_size "${BATCH_SIZE}"

echo "Done. CSV artifacts are under examples/LARYBench/artifacts/."
