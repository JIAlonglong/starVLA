# LARYBench Integration (Qwen `last_hidden`)

This directory provides a minimal integration to evaluate StarVLA checkpoints on LARYBench using Qwen visual-language hidden features as latent-action representations.

Current scope:
- Regression: `calvin`
- Classification: `robot_1st`, `libero`
- Feature source: Qwen VL `hidden_states[-1]`
- Representation: pooled source/target hidden states, latent = `tgt - src`

## 1) Export latents

### 1.1 Calvin regression

```bash
STARVLA_CKPT=/path/to/steps_xxx_pytorch_model.pt \
LARY_ROOT=/path/to/LARYBench \
DATA_DIR=/path/to/LARYBench \
LARY_LA_DIR=/path/to/latent_actions \
bash examples/LARYBench/eval_files/run_export_calvin_qwen_hidden.sh
```

This writes:
- NPZ: `$LARY_LA_DIR/calvin/stride_5/{train,val}/starvla_qwen_last_hidden/`
- CSV: `examples/LARYBench/artifacts/train_la_calvin_5_starvla_qwen_last_hidden.csv`
- CSV: `examples/LARYBench/artifacts/val_la_calvin_5_starvla_qwen_last_hidden.csv`

### 1.2 Libero / Robot_1st classification

```bash
STARVLA_CKPT=/path/to/steps_xxx_pytorch_model.pt \
LARY_ROOT=/path/to/LARYBench \
DATA_DIR=/path/to/LARYBench \
LARY_LA_DIR=/path/to/latent_actions \
DATASETS="robot_1st libero" \
bash examples/LARYBench/eval_files/run_export_classify_qwen_hidden.sh
```

This writes:
- NPZ: `$LARY_LA_DIR/robot_1st/starvla_qwen_last_hidden/`
- NPZ: `$LARY_LA_DIR/libero/starvla_qwen_last_hidden/`
- CSV artifacts:
  - `examples/LARYBench/artifacts/train_la_robot_1st_starvla_qwen_last_hidden.csv`
  - `examples/LARYBench/artifacts/val_la_robot_1st_starvla_qwen_last_hidden.csv`
  - `examples/LARYBench/artifacts/train_la_libero_starvla_qwen_last_hidden.csv`
  - `examples/LARYBench/artifacts/val_la_libero_starvla_qwen_last_hidden.csv`

## 2) Run LARY tasks

### 2.1 Calvin regression

```bash
LARY_ROOT=/path/to/LARYBench \
bash examples/LARYBench/eval_files/run_lary_regress_calvin.sh
```

Equivalent core command:

```bash
python -m lary.cli regress \
  --model starvla_qwen_last_hidden \
  --dataset calvin \
  --stride 5 \
  --model-type mlp
```

### 2.2 Libero / Robot_1st classification

```bash
LARY_ROOT=/path/to/LARYBench \
GPUS=0 \
DATASETS="robot_1st libero" \
bash examples/LARYBench/eval_files/run_lary_classify_libero_robot.sh
```

Notes:
- Script copies artifact CSVs into `${LARY_ROOT}/data/` as:
  - `train_la_<dataset>_starvla_qwen_last_hidden.csv`
  - `val_la_<dataset>_starvla_qwen_last_hidden.csv`
- It auto-computes:
  - `--classes` from unique `action` in train CSV
  - `--dim` from first latent `.npz` (`tokens.shape[-1]`)

## 3) Smoke check (la_path + startup)

```bash
LARY_ROOT=/path/to/LARYBench \
GPUS=0 \
SMOKE_SECONDS=120 \
bash examples/LARYBench/eval_files/run_lary_smoke_qwen_hidden.sh
```

This script:
- validates all expected CSVs exist
- validates every `la_path` is resolvable and file exists
- launches `lary.cli regress` with timeout
- launches `lary.cli classify` for `robot_1st/libero` with timeout

## Notes

- `robot_1st` uses `video_path + sample_indices` from metadata; exporter uses first/last sampled frame as source/target.
- `libero` and `calvin` use `src_img/tgt_img` from metadata.
- Exported CSV `la_path` is written as absolute path (regression + classification) to avoid downstream resolver/model-name mismatch issues.
