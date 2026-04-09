# Stage 1 (Implementation and Usage)

> This file explains how to run the current Stage 1 baseline.
> For experiment assumptions and constraints, refer to `setting.md`.

## 1) Single-file Entry Points

- `experiment.py`
  - `train`: supervised regression baseline from two frames to camera translation.
  - `eval`: evaluation with MAE/RMSE and prediction scatter plots.
- `data_generation.py`
  - train/val/test dataset generation and preview export.

## 2) Prepare Data

```bash
conda activate mujoco
python data_generation.py --output-root outputs/datasets/trainable_dataset --train-count 5000 --val-count 500 --test-count 500
```

## 3) Train

```bash
conda activate mujoco
python experiment.py train \
  --data-root outputs/datasets/trainable_dataset \
  --output-dir outputs/experiments/stage1_translation/run_20ep \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3
```

Main outputs:
- `best.pt`
- `last.pt`
- `history.json`
- `curves.png`
- `summary.json`

## 4) Evaluate

```bash
conda activate mujoco
python experiment.py eval \
  --data-root outputs/datasets/trainable_dataset \
  --checkpoint outputs/experiments/stage1_translation/run_20ep/best.pt \
  --output-dir outputs/experiments/stage1_translation/run_20ep_eval_test \
  --split test
```

Evaluation outputs:
- `metrics.json`
- `prediction_scatter.png`

## 5) Notes

This Stage 1 setup is a simplified supervised baseline (frame pair -> camera translation) to validate the data/training pipeline first.
Motion-only geometry conclusions must follow the boundaries in `setting.md`.
