# Vmotion

This repo is now split into independent parts so experiments do not interfere with each other.

## Layout

* `data_gen/generate_dataset.py`
  Single-file renderer plus paired-frame dataset generation.
* `experiments/stage1_translation/`
  `run.py` is the single-file minimal supervised baseline: regress camera translation from two consecutive frames.
* `outputs/datasets/`
  Generated datasets only.
* `outputs/experiments/`
  Training runs, checkpoints, curves, and evaluation outputs only.

## Generate A Dataset

Use the `mujoco` conda environment:

```bash
conda activate mujoco
python -m data_gen.generate_dataset --output-root outputs/datasets/render_preview --train-count 24 --val-count 8 --test-count 8
```

For a trainable dataset:

```bash
conda activate mujoco
python -m data_gen.generate_dataset --output-root outputs/datasets/trainable_dataset --train-count 5000 --val-count 500 --test-count 500
```

Important outputs:

* `outputs/datasets/render_preview/train_preview_grid.png`
* `outputs/datasets/render_preview/val_preview_grid.png`
* `outputs/datasets/render_preview/test_preview_grid.png`

## Dataset Format

Each sample directory contains:

* `img_t.png`
* `img_t1.png`
* `meta.json`

Split-level files:

* `manifest.json`

Root-level files:

* `dataset_manifest.json`
* `train_preview_grid.png`
* `val_preview_grid.png`
* `test_preview_grid.png`

## Stage 1 Supervised Baseline

Train:

```bash
conda activate mujoco
python -m experiments.stage1_translation.run train --data-root outputs/datasets/trainable_dataset --output-dir outputs/experiments/stage1_translation/run_20ep --epochs 20 --batch-size 128 --lr 1e-3
```

Evaluate:

```bash
conda activate mujoco
python -m experiments.stage1_translation.run eval --data-root outputs/datasets/trainable_dataset --checkpoint outputs/experiments/stage1_translation/run_20ep/best.pt --output-dir outputs/experiments/stage1_translation/run_20ep_eval_test --split test
```
