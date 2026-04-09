# Vmotion

This repository now keeps only two core scripts at the project root for readability and fast iteration.

## Layout

- `data_generation.py`
  - Single-file dataset renderer and exporter (including preview grids).
- `experiment.py`
  - Single-file Stage 1 baseline (dataset loading, model, train, eval).
- `outputs/datasets/`
  - Dataset artifacts.
- `outputs/experiments/`
  - Experiment artifacts (checkpoints, curves, metrics, plots).

## Generate a Dataset

```bash
conda activate mujoco
python data_generation.py --output-root outputs/datasets/render_preview --train-count 24 --val-count 8 --test-count 8
```

For a trainable dataset:

```bash
conda activate mujoco
python data_generation.py --output-root outputs/datasets/trainable_dataset --train-count 5000 --val-count 500 --test-count 500
```

## Run Stage 1 Baseline

Train:

```bash
conda activate mujoco
python experiment.py train --data-root outputs/datasets/trainable_dataset --output-dir outputs/experiments/stage1_translation/run_20ep --epochs 20 --batch-size 128 --lr 1e-3
```

Evaluate:

```bash
conda activate mujoco
python experiment.py eval --data-root outputs/datasets/trainable_dataset --checkpoint outputs/experiments/stage1_translation/run_20ep/best.pt --output-dir outputs/experiments/stage1_translation/run_20ep_eval_test --split test
```
