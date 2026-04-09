# Vmotion

This repository keeps the core dataset-generation and experiment code at the project root for fast iteration.

## Layout

- `data_generation.py`
  - Single-file dataset renderer and exporter, including preview grids.
- `stage1_exp1.py`
  - Stage 1 Experiment 1 pipeline for 2D displacement regression.
- `stage1_exp2.py`
  - Stage 1 Experiment 2 pipeline for 3D camera translation regression.
- `stage1.md`
  - Concise Stage 1 document covering Experiment 1 and Experiment 2.
- `stage2_exp1.py`
  - Stage 2 Experiment 1 pipeline for frozen local depth probing.
- `stage2.md`
  - Concise Stage 2 document for local geometry probing.
- `setting.md`
  - Project-level research question, experiment tree, and current active experiment.
- `outputs/datasets/`
  - Dataset artifacts.
- `outputs/experiments/`
  - Experiment artifacts such as checkpoints, curves, metrics, and plots.

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

## Run Stage 1 Experiment 1

Train:

```bash
conda activate mujoco
python stage1_exp1.py train --data-root outputs/datasets/trainable_dataset --output-dir outputs/experiments/experiment1_displacement/run_20ep --epochs 20 --batch-size 128 --lr 1e-3
```

Evaluate:

```bash
conda activate mujoco
python stage1_exp1.py eval --data-root outputs/datasets/trainable_dataset --checkpoint outputs/experiments/experiment1_displacement/run_20ep/best.pt --output-dir outputs/experiments/experiment1_displacement/run_20ep_eval_test --split test
```

## Run Stage 1 Experiment 2

Train:

```bash
conda activate mujoco
python stage1_exp2.py train --data-root outputs/datasets/trainable_dataset --output-dir outputs/experiments/experiment2_translation/run_20ep --epochs 20 --batch-size 128 --lr 1e-3
```

Evaluate:

```bash
conda activate mujoco
python stage1_exp2.py eval --data-root outputs/datasets/trainable_dataset --checkpoint outputs/experiments/experiment2_translation/run_20ep/best.pt --output-dir outputs/experiments/experiment2_translation/run_20ep_eval_test --split test
```

## Run Stage 2 Experiment 1

Train:

```bash
conda activate mujoco
python stage2_exp1.py train --data-root outputs/datasets/trainable_dataset --backbone-checkpoint outputs/experiments/experiment2_translation/run_20ep/best.pt --output-dir outputs/experiments/stage2_depth_probe/run_20ep --epochs 20 --batch-size 128 --lr 1e-3
```

Evaluate:

```bash
conda activate mujoco
python stage2_exp1.py eval --data-root outputs/datasets/trainable_dataset --checkpoint outputs/experiments/stage2_depth_probe/run_20ep/best.pt --output-dir outputs/experiments/stage2_depth_probe/run_20ep_eval_test --split test
```

## Current Stage

The current stage is Stage 1, with two sanity-check experiments:

- Experiment 1: 2D ball-center displacement decoding
- Experiment 2: 3D camera translation decoding
