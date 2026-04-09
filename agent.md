# Agent Notes

## Source Of Truth

Use [setting.md](./setting.md) as the primary experiment specification.

If implementation and `setting.md` disagree, update implementation to match `setting.md`.

## Environment

Use the conda environment:

```bash
conda activate mujoco
```

Assume a simple deterministic pinhole renderer unless the user explicitly requests a simulator-backed renderer.

## Dataset Expectations

The dataset should follow the experiment specification:

- static scene
- one sphere
- one tabletop
- one wall
- only camera motion between `t` and `t+1`

Each sample should include:

- `img_t.png`
- `img_t1.png`
- camera intrinsics `K`
- relative pose `T_t_to_t1`
- `ball_center_3d_t`
- `ball_center_2d_t`
- `ball_center_2d_t1`

## Working Style

Prefer small, inspectable outputs first:

- preview grids
- readable `meta.json`

Before scaling dataset size, verify render quality visually.

## File Layout

Active top-level scripts:

- `data_generation.py`: rendering + dataset generation
- `experiment.py`: minimal supervised translation-regression baseline
