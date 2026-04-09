# Agent Notes

## Source Of Truth

Use [setting.md](./setting.md) as the primary experiment spec.

If code and `setting.md` disagree, update the code to match `setting.md`.

## Environment

Use the conda environment:

```bash
conda activate mujoco
```

Assume a simple deterministic pinhole renderer is preferred unless the user explicitly asks for a simulator-backed renderer.

## Dataset Expectations

The rendered dataset should follow the experiment spec:

* static scene
* one sphere
* one tabletop
* one wall
* only camera motion between `t` and `t+1`

Each sample should expose:

* `img_t.png`
* `img_t1.png`
* camera intrinsics `K`
* relative pose `T_t_to_t1`
* `ball_center_3d_t`
* `ball_center_2d_t`
* `ball_center_2d_t1`

## Working Style

Prefer small, inspectable outputs first:

* preview grids
* readable `meta.json`

Before scaling dataset size up, verify the render quality visually.

## File Layout

Keep experiments isolated.

Active top-level code areas:

* `data_gen/`
  Rendering, dataset generation, shared dataset loader.
* `experiments/stage1_translation/`
  Minimal supervised translation-regression baseline.

Do not mix experiment-specific training code back into `data_gen/`.
