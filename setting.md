# Setting (Single Source of Truth)

> This file defines what the experiment is testing, what data is required, and how success is interpreted.
> Implementation details and run commands are intentionally moved to `stage1.md`.

## 1) Core Question

Can we use only:
- monocular adjacent frames `I_t, I_{t+1}`
- known relative camera motion `T_{t->t+1}`
- camera intrinsics `K`

to induce local visual features that carry decodable geometric information (for example, ball-center 3D position)?

## 2) Scene and Camera

- Scene: one ball + one table + one background wall.
- Static world: the ball is fixed; only camera motion happens from `t` to `t+1`.
- Image resolution: `128 x 128`.
- Intrinsics: `fx = fy = 100.0`, `cx = cy = 64.0`.

## 3) Per-sample Data

- `img_t.png`
- `img_t1.png`
- `meta.json` containing:
  - `K`
  - `T_t_to_t1`
  - `ball_center_3d_t`
  - `ball_center_2d_t`
  - `ball_center_2d_t1`

## 4) Training Constraints

When the experiment is declared motion-only, do not use:
- ground-truth depth supervision
- semantic/segmentation labels
- reconstruction losses

## 5) Downstream Probe

After training, freeze the backbone and train only a lightweight readout on the local feature at the ball-center pixel to predict:
- depth, or
- 3D coordinates

## 6) Minimal Success Criterion

If motion-supervised features clearly outperform random (or weak baseline) features on ball-center depth/3D decoding, we have initial evidence that geometry-relevant local information emerged.
