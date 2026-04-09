# Motion-Supervised Point Geometry Experiment

## Problem
**Can monocular image pairs and known ego-motion alone induce a local visual representation that contains decodable point-wise 3D geometry?**

Model uses: current image `I_t`, next image `I_{t+1}`, relative camera motion `T_{t->t+1}`, intrinsics `K`.
Model does NOT use: image reconstruction, ground-truth depth, direct 3D supervision, semantic labels.

Downstream test: Given a ball center pixel, can we decode its 3D position in camera frame from the learned local feature? Goal is NOT full scene reconstruction, but to test if motion supervision places usable local geometry into a spatial feature map.

## Hypothesis
If a model enforces cross-frame feature consistency under geometry-induced warping, then the local feature at an image point should contain enough geometric information to decode that point's 3D position.

## Minimal Validation
1. Train encoder using motion supervision only.
2. Freeze encoder, sample feature at ball-center pixel.
3. Train small MLP to predict ball-center 3D position.
4. Evaluate 3D error and reprojection error.

If this works, the learned feature likely contains ego-motion-consistent local geometry.

## Minimal Code Structure
```
data_gen/
├── renderer.py
├── dataset.py
└── generate_dataset.py

experiments/
└── stage1_translation/
    ├── model.py
    ├── train.py
    └── eval.py
```
- `data_gen/renderer.py`: rendering and dataset creation
- `data_gen/dataset.py`: paired-frame data loading
- `data_gen/generate_dataset.py`: dataset generation entry point
- `experiments/stage1_translation/model.py`: supervised frame-pair regressor
- `experiments/stage1_translation/train.py`: supervised training
- `experiments/stage1_translation/eval.py`: supervised evaluation

## Data
**Scene:** one sphere, one planar tabletop, one background wall, no other objects.

**Camera:** image size 128×128, fx = fy = 100.0, cx = cy = 64.0.
```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

**Ball placement:** x in [-0.25, 0.25], y = radius, z in [0.25, 0.75].

**Relative camera motion:** x/y/z translation in [-0.08, 0.08] m, yaw/pitch/roll in [-8, 8] deg.

**Per sample:** img_t.png, img_t1.png, T_t_to_t1 (4×4), K (3×3), ball_center_3d_t, ball_center_2d_t, ball_center_2d_t1.

**Dataset:** train 50,000, val 5,000, test 5,000.

**Important:** scene is static, only camera moves, ball does not move.

## Model

### Input
`I_t`, `I_{t+1}`, `T_{t->t+1}`, `K`.

### Output
- current-frame feature map: `z_t = E(I_t)`
- next-frame feature map: `z_t1 = E(I_{t+1})`
- current-frame depth-like map: `d_t = D(z_t)`

### Encoder
Input: [B, 3, 128, 128] → Output: [B, 16, 64, 64]
```
Conv(3, 16, 5, stride=2, padding=2), ReLU
Conv(16, 32, 3, stride=1, padding=1), ReLU
Conv(32, 16, 3, stride=1, padding=1)
```

### Geometry Head
Input: [B, 16, 64, 64] → Output: [B, 1, 64, 64]
```
Conv(16, 8, 3, padding=1), ReLU
Conv(8, 1, 1)
softplus + 1e-3
```

### Readout MLP
Input: sampled ball-center feature [B, 16] → Output: predicted 3D coordinate [B, 3]
```
Linear(16, 64), ReLU
Linear(64, 64), ReLU
Linear(64, 3)
```

## Geometry and Loss
For each feature location `p`:
```
X_hat_t(p) = d_hat_t(p) * K^{-1} * p_tilde
X_hat_t1'(p) = T_t_to_t1 * X_hat_t(p)
p' = pi(X_hat_t1'(p))
```
Sample next-frame feature at `p'`.

**Loss:** `L_feat = (1 / N) * sum_{p in Omega_valid} || z_t(p) - z_t1(p') ||_1`

This supervises depth indirectly through cross-frame correspondence, NOT directly.

**Do NOT add:** reconstruction loss, depth supervision, semantic supervision, smoothness loss, cycle loss, contrastive loss.

## Training

### Motion Training
Train `Encoder` and `GeometryHead` with input `I_t, I_t1, T, K` and loss `L_feat`.
```python
IMAGE_SIZE = 128
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
DEVICE = "cuda"
```

### Readout Training
Freeze `Encoder` and `GeometryHead`, train only `ReadoutMLP` to predict `ball_center_3d_t`.
```python
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
DEVICE = "cuda"
MOTION_CKPT = "./motion_ckpt.pt"
```
Loss: `L_readout = || X_hat_ball_t - X_ball_t ||_1`

## Evaluation

### 3D Error
`Err_3D = (1 / M) * sum_i || X_hat_i - X_i^* ||_2` (meters)

### Reprojection Error
`Err_reproj = (1 / M) * sum_i || p_hat_t1 - p_t1^* ||_2` where `p_hat_t1 = pi(T_t_to_t1 * X_hat_t)` (pixels)

### Depth Error at Queried Point
`Err_depth = (1 / M) * sum_i | d_hat_i - d_i^* |` (meters)

## Baseline
**Single-frame supervised readout:** train encoder and readout network directly with 3D supervision, without motion pretraining or cross-frame constraint. This baseline diagnoses task difficulty—if it performs strongly, good downstream decoding may come from static visual shortcuts (ball size, image height, table contact) rather than motion-induced geometry.

## Final Interpretation
1. Train spatial encoder with motion-only supervision.
2. Enforce feature consistency through geometry-induced warping.
3. Freeze learned representation.
4. Test whether local feature at a selected point supports 3D decoding.

This is the smallest clean test of whether ego-motion supervision induces point-wise local geometry.
