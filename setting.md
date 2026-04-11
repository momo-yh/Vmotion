# Setting (Single Source of Truth)

> This file defines the project-level research question, the core modeling claims, and the minimal experiment sequence needed to answer that question.
> The current Stage 2 summary is in `stage2.md`.
> Run commands are in `README.md`.

## 1) Root Question

Can motion-only supervision using only:

- adjacent monocular frames `I_t, I_t+1`
- known camera motion `T_t_to_t1`
- camera intrinsics `K`

induce local visual features that contain decodable geometric information such as ball-center depth or 3D position?

## 2) Root Claim

The proposed pipeline is based on three linked claims.

### 1. Motion-sensitive local features

Not all image features are equally useful for motion-based supervision.
We want to extract local features that are sensitive to the structured image changes induced by camera motion.

These features should:

- respond to motion in a stable way
- preserve locally meaningful structure
- support reliable supervision across adjacent frames

### 2. Temporal correspondence between adjacent frames

Adjacent frames are temporally related observations of the same scene.
Many local features in one frame should correspond to local features in the next frame.

We therefore want to learn temporal relations so that:

- corresponding local features can be matched across adjacent frames
- the model can use these matches as motion-related supervision

### 3. Spatial relations among informative local features

Useful structure is not only temporal but also spatial.
Informative local features lie in a geometric layout on the image plane.
Their meaning depends not only on local appearance, but also on spatial relations.

These spatial relations include:

- position relative to the image center
- relations to other informative local features
- stable geometric configurations formed by multiple points or corners

Such relations can provide additional geometric structure beyond isolated pointwise matching.

## 3) Scope of the Current Minimal Implementation

The current implementation is intended as a first validation of the proposed dataflow, not as a complete model of all motion types.

At present, it focuses on:

- local feature extraction
- local temporal relation construction
- local reliability weighting
- aggregation
- motion prediction

What it does **not** yet include explicitly is:

- explicit spatial relation modeling with respect to the image center
- explicit relation modeling among multiple informative local regions

Because of this, the current minimal version is best suited to motion types whose image effects can be described mainly by local displacement patterns.

This makes the first validation setting most appropriate for:

- left-right translation
- up-down translation

For these motions, local matching across adjacent frames provides a direct and meaningful supervision signal.

By contrast, the current minimal version is less well matched to:

- forward-backward translation
- camera rotation

These cases depend more strongly on global spatial structure:

- forward-backward motion induces expansion or contraction relative to the image center
- rotation induces motion patterns whose meaning also depends on image position

Those cases require stronger explicit spatial relation modeling, which is not yet included in the minimal implementation.

## 4) Computational Dataflow

The current minimal implementation follows the dataflow below:

```text
(I_t, I_t+1)
-> (Z_t, Z_t+1)
-> R_t
-> h_t
-> y_hat_t
```

Step 1: shared local feature extraction

```text
Z_t   = E_theta(I_t)
Z_t+1 = E_theta(I_t+1)
```

Step 2: local cross-frame relation construction

```text
R_t = R(Z_t, Z_t+1)
```

Step 3: relation aggregation into a compact representation

```text
h_t = A(R_t)
```

Step 4: motion prediction

```text
y_hat_t = H_psi(h_t)
```

Training uses motion supervision:

```text
L = L_motion(y_hat_t, y_t)
```

In the current minimal implementation, these abstract modules correspond to:

- `E_theta`: shared CNN encoder
- `R`: local cross-frame feature comparison
- `A`: spatial aggregation of relation features
- `H_psi`: regression head for the stage-specific motion target

## 5) Testing Overview

Depth decoding does **not** require a single-frame feature to directly encode `Z`.

What is needed is a motion-based relation:

- `Z_t`, `Z_t+1`, and motion direction
- per-point displacement `Delta u(p)` recovered from matching
- a geometric relation such as `Z(p) = fx * tau_x / Delta u(p)`

The key point is:

- depth can emerge from cross-frame correspondence and motion
- it does not need to appear as an explicit single-frame semantic code

Therefore, the correct test is not simply:

- does one frame alone contain `Z`

The correct test is:

- does the motion-trained representation support the cross-frame relations needed to recover `Z`

## 6) Minimal Experiment Sequence

Only a small number of experiments are needed to answer the root question.

### Stage 1: Motion-Decoding Sanity Check

Question:

- can the model decode a meaningful motion quantity from two frames?

Current task:

- input: `I_t, I_t+1`
- target: `T_t_to_t1[:3, 3] = (tx, ty, tz)`

Purpose:

- verify that the pipeline can use cross-frame visual change
- verify that the learned representation is motion-sensitive

Interpretation:

- If this fails, do not make geometry claims.
- If this succeeds, proceed to the geometry-facing test.

### Stage 2: Depth-Decoding Probe

Question:

- does the motion-trained representation contain enough information to support depth decoding at the ball-center location?

Current task:

- train the backbone with motion supervision
- freeze the backbone
- extract the local feature at the ball-center pixel
- train a lightweight probe to predict depth `z_t`

Purpose:

- test whether geometry-relevant information is decodable from the learned representation

Interpretation:

- If the frozen-feature depth probe beats weak baselines, there is initial evidence that geometry-relevant information emerged.
- If not, either geometry did not emerge or the current training objective is still too weak.

### Stage 3: Causal Control

Question:

- is the depth-decoding gain really caused by motion-trained features?

Required control:

- random frozen encoder plus the same depth probe

Optional additional controls:

- weak single-frame baseline
- corrupted motion supervision

Purpose:

- separate real motion-induced structure from dataset shortcuts or probe-only effects

Interpretation:

- If motion-trained features clearly outperform the random frozen control, the result supports a causal role for motion supervision.
- If the gap is weak or disappears, the apparent depth signal is not strong evidence for motion-induced geometry.

## 7) Decision Logic

### Case 1: Stage 1 fails

Meaning:

- the pipeline cannot yet use cross-frame motion signal reliably

Next:

- improve the architecture
- do not continue to geometry claims

### Case 2: Stage 1 succeeds, Stage 2 fails

Meaning:

- the model learns motion
- but the learned representation does not yet support useful depth decoding

Next:

- redesign the motion objective
- strengthen locality and correspondence structure

### Case 3: Stage 1 succeeds, Stage 2 succeeds, Stage 3 fails

Meaning:

- depth is decodable
- but not specifically because of motion-trained structure

Next:

- diagnose shortcuts
- redesign controls or supervision

### Case 4: Stage 1 succeeds, Stage 2 succeeds, Stage 3 succeeds

Meaning:

- motion supervision likely induced geometry-relevant information in the learned representation

Next:

- only then move to stronger claims such as richer 3D decoding

## 8) Core Success Criterion

The root question gets initial support only if all of the following happen:

1. the backbone can use motion signal
2. the frozen-feature depth probe beats weak baselines
3. the gain remains clearly better than the random frozen encoder control

Only then can we say:

> Motion-only supervision likely induced geometry-relevant local representation.

Before that, all conclusions should remain partial.

## 9) Shared Scene and Data Assumptions

- Scene: one ball, one table, one background wall
- Static world: the ball is fixed in the world; only camera motion changes from `t` to `t+1`
- Image resolution: `128 x 128`
- Intrinsics: `fx = fy = 100.0`, `cx = cy = 64.0`

Per-sample data:

- `img_t.png`
- `img_t1.png`
- `meta.json`
- `meta.json` includes `K`
- `meta.json` includes `T_t_to_t1`
- `meta.json` includes `ball_center_3d_t`
- `meta.json` includes `ball_center_2d_t`
- `meta.json` includes `ball_center_2d_t1`

## 10) Current Active Focus

The current minimal path is:

- Stage 1: 3D camera translation decoding
- Stage 2: frozen-feature depth probe
- Stage 3: random frozen encoder control

Current architectural constraint:

- shared encoder for `img_t` and `img_t1`
- explicit feature fusion with `[F_t, F_t1, F_t1 - F_t]`
- no global average pooling
- regression from spatial feature maps
- no explicit image-center relation modeling in the minimal version
- no explicit relation modeling among multiple local features in the minimal version

Current training constraint:

- use only the task supervision defined for each stage
- do not add ground-truth depth supervision into Stage 1
- do not add semantic or segmentation labels
- do not add reconstruction losses

## 11) Detailed Implementation

This section gives a minimal implementation that can be followed directly in code.
Each engineering choice is tied to the modeling logic above.

The implementation has four parts:

- a region encoder
- a local correlation module
- an aggregation and prediction module
- a training objective

The explicit reliability-weighting stage used in earlier drafts is removed in the current minimal version.
The reason is simple:

- the aggregation step and prediction head can learn to weight informative displacements implicitly
- an extra weighting module adds parameters without clear grounding in the current mathematical model

### 11.1 Scope of the Minimal Implementation

The minimal implementation is a first validation of the proposed dataflow, not a complete model of all motion types.

The current design explicitly models:

- local temporal correspondence across adjacent frames

It does **not** yet explicitly model:

- spatial relations with respect to the image center
- geometric relations among different local regions

Because of this, the current implementation is best suited to motion types whose image effects can be described mainly by local displacement patterns that are approximately spatially uniform.

We therefore restrict the first validation setting to:

- left-right translation
- up-down translation

For these motions, the induced displacement field satisfies approximately:

```text
Delta u(p) ~= fx * tau_x / Z(p)
Delta v(p) ~= fy * tau_y / Z(p)
```

This means:

- displacement at position `p` is a direct function of depth `Z(p)`
- the relation is grounded by the known translation direction
- local matching across adjacent frames provides a direct supervision signal

By contrast:

- forward-backward translation produces radial expansion or contraction relative to the image center
- camera rotation produces displacement patterns that depend explicitly on image position and rotation axis

Both cases require explicit spatial relation modeling with respect to image position, which is not included in the minimal version.

The purpose of the current implementation is therefore narrower:

- test whether the proposed pipeline already supports motion prediction
- test whether it induces geometry-relevant representation in the simpler translational setting

### 11.2 Region Encoder

Given two adjacent RGB frames `I_t` and `I_t+1`, we extract dense local features with a shared convolutional encoder:

```text
Z_t   = E_theta(I_t)
Z_t+1 = E_theta(I_t+1)
```

with

```text
Z_t, Z_t+1 in R^(H' x W' x d)
```

where:

- `H' = H / 4`
- `W' = W / 4`
- `d = 128`

The encoder is a three-layer CNN:

- `Conv 3 -> 32`, `3 x 3`, stride `1`, ReLU
- `Conv 32 -> 64`, `3 x 3`, stride `2`, ReLU
- `Conv 64 -> 128`, `3 x 3`, stride `2`, ReLU

Each design choice has a concrete reason.

Three layers:

- one layer gives only low-level edge responses
- the model needs a local receptive field large enough to capture region-level appearance structure
- three layers are the smallest practical depth that keeps the encoder simple while giving meaningful local support

`3 x 3` kernels:

- `1 x 1` kernels do not model spatial neighborhood structure
- local features should depend on the image content in a neighborhood around each position
- `3 x 3` is the smallest standard kernel that satisfies this

Stride `2` on layers 2 and 3:

- this gives a total downsampling factor of `4`
- it reduces the cost of the correlation step
- it also reduces effective displacement size on the feature map by the same factor

Shared weights:

- the two frames observe the same physical scene under small motion
- the same encoder should produce comparable features for the same physical surface
- this shared encoding is necessary for stable temporal correspondence

### 11.3 Local Correlation Module

The central object in the implementation is the local correlation between the two feature maps.
This directly instantiates:

```text
R_t = R(Z_t, Z_t+1)
```

For each position `p` in frame `t`, define a local search window in frame `t+1`:

```text
N_r(p) = { q : ||q - p||_inf <= r }
```

where `r` is the search radius on the feature map.

The search radius must satisfy:

```text
r >= max_p ||Delta p(p)||_inf / 4
```

where:

- `Delta p(p)` is the true inter-frame pixel displacement at position `p`
- the factor `4` comes from the encoder downsampling

This condition matters because:

- if the true match falls outside the search window, the correlation becomes unreliable
- that failure can happen silently unless it is checked before training

For each candidate `q` in the local window, compute a scaled dot-product score:

```text
C_t(p, q) = z_t(p)^T z_t+1(q) / sqrt(d)
```

Collecting all positions and all candidates gives the local correlation volume:

```text
C_t in R^(H' x W' x (2r+1)^2)
```

This is the minimal sufficient object for the translational setting because:

- it keeps the full displacement distribution at each position
- it does not collapse matching evidence too early
- the location of the correlation peak carries the displacement information needed for motion prediction

This is also the key geometric point:

- depth is not decoded from `Z_t(p)` alone
- depth becomes decodable from the interaction between two frames and the motion direction

In the translational case:

```text
Delta u(p) = fx * tau_x / Z(p)
```

So depth is implicit in where the matching distribution `C_t(p, .)` peaks.
That is why the correlation volume is the correct object for geometry-relevant validation, not single-frame features alone.

### 11.4 Aggregation and Motion Prediction

The correlation volume is aggregated by averaging over spatial positions while preserving the displacement dimension:

```text
h_t = (1 / |Omega'|) * sum_p C_t(p, .)
```

with

```text
h_t in R^((2r+1)^2)
```

This aggregation is justified for pure translation because:

- the dominant displacement is approximately uniform across the image
- spatial averaging preserves the shared displacement signal
- spatial averaging removes irrelevant position-specific variation

The displacement dimension must be preserved.
If it is collapsed too early:

- left and right motion become much harder to distinguish
- the prediction head loses the displacement pattern needed for direction prediction

The aggregated representation is then mapped to a motion prediction with a single linear layer:

```text
y_hat_t = W h_t + b
```

For the minimal translational setting:

```text
y_hat_t in R^2
```

predicting:

- horizontal translation
- vertical translation

The head is kept linear on purpose:

- the test should reveal whether the aggregated correlation representation already contains usable motion information
- a larger nonlinear head would obscure that question by compensating for a weak representation

### 11.5 Objective

The motion components can have different numerical ranges in the dataset.
We therefore normalize prediction and target before applying the loss.

```text
y_bar_t     = [Delta x_t / s_x, Delta y_t / s_y]
y_hat_bar_t = [Delta x_hat_t / s_x, Delta y_hat_t / s_y]
```

where:

- `s_x` is the maximum absolute horizontal range in the training set
- `s_y` is the maximum absolute vertical range in the training set

The training objective is mean squared error on the normalized motion target:

```text
L = 0.5 * || y_hat_bar_t - y_bar_t ||_2^2
```

### 11.6 Summary

The minimal implementation has four stages:

1. each frame is encoded by a shared three-layer CNN into a dense local feature map
2. local dot-product matching across a search window forms a correlation volume
3. the correlation volume is averaged over spatial positions while preserving the displacement dimension
4. a linear head predicts translation from the aggregated displacement representation

The central object throughout is the local correlation volume `C_t`.
It directly instantiates the temporal relation `R_t` from the modeling section and carries the displacement structure from which depth becomes decodable given motion direction.

### 11.7 Pseudocode

```text
===========================================================
  MINIMAL MOTION PIPELINE
===========================================================

INPUT
  I_t, I_t+1 : adjacent RGB frames
  y_t        : ground-truth translation (dx, dy)
  s_x, s_y   : normalization scales

PRECONDITION
  assert r >= max_pixel_displacement_in_dataset / 4
  if not, true correspondences can fall outside the search window

STAGE 1 -- Region Encoder  (shared weights)
  Z_t   = CNN(I_t)      -- [H', W', 128], H' = H/4, W' = W/4
  Z_t+1 = CNN(I_t+1)    -- [H', W', 128]

  CNN:
    3x3 conv + ReLU
    channels 3 -> 32 -> 64 -> 128
    strides  1, 2, 2

STAGE 2 -- Local Correlation
  for each position p in [H', W']:
    for each q in N_r(p):
      C[p, q] = dot(Z_t[p], Z_t+1[q]) / sqrt(128)

  C shape: [H', W', (2r+1)^2]
  C[p, :] is the displacement distribution at p

STAGE 3 -- Aggregation
  h = mean over p of C[p, :]   -- [(2r+1)^2]

STAGE 4 -- Prediction Head
  y_hat = W h + b              -- [2]

TRAINING OBJECTIVE
  y_bar     = (dx / s_x, dy / s_y)
  y_hat_bar = (dx_hat / s_x, dy_hat / s_y)
  L         = 0.5 * || y_hat_bar - y_bar ||^2
```

```text
===========================================================
  GEOMETRY VALIDATION  (after training)
===========================================================

  freeze encoder
  for held-out (I_t, I_t+1) with known depth:
    Z_t, Z_t+1 = Encoder(I_t, I_t+1)
    C          = LocalCorrelation(Z_t, Z_t+1)
    probe depth from C[p, :] at each position p
    not from Z_t[p] alone
```
