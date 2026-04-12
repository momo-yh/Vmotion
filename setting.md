# Setting (Single Source of Truth)

> This file defines the current project setting.
> The active implementation is now the three-frame, left-right-only minimal pipeline.

## 1) Root Question

Can motion-only supervision induce local visual features whose cross-frame correlations contain decodable geometric information, without using any depth label?

The current version answers this in the narrowest grounded setting:

- three adjacent monocular frames `I_t`, `I_t+1`, `I_t+2`
- known camera motions `T_t_to_t1`, `T_t1_to_t2`
- known camera intrinsics `K`
- left-right camera translation only

## 2) Computational Implementation

### 2.1 Overview: What We Want to Learn

The computational goal is to learn three things.

1. Motion-sensitive local features

- local features should respond to structured image changes induced by camera motion
- the response should be stable enough to support reliable matching

2. Temporal correspondence between adjacent frames

- corresponding scene points should produce matchable features across adjacent frames
- the correlation pattern should localize the correct displacement

3. Geometry-relevant structure in the learned representation

- depth is not a single-frame property here
- it becomes decodable from cross-frame matching together with known motion
- therefore the useful representation is the cross-frame correlation, not the single-frame feature alone

The core claim is:

> if temporal correspondence is learned precisely enough, then depth becomes decodable from the correlation pattern given the known camera motion.

### 2.2 A Minimal Computational Dataflow

The model takes three adjacent frames and two known motions:

```text
(I_t, I_t+1, I_t+2)
-> (F_t, F_t+1, F_t+2)
-> (C_1, C_2)
-> D_rel
-> q_pred
-> L
```

Shared encoder:

```text
F_t   = E_theta(I_t)
F_t+1 = E_theta(I_t+1)
F_t+2 = E_theta(I_t+2)
```

Two local correlation volumes:

```text
C_1 = C(F_t,   F_t+1)
C_2 = C(F_t+1, F_t+2)
```

Relative depth from the first correlation:

```text
D_rel(p) = fx * tau_1x / (mu_1x(p) * s)
```

Predicted position in frame `t+2`:

```text
q_pred(p) = (1 / s) * Pi(R_2 P(p) + tau_2, K)
```

Supervision:

- `q_pred(p)` is compared with the actual position implied by `C_2`
- the loss is per-point and uses no depth label

### 2.3 Scope of the Minimal Implementation

This implementation is a first validation of the pipeline, not a complete motion model.

Current restriction:

- left-right camera translation only

Reason:

- under pure horizontal translation, the image displacement is a clean function of depth
- the depth recovery relation is direct and geometrically grounded
- the camera must move far enough between adjacent frames so that scene points produce measurable image displacement

For this setting:

```text
Delta u(p) ~= fx * tau_1x / D(p)
Delta v(p) ~= 0
```

This is why the current implementation does not yet include:

- forward-backward translation
- camera rotation
- explicit image-center relation modeling
- explicit geometric relation modeling among multiple local regions

## 3) Detailed Implementation

The implementation has four parts:

- a region encoder
- a local correlation module
- a per-point supervision module
- a training objective

### 3.1 Region Encoder

Given three adjacent frames, a shared encoder produces:

```text
F_t, F_t+1, F_t+2 in R^(B x d x H' x W')
```

with:

- `s = 4`
- `H' = H / s`
- `W' = W / s`
- `d = 128`

Encoder architecture:

- `Conv 3 -> 32`, `3x3`, stride `1`, ReLU
- `Conv 32 -> 64`, `3x3`, stride `2`, ReLU
- `Conv 64 -> 128`, `3x3`, stride `2`, ReLU

Design reasons:

- three layers give a meaningful local receptive field
- `3x3` kernels preserve local spatial context
- stride `2` on layers 2 and 3 reduces correlation cost
- shared weights enforce temporal consistency across frames

### 3.2 Local Correlation Module

For each position `p` in the feature map, define a local search window:

```text
N_r(p) = { q : ||q - p||_inf <= r }
```

Precondition:

```text
r >= max_p ||Delta p(p)||_inf / s
```

Practical data constraint:

```text
|u_t+1 - u_t| must be large enough to be measurable on the image
```

If motion is too small:

- the correlation peak becomes too flat
- depth inversion becomes numerically unstable
- training can stall without learning useful correspondence

Scaled dot-product score:

```text
C_1(p, q) = F_t(p)^T F_t+1(q) / sqrt(d)
C_2(p, q) = F_t+1(p)^T F_t+2(q) / sqrt(d)
```

Both correlation volumes have shape:

```text
[B, H', W', (2r+1)^2]
```

### 3.3 Per-Point Supervision Module

Step 1: soft displacement from `C_1`

```text
a_1(p, q) = softmax_q C_1(p, q)
mu_1x(p)  = sum_q a_1(p, q) * (q_x - p_x)
Delta u(p)= mu_1x(p) * s
```

Step 2: relative depth recovery

```text
D_rel(p) = fx * tau_1x / (Delta u(p) + eps)
```

with clipping to a valid range `[D_min, D_max]`.

Step 3: predict the position in frame `t+2`

For the current left-right only setting:

```text
q_pred_x(p) = p_x + fx * tau_2x / (D_rel(p) * s)
q_pred_y(p) = p_y
```

The sign follows the implementation convention used in the generated dataset.

Step 4: actual position from `C_2`

```text
a_2(p, q)    = softmax_q C_2(p, q)
mu_2(p)      = sum_q a_2(p, q) * (q - p)
q_actual(p)  = p + mu_2(p)
```

Step 5: validity mask

A point is valid if:

- `D_rel(p) > 0`
- `q_pred(p)` stays inside the feature map

### 3.4 Training Objective

Per-point loss:

```text
L(p) = || q_pred(p) - q_actual(p) ||^2
```

Total loss:

```text
L = mean over valid p of L(p)
```

This forces the encoder to learn features whose correlation peaks are geometrically consistent across motion history.

## 4) Validation

### 4.1 What the Validation Tests

Successful training alone is not enough.
The model could reduce loss by learning shallow matching shortcuts.

Validation therefore tests the three objectives separately.

Two clarifications define the validation logic.

Depth is a two-frame quantity, not a single-frame property:

- depth is decoded from `C_1(p, .)`, not from `F_t(p)` alone

The system is a motion-history depth estimator:

- inference always uses frame pairs and known motion
- the model is never asked to predict depth from a single frame alone

### 4.2 Objective 1: Motion-Sensitive Local Features

Visualization:

- render the soft displacement field `mu_1x(p)` over the image

Expected pattern under correct learning:

- larger displacement at shallower depth
- smaller displacement at larger depth
- spatially smooth variation aligned with scene structure

Failure pattern:

- incoherent field
- nearly constant field
- field unrelated to scene depth

### 4.3 Objective 2: Temporal Correspondence

Qualitative test:

- for random points `p`, find `q* = argmax_q C_1(p, q)`
- draw lines from frame `t` to frame `t+1`

Expected pattern:

- mostly horizontal parallel correspondences
- longer lines for shallower regions
- shorter lines for deeper regions

Quantitative test:

- compare `q*_x(p)` against dense ground-truth correspondence derived from known motion and depth
- report mean matching error on valid points

### 4.4 Objective 3: Geometry-Relevant Representation

Level A: geometric recovery without additional learning

```text
D_rel(p) = fx * tau_1x / (mu_1x(p) * s)
```

This is the direct geometric test.
It is the most honest validation because it adds no decoder.

Level B: learned depth probe

- freeze the encoder
- operate on `C_1(p, .)` rather than on `F_t(p)` alone
- use a shallow probe or pointwise decoder to predict depth

If a shallow probe works, geometric information is already present in the correlation representation.

### 4.5 Control Comparisons

Control 1: random frozen encoder

- same correlation and probe pipeline
- encoder is untrained

Control 2: single-frame feature probe

- depth probe sees `F_t(p)` alone
- no cross-frame correlation

Interpretation:

- the motion-trained correlation probe should outperform both controls
- if not, the claimed geometry signal is weak or shortcut-driven

## 5) Active Experiment Plan

The active experiment bundle for this setting is:

1. generate a three-frame, left-right-only two-ball dataset
2. train the three-frame self-supervised model with per-point consistency loss
3. evaluate Stage 1 training behavior
4. run claim-wise validation
5. train a shallow correlation-based depth decoder
6. compare against:
   - random frozen encoder
   - single-frame feature probe

## 6) Decision Logic

Case 1: training loss does not decrease and motion visualizations collapse

- the pipeline is not yet learning stable motion-sensitive features
- do not make geometry claims

Case 2: training succeeds but Level A and Level B both fail

- correspondence is not precise enough for geometry
- redesign the supervision or stabilize the correlation module

Case 3: Level B works but Level A fails badly

- the representation contains usable geometric information
- the direct analytic inversion is still too unstable

Case 4: Level A, Level B, and controls all support the same result

- this is the strongest evidence that motion-only supervision induced geometry-relevant structure

## 7) Shared Scene and Data Assumptions

- two balls with different radii
- one table and one background wall
- static world; only camera motion changes
- image resolution `128 x 128`
- intrinsics `fx = fy = 100.0`, `cx = cy = 64.0`
- current motion scope: pure left-right translation only

Per-sample triplet data:

- `img_t.png`
- `img_t1.png`
- `img_t2.png`
- `meta.json`

`meta.json` includes:

- `K`
- `T_t_to_t1`
- `T_t1_to_t2`
- `world_to_camera_t`
- `world_to_camera_t1`
- `world_to_camera_t2`
- 2D and 3D ball centers in all three frames

## 8) Current Success Criterion

The current root question gets initial support only if all of the following happen:

1. the self-supervised three-frame loss decreases clearly
2. the motion field and correspondence visualizations are geometrically coherent
3. direct geometric depth recovery correlates with ground truth on valid regions
4. the correlation-based depth probe beats both:
   - random frozen encoder
   - single-frame feature probe

Before that, conclusions remain partial.
