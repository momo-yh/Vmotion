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

## 3) Updated Implementation Settings

The implementation has five parts:

- a shared encoder
- a 1D horizontal correlation module
- soft displacement recovery
- depth-based third-frame prediction
- a visible-only hard physical mask and two loss terms (`L_geom` + `L_sharp`)

### 3.1 Region Encoder

Given three adjacent frames, a shared encoder produces:

```text
F_t, F_t+1, F_t+2 in R^(B x d x H' x W')
```

with:

- `s = 4`
- `H' = H / s`
- `W' = W / s`
- `d = 64`

Encoder architecture:

- `Conv 3 -> 32`, `3x3`, stride `1`, ReLU
- `Conv 32 -> 64`, `3x3`, stride `1`, ReLU
- `Conv 64 -> 64`, `3x3`, stride `2`, ReLU

Design reasons:

- three layers give a meaningful local receptive field
- `3x3` kernels preserve local spatial context
- stride `2` on layers 2 and 3 reduces correlation cost
- shared weights enforce temporal consistency across frames

### 3.2 1D Horizontal Correlation Module

For each position `p` in the feature map, define a 1D horizontal search window:

```text
N_r^1D(p) = { q : |q_x - p_x| <= r, q_y = p_y }
```

This restriction is required by the motion scope.
Under left-right translation:

```text
Delta v(p) ~= 0
```

Vertical search therefore introduces physically irrelevant candidates,
flattens the softmax, and corrupts the displacement estimate.

Precondition:

```text
r >= max_p |Delta u(p)| / s
```

The matching score uses cosine similarity with fixed temperature scaling:

```text
C_1(p, dx) = <F~_t(p), F~_t+1(p + dx x_hat)> / tau
C_2(p, dx) = <F~_t+1(p), F~_t+2(p + dx x_hat)> / tau
```

where:

- `F~` denotes L2-normalized features
- `tau = 0.07` is fixed and not learned

Both correlation volumes therefore have shape:

```text
[B, H', W', (2r+1)]
```

### 3.3 Soft Displacement and Depth Recovery

Step 1: soft displacement from the two correlation strips

```text
a_1(p, dx) = softmax_dx C_1(p, dx)
a_2(p, dx) = softmax_dx C_2(p, dx)

mu_1x(p)   = sum_dx a_1(p, dx) * dx
mu_2x(p)   = sum_dx a_2(p, dx) * dx
```

This is a differentiable approximation of the matching peak.
If `a_1(p, .)` is flat, then `mu_1x(p)` stays near zero and the point is later
rejected by the hard mask.

Step 2: recover relative depth from the first frame pair

```text
Delta u(p) = mu_1x(p) * s
D_rel(p)   = fx * tau_1x / (Delta u(p) + eps)
```

with clipping to `[D_min, D_max]`.

This uses only the displacement-depth relation.
No depth label is used anywhere in the pipeline.

### 3.4 Third-Frame Prediction and Hard Mask

Step 3: predict the position in frame `t+2`

For the current left-right-only setting:

```text
q_pred_x(p) = p_x + fx * tau_2x / (D_rel(p) * s)
q_pred_y(p) = p_y
```

This is exact under pure horizontal translation.

Step 4: actual position from `C_2`

```text
q_actual_x(p) = p_x + mu_2x(p)
q_actual_y(p) = p_y
```

Step 5: hard physical mask

Only points that are still visible in frame `t+2` are supervised.

Condition: the point is still visible in frame `t+2`

```text
q_pred(p) in Omega'
```

If the point moves outside the field of view, then there is no valid
observation to compare against.

The hard mask is therefore:

```text
m(p) = 1 if visible condition holds, else 0
```

### 3.5 Training Objective

Each training triplet must satisfy a minimum measurable displacement at the
scene center:

```text
|Delta u_center| >= 8 px
```

Samples below this threshold are excluded from training.
This keeps `D_rel` numerically stable across the dataset.

Per-point geometric loss:

```text
L_point(p) = (q_pred_x(p) - q_actual_x(p))^2
```

Geometric term:

```text
L_geom = mean over { p : m(p) = 1 } of L_point(p)
```

Sharpness regularizer (using both `a_1` and `a_2`):

```text
H1(p) = - sum_dx a_1(p,dx) * log(a_1(p,dx) + eps)
H2(p) = - sum_dx a_2(p,dx) * log(a_2(p,dx) + eps)

H1_norm(p) = H1(p) / log(2r+1)
H2_norm(p) = H2(p) / log(2r+1)

L_sharp = mean over { p : m(p) = 1 } of (H1_norm(p) + H2_norm(p)) / 2
```

Total loss:

```text
L_total = L_geom + lambda_sharp * L_sharp
```

with a small default such as `lambda_sharp = 0.02`.

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

1. generate a three-frame, left-right-only dataset
2. verify the two preconditions before training:
   - minimum sample displacement is at least `8 px`
   - search radius covers the maximum horizontal displacement
3. train the self-supervised three-frame model with:
   - shared encoder
   - 1D horizontal correlation
   - soft displacement recovery
   - hard two-condition mask
   - one per-point loss term
4. evaluate Stage 1 training behavior
5. run claim-wise validation
6. train a shallow correlation-based depth decoder
7. compare against:
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

The setting is intentionally independent of any one visual scene template.
Different datasets may use different object categories or layouts, as long as
they respect the motion scope and preconditions.

Shared assumptions:

- static world; only camera motion changes
- image resolution `128 x 128`
- intrinsics `fx = fy = 100.0`, `cx = cy = 64.0`
- current motion scope: pure left-right translation only
- supervision uses no depth label during training

Per-sample triplet data:

- `img_t.png`
- `img_t1.png`
- `img_t2.png`
- `meta.json`

`meta.json` includes at minimum:

- `K`
- `T_t_to_t1`
- `T_t1_to_t2`
- `world_to_camera_t`
- `world_to_camera_t1`
- `world_to_camera_t2`

Optional dataset-specific metadata may include:

- object centers
- object sizes
- object categories
- precomputed depth maps for validation

## 8) Updated Success Criterion

The root question gets support only if all of the following happen:

1. the per-point loss decreases clearly on the masked valid set
2. the 1D displacement field is spatially coherent and depth-aligned
3. direct geometric depth recovery correlates with ground truth on valid regions
4. the correlation-based depth probe beats both:
   - random frozen encoder
   - single-frame feature probe

Criterion 4 is the decisive test.
The previous result showed the single-frame probe winning.
The final cleaned setting is intentionally minimal so that any failure can be
attributed directly to correspondence quality rather than to extra loss terms.
