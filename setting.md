# Setting (Single Source of Truth)

> This file defines the project-level research question, the claim decomposition, the experimental tree, and the current active experiment.
> The current Stage 1 summary is in `stage1.md`.
> The current Stage 2 summary is in `stage2.md`.
> Run commands are in `README.md`.

## 1) Root Question

Can motion-only supervision using only:

- adjacent monocular frames `I_t, I_t+1`
- known camera motion `T_t_to_t1`
- intrinsics `K`

induce local visual features that contain decodable geometric information such as ball-center depth or 3D position?

## 2) Root Claim Decomposition

To support the root claim, the following subclaims must hold.

### A. Motion signal is learnable from the visual input

The training setup must contain enough usable signal linking image change to camera motion.

### B. The backbone must encode local cross-frame structure, not only global shortcuts

The learned representation must preserve local information relevant to correspondence and displacement.

### C. The learned local feature at the ball-center pixel must contain geometry-relevant information

After freezing the backbone, a lightweight probe should decode depth or 3D coordinates better than weak baselines.

### D. The decoded geometry must come from motion-induced structure, not accidental dataset shortcuts

The result must survive basic controls and ablations.

## 3) Experimental Tree

### Stage 1: Representation Sanity Check

### Q1. Can the model recover a simple observable motion quantity from two frames?

This is a diagnostic question, not the root question.

### Experiment 1

Train a model to predict ball-center 2D displacement:

- input: `I_t, I_t+1`
- target: `(du, dv)`

### Purpose

Check whether the architecture can use cross-frame visual change at all.

### Interpretation

- If this fails badly, the current architecture or training setup is not yet suitable.
- If this fails badly, do not proceed to geometry claims.
- If this succeeds, motion-related signal is present.
- If this succeeds, continue to more geometry-relevant tests.

### Auxiliary Motion Baseline: Experiment 2

Train a model to predict 3D camera translation:

- input: `I_t, I_t+1`
- target: `T_t_to_t1[:3, 3] = (tx, ty, tz)`

Purpose:

- test a stronger global motion-decoding target than Experiment 1
- compare 2D image-plane displacement decoding against 3D camera translation decoding

Interpretation:

- If this succeeds, adjacent frames support a stronger 3D motion target.
- If this fails while Experiment 1 succeeds, the current setup supports 2D motion decoding more readily than 3D camera translation decoding.
- Even if this succeeds, it still does not prove local geometry emergence.

### Stage 2: Local Feature Emergence

### Q2. Does the backbone learn a local feature that is useful at the ball-center pixel?

This is the first real bridge toward the root question.

### Stage 2 Experiment 1

Train the backbone with motion-only supervision. Then freeze it. At the ball-center pixel, extract the local feature and train a lightweight probe to predict:

- depth `z_t`

### Purpose

Test whether geometry-relevant information is present in the local representation.

### Interpretation

- If the frozen-feature probe clearly beats weak baselines, there is initial evidence that geometry-relevant local information emerged.
- If not, either geometry did not emerge.
- If not, the training task may have encouraged only shallow motion shortcuts.

### Stage 3: Causal Role of Motion Supervision

### Q3. Is the geometry information specifically induced by motion supervision?

This separates "representation contains geometry" from "geometry happened accidentally."

### Experiment 3A: Random-Feature Baseline

Compare frozen-probe performance against:

- random initialized backbone plus the same probe

### Experiment 3B: Weak Visual Baseline

Compare against:

- single-frame backbone
- or a backbone trained on an image-only nuisance target

### Experiment 3C: Motion Corruption Control

Break the motion supervision by:

- shuffling `T_t_to_t1` across samples
- or mismatching the second frame and motion label

### Purpose

Check whether performance depends on correct motion structure.

### Interpretation

- If correct motion supervision is clearly better than corrupted motion supervision, this supports a causal role for motion.
- If performance barely changes, the apparent signal likely comes from shortcut statistics.

### Stage 4: What Kind of Geometry Emerged?

### Q4. Is the representation only learning 2D correspondence-like cues, or actual 3D geometry?

This is a level-of-abstraction test.

### Experiment 4A

Probe only depth `z_t`.

### Experiment 4B

Probe full 3D point `(x_t, y_t, z_t)`.

### Experiment 4C

Compare probe difficulty across:

- depth only
- lateral coordinates only
- full 3D

### Purpose

Determine whether the feature contains:

- only image-plane displacement information
- relative depth structure
- or richer metric 3D information

### Interpretation

- Good depth but weak full 3D suggests the representation may carry partial geometry.
- Good full 3D gives stronger evidence of metric local geometry.

### Stage 5: Robustness and Anti-Shortcut Checks

### Q5. Is the result robust across nuisance variation?

A positive result is weak if it works only under one narrow configuration.

### Experiment 5A

Vary:

- ball position
- camera height
- camera azimuth
- camera distance

### Experiment 5B

Change appearance:

- background texture
- ball color or texture
- table appearance

### Experiment 5C

Train and test with splits by nuisance factors, not just random splits.

### Purpose

Check whether the learned feature is geometry-related rather than memorizing appearance correlations.

### Interpretation

- Strong generalization supports a real geometric mechanism.
- Sharp failure under nuisance shift suggests shortcut learning.

## 4) Decision Tree

### Case 1: Stage 1 fails

Meaning:

- the architecture cannot reliably use cross-frame motion signal yet

Next:

- improve the architecture for cross-frame comparison
- do not claim anything about geometry

### Case 2: Stage 1 succeeds, Stage 2 fails

Meaning:

- the model can solve a simple motion task
- but local geometry is not yet encoded in the frozen feature

Next:

- redesign the motion supervision objective
- strengthen locality or correspondence bias
- test whether the current task is too weak or too global

### Case 3: Stage 1 succeeds, Stage 2 succeeds, Stage 3 fails

Meaning:

- the feature appears useful
- but not because of the intended motion mechanism

Next:

- diagnose shortcuts
- redesign controls and supervision

### Case 4: Stage 1 succeeds, Stage 2 succeeds, Stage 3 succeeds

Meaning:

- motion supervision likely induces useful local geometric information

Next:

- move to stronger geometry probes
- test robustness and limits

## 5) Minimal Recommended Sequence

1. **Experiment 1**: 2D displacement recovery from two frames. Goal: verify usable cross-frame signal.
2. **Stage 2 Experiment 1**: Freeze the backbone and probe ball-center depth. Goal: test whether local geometry is decodable.
3. **Experiment 3**: Compare with random, single-frame, and corrupted-motion controls. Goal: test whether motion is causally responsible.
4. **Experiment 4**: Compare depth vs full 3D decoding. Goal: determine what geometric content actually emerged.
5. **Experiment 5**: Test robustness under nuisance shifts. Goal: rule out shortcut explanations.

## 6) What Experiment 1 Actually Tells Us

Experiment 1 does **not** answer the root question.

It only answers:

> Can the current architecture learn a simple observable cross-frame motion quantity?

So Experiment 1 is a gatekeeping sanity check.

Passing it means:

- the architecture is not obviously broken
- motion signal is accessible
- proceeding to local geometry probing is justified

Failing it means:

- the current setup is not ready for geometry claims

## 7) Core Success Logic for the Whole Project

The root question gets initial support only if all of the following happen:

1. the backbone can use motion signal
2. the frozen local feature supports depth or 3D decoding
3. this advantage disappears or weakens under motion-corruption controls
4. the result survives nuisance variation

Only then can we say:

> Motion-only supervision likely induced a geometry-relevant local representation.

Before that, all conclusions should remain partial.

## 8) Shared Scene and Data Assumptions

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

## 9) Current Active Experiment

The current active stage is **Stage 1**.

Stage 1 includes two experiments:

- Experiment 1: 2D ball-center displacement decoding
- Experiment 2: 3D camera translation decoding

Experiment 1 target:

- `du = ball_center_2d_t1[0] - ball_center_2d_t[0]`
- `dv = ball_center_2d_t1[1] - ball_center_2d_t[1]`

Experiment 2 target:

- `T_t_to_t1[:3, 3] = (tx, ty, tz)`

Shared architectural constraint:

- shared encoder for `img_t` and `img_t1`
- explicit feature fusion with `[F_t, F_t1, F_t1 - F_t]`
- no global average pooling
- regression from spatial feature maps

Shared training constraint:

- use only the task supervision defined for each experiment
- do not add ground-truth depth supervision
- do not add semantic or segmentation labels
- do not add reconstruction losses

## 10) Current Minimal Success Criterion

If Stage 1 can stably solve its motion-decoding tasks on held-out data with clearly non-trivial error reduction relative to weak baselines, then the project may proceed to local geometry probing.

If Stage 1 fails, it is not justified to move on to later geometry claims.
