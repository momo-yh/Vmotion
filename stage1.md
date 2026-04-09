# Minimal Setting: Does Motion-Only Induce Geometry-Relevant Local Features?

## 1. Question

We test the following minimal question:

> If a model is trained only with monocular image pairs and known ego-motion, without any direct depth supervision, can its local visual feature at a queried image point support depth decoding?

The goal is **not** to train a depth predictor for the object directly.  
The goal is to test whether **motion-only supervision** makes local features become **geometry-relevant**.

---

## 2. Training Inputs

For each sample, training uses only:

- image at time `t`: `I_t`
- image at time `t+1`: `I_{t+1}`
- known camera motion: `T_{t->t+1}`
- camera intrinsics: `K`

Training does **not** use:

- ground-truth depth
- ground-truth 3D point
- object labels
- segmentation masks
- image reconstruction loss

---

## 3. Scene

Use the simplest static scene:

- one ball
- one table
- one background wall
- only the camera moves
- the ball stays fixed

This keeps the environment simple enough for a first existence test.

---

## 4. Encoder

Train an encoder to produce a spatial feature map:

- `z_t = E(I_t)`
- `z_{t+1} = E(I_{t+1})`

and a depth-like prediction:

- `d_t = D(z_t)`

Here `d_t` is **not supervised directly**.  
It is only used to construct motion-based geometric warping.

---

## 5. Motion-Only Training Objective

Training uses only image pairs, known camera motion, and intrinsics.

For sampled image points `p` in frame `t`:

1. predict a depth-like value `d_t(p)`
2. back-project `p` into 3D
3. transform it with the true camera motion `T_gt`
4. project it into frame `t+1` to get `p'_pos`
5. also transform it with an incorrect motion `T_neg`
6. project it into frame `t+1` to get `p'_neg`

Formally:

`X_t(p) = d_t(p) * K^{-1} * p_tilde`

`p'_pos = pi(T_gt * X_t(p))`

`p'_neg = pi(T_neg * X_t(p))`

Define:

`e_pos(p) = || z_t(p) - z_{t+1}(p'_pos) ||_1`

`e_neg(p) = || z_t(p) - z_{t+1}(p'_neg) ||_1`

Training loss:

`L_motion = (1 / M) * sum_{p in P} max(0, m + e_pos(p) - e_neg(p))`

where `P` is a set of sampled valid image points.

This loss enforces that the true camera motion should explain local cross-frame feature consistency better than an incorrect motion.

No depth supervision, reconstruction loss, semantic labels, or object annotations are used.

---

## 6. What Is Learned During Training

During training, the model is never told:

- where the ball center is
- what the ball depth is
- what the 3D coordinate is

So the training stage does **not** directly optimize the queried-point depth task.

This is important:  
it keeps the test focused on whether geometry-relevant information **emerges** in the local feature.

---

## 7. Downstream Test

After motion-only training:

1. freeze the encoder `E`
2. take the ball-center pixel `p_ball`
3. sample the local feature `z_t(p_ball)`
4. train a small readout MLP to predict the ball-center depth or 3D position

Two possible readouts:

- **Depth readout:** `z_t(p_ball) -> d_ball`
- **3D readout:** `z_t(p_ball) -> X_ball = (x, y, z)`

The readout is the only place where supervision on the queried point is allowed.

---

## 8. Success Criterion

If the frozen local feature at the queried point supports accurate depth / 3D decoding, then motion-only training has induced **geometry-relevant local information**.

A careful interpretation is:

- this does **not** yet prove the feature is a full geometric representation
- but it does show that motion-only supervision can make local features contain usable geometric information

---

## 9. Minimal Baseline

Compare against:

- a randomly initialized frozen encoder + same readout MLP

If motion-trained features decode depth much better than random features, that is the first evidence that motion supervision induced useful local geometry-related structure.

---

## 10. Minimal Claim

This setting tests:

> whether motion-only supervision can make local features at a queried image point become informative enough to decode that point's depth / 3D.

It does **not** yet test:

- full scene depth estimation
- full 3D reconstruction
- object-level depth supervision
- geometry quality everywhere in the image