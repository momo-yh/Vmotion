# Stage 1

> Stage 1 is a representation sanity check.
> It tests whether adjacent-frame visual input contains usable motion signal and whether the current two-frame architecture can decode that signal reliably.

## Why Stage 1

Before claiming anything about local geometry, we first need to verify a simpler prerequisite:

- the two-frame input really contains learnable motion information
- the current Siamese no-GAP architecture can use that information

If Stage 1 fails, geometry claims are not justified.
If Stage 1 succeeds, it becomes reasonable to continue to later geometry probes.

## Shared Setup

Input:

- adjacent RGB frames `I_t` and `I_t+1`

Data:

- `img_t.png`
- `img_t1.png`
- `meta.json` with `T_t_to_t1`, `ball_center_2d_t`, `ball_center_2d_t1`

Shared model idea:

- two frames are encoded separately with a shared CNN encoder
- fused feature uses `[F_t, F_t+1, F_t+1 - F_t]`
- no global average pooling
- regression is done from spatial feature maps

Shared purpose:

- verify that cross-frame motion signal is present and decodable

## Experiment 1

Task:

- decode ball-center 2D image-plane displacement

Input:

- `I_t`, `I_t+1`

Output:

- `(du, dv)`

Target:

- `du = u_t+1 - u_t`
- `dv = v_t+1 - v_t`

Network:

- shared encoder
- explicit feature differencing
- no GAP
- 2D regression head

Loss:

- L1 loss on `(du, dv)`

What it verifies:

- whether the model can decode a minimal observable motion quantity from two frames

## Experiment 2

Task:

- decode 3D camera translation between adjacent frames

Input:

- `I_t`, `I_t+1`

Output:

- `(tx, ty, tz)`

Target:

- `T_t_to_t1[:3, 3]`

Network:

- same backbone structure as Experiment 1
- output dimension changed from 2 to 3

Loss:

- L1 loss on `(tx, ty, tz)`

What it verifies:

- whether the model can decode a stronger 3D motion quantity from two frames

Note:

- in the current dataset this is 3D camera translation, not full 6-DoF pose recovery

## Stage 1 Conclusions

Experiment 1 conclusion:

- success
- test MAE is about `0.333 px`
- the model can reliably decode 2D image-plane displacement from two adjacent frames

Experiment 2 conclusion:

- success
- test MAE is about `0.00651 m`
- the model can reliably decode 3D camera translation from two adjacent frames

Overall Stage 1 conclusion:

- the current two-frame visual setup contains strong learnable motion signal
- the current Siamese no-GAP architecture can decode both a minimal 2D motion target and a stronger 3D translation target
- this supports moving on to later geometry-focused experiments
- this still does not prove that local geometry has emerged in the ball-center feature
