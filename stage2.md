# Stage 2

> Stage 2 tests whether the motion-trained representation contains local geometry-relevant information.
> The first probe uses the Stage 1 Experiment 2 backbone and asks whether ball-center depth can be decoded from the frozen local feature.

## Why Stage 2

Stage 1 only showed that adjacent frames contain decodable motion signal.
It did not show that the local feature at the ball center contains geometry.

Stage 2 moves to the first geometry-facing question:

- after motion training, does the frozen local feature support depth decoding?

## Shared Setup

Backbone source:

- use the pretrained encoder from `stage1_exp2.py`
- load it from a Stage 1 Experiment 2 checkpoint
- freeze the encoder during probe training

Local feature:

- take the current-frame feature map `F_t`
- locate the ball-center pixel in `img_t`
- map that pixel to the `8 x 8` feature grid
- extract the local feature vector at that location

Shared purpose:

- test whether the motion-trained local representation contains decodable geometry

## Experiment 1

Task:

- decode ball-center depth from the frozen local feature

Input:

- `I_t`
- ball-center pixel location in frame `t`

Output:

- `z_t`

Target:

- `ball_center_3d_t[2]`

Network:

- frozen encoder from Stage 1 Experiment 2
- local feature extraction at the ball-center position
- lightweight MLP probe

Loss:

- L1 loss on depth

What it verifies:

- whether depth is present in the local motion-trained feature

## Stage 2 Status

Experiment 1 status:

- designed
- code implemented in `stage2_exp1.py`
- results not yet filled in

What a positive result would mean:

- the motion-trained backbone contains local depth-relevant information at the ball-center feature

What it would still not prove:

- full 3D geometry has emerged
- the effect is necessarily causal without later controls
