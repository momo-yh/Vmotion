The latest implementation should match the pseudocode below exactly.
This is the final cleaned version:

- 1D horizontal correlation only
- no learned mask
- no consistency loss term in the objective
- visible-only hard mask
- per-point loss + sharpness loss on both a1 and a2

============================================================
MINIMAL THREE-FRAME PIPELINE
Left-right translation only.
Final cleaned implementation version.
============================================================

INPUT
  I_t, I_t+1, I_t+2    adjacent RGB frames      [B, 3, H, W]
  tau1_x, tau2_x       known horizontal translations
  K                    known camera intrinsics
  s = 4                encoder downsampling factor

PRECONDITIONS
  dataset level: min sample displacement >= 8 px
  search radius: r >= max_horizontal_displacement / s

------------------------------------------------------------
STAGE 1 - Shared Encoder
  F_t  = CNN(I_t)      [B, 128, H', W']
  F_t1 = CNN(I_t+1)    [B, 128, H', W']
  F_t2 = CNN(I_t+2)    [B, 128, H', W']

  Shared weights across all three frames.
  Feature maps are L2-normalized before correlation.

------------------------------------------------------------
STAGE 2 - 1D Horizontal Correlation
  For each feature position p and offset dx in {-r, ..., +r}:

    C1[p, dx] = dot(F_t[p],  F_t1[p + dx * x_hat]) / temp
    C2[p, dx] = dot(F_t1[p], F_t2[p + dx * x_hat]) / temp

  temp = 0.07

  Tensor shape in code:
    C1, C2 in [B, 2r+1, H', W']

------------------------------------------------------------
STAGE 3 - Soft Displacement
  a1(p, dx) = softmax over dx of C1[p, dx]
  a2(p, dx) = softmax over dx of C2[p, dx]

  mu1_x(p) = sum_dx a1(p, dx) * dx
  mu2_x(p) = sum_dx a2(p, dx) * dx

  Delta_u(p) = mu1_x(p) * s

------------------------------------------------------------
STAGE 4 - Relative Depth Recovery
  signed_eps = eps * sign(tau1_x)

  D_rel_raw(p) = fx * tau1_x / (Delta_u(p) + signed_eps)
  D_rel(p)     = clamp(D_rel_raw(p), D_min, D_max)

------------------------------------------------------------
STAGE 5 - Predict Position in Frame t+2
  q_pred_x(p) = p_x + mu1_x(p) * tau2_x / (tau1_x + signed_eps)
  q_pred_y(p) = p_y

  This is algebraically equivalent to:
    p_x + fx * tau2_x / (D_rel(p) * s)
  under pure horizontal translation.

------------------------------------------------------------
STAGE 6 - Actual Position from C2
  q_actual_x(p) = p_x + mu2_x(p)
  q_actual_y(p) = p_y

------------------------------------------------------------
STAGE 7 - Hard Mask
  Condition: point is visible in frame t+2

    q_pred_x(p) in [0, W' - 1]
    q_pred_y(p) in [0, H' - 1]

  Mask:

    m(p) = 1 if visible, else 0

------------------------------------------------------------
STAGE 8 - Losses
  L_point(p) = (q_pred_x(p) - q_actual_x(p))^2

  L_geom =
      sum_p m(p) * L_point(p)
      -----------------------
      sum_p m(p) + eps

  H1(p) = -sum_dx a1(p, dx) * log(a1(p, dx) + eps)
  H2(p) = -sum_dx a2(p, dx) * log(a2(p, dx) + eps)

  H1_norm(p) = H1(p) / log(2r + 1)
  H2_norm(p) = H2(p) / log(2r + 1)

  L_sharp =
      sum_p m(p) * (H1_norm(p) + H2_norm(p)) / 2
      -------------------------------------------
      sum_p m(p) + eps

  L_total = L_geom + lambda_sharp * L_sharp

============================================================
FULL FORWARD PASS
============================================================

  F_t, F_t1, F_t2 = Encoder(I_t, I_t+1, I_t+2)
  C1, C2          = Corr1D(F_t, F_t1, F_t2, radius=r, temp=0.07)
  a1, a2          = Softmax(C1, C2)
  mu1_x, mu2_x    = SoftDisplacement(a1, a2)
  D_rel           = DepthRecovery(mu1_x, tau1_x, fx, s)
  q_pred          = PredictPosition(mu1_x, tau1_x, tau2_x)
  q_actual        = ActualPosition(mu2_x)
  m               = HardMask(q_pred)
  L_geom          = PerPointLoss(q_pred, q_actual, m)
  L_sharp         = SharpnessLoss(a1, a2, m)
  L_total         = L_geom + lambda_sharp * L_sharp

============================================================
INFERENCE
============================================================

  Given I_{t-1}, I_t, tau_x, K:

    F_{t-1}, F_t = Encoder(I_{t-1}, I_t)

    For each p and dx:
      C[p, dx] = dot(F_{t-1}[p], F_t[p + dx * x_hat]) / temp

    a(p, dx) = softmax over dx of C[p, dx]
    mu_x(p)  = sum_dx a(p, dx) * dx
    D(p)     = fx * tau_x / (mu_x(p) * s)

  Output:
  dense depth map of the current frame, using motion history only.
