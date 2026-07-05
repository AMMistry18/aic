# Reward Function Spec — Residual SAC for Last-Inch Cable Insertion

**Status:** historical. `RL/reward.py` now uses the simpler geometry-first
reward described in `RL/README.md` §3.3: depth progress + sparse success, with
small alignment/contact/action safety costs and image distance default-off. The
older spec below is kept only as design history.

**Previous status:** v0.3 (2026-07-03) — recalibrated to the **fixed** MuJoCo scene.
v0.2 → v0.3 changes (these SUPERSEDE the older numbers below where they conflict):
- **Scene was broken** (solid brick, no cavity, force always 0 → success could
  never fire). Now a real open-top socket; contact force reads **0–8 N**
  (seating 3–5 N). See §11 (rewritten) and the `rl-env-scene-fix` note.
- **Killed the double +50 bonus**: `β_s = 0` (image bonus off); the single
  success bonus lives in `r_done` (+50). v0.2 paid +100 at the success step.
- **Force thresholds recalibrated** to the measured band: `f_low=0.5`,
  `f_optimal=4.0`, `f_max=8.0 N` (were 2/10/20, never realized in-sim). Abort
  `f_abort=15`, `f_linger=10` (were 24/16). Positive force branch gated on
  `depth_norm>0.3` so it can't be farmed at the entrance.
- **Added `r_depth`**: potential-based insertion-depth progress (`w_depth=2.0`),
  the primary low-variance last-inch signal (Ng et al. 1999).
- **Fixed numerically-dead scaling** of `r_xy` (normalise by 5 mm) and
  `r_lateral` (per-N), which were ~1e-4 in metres.
- Reverse curriculum start now interpolates plug z from **near-seated** (level 0)
  up to **above the mouth** (level 1) — v0.2 wrongly started at the entrance for
  all levels.

---

**Status (historical):** v0.2 — tuned for **arbitrary-start last-inch** regime.
v0.1 → v0.2 changes:
- Episode structure now starts the plug from random poses inside the last-inch box (not from a fixed handoff pose).
- `r_xy` down-weighted so the policy is not incentivised to sit at port centre.
- Image-success threshold tightened; force shaping unchanged.
- Added section §11: MuJoCo training scene + port randomization.

---

## 1. Setting

| | |
|---|---|
| **Task** | Insert SFP / SC plug into port, last-inch only (handed off near the port) |
| **Approach** | Residual RL: `u_total = u_pi_H + u_residual`, where `π_H = PerceptionInsert` and `u_residual ~ π_θ(s)` is learned via SAC |
| **π_θ obs** | 3 wrist cameras stacked + current F/T + TCP pose + tip–port XY error + last residual action |
| **π_θ act** | 6D pose delta added to `π_H`'s command (mx, my, mz, drx, dry, drz) |
| **Scoring goal** | Maximize Phase-1 AIC `Tier 2 + Tier 3` (smoothness + success), avoid Tier 2 force penalty (>20 N) and Tier 2 off-limit contacts |

We want `π_θ` to fix what `π_H` gets wrong on the last 5–15 mm of insertion, where perception noise + tolerance stack up. The hand controller already reaches the port entrance; the residual learns small corrective patterns.

---

## 2. Reward components

Six additive components. All are dense (computed every step) **except** `r_image_l1` and `r_done`, which form the main signal.

Let `f_z` = vertical (insertion-axis) force at the wrist, `f_xy = ‖f_xy‖`, `tip_xy = plug tip XY`, `port_xy = port entrance XY`, `a_t = u_residual_t`.

### 2.1 Sparse image reward (main signal, ~±50 / step)

```
r_image_l1 = -α · ‖I_t − I_goal‖_1 / (H·W)
r_image_done = +β_s   if ‖I_t − I_goal‖_1 < ε   else 0
r_image = r_image_l1 + r_image_done
```

Defaults: `α = 1.0`, `β_s = 50.0`, `ε = 0.02`, `H=W=32`.

**Why this is the primary signal:** it directly measures visual progress (the only thing that correlates with Tier-3 success in the cloud scoring). See Schoettler et al. 2019 — sparse goal-image L1 worked well on USB/D-Sub/Model-E connectors.

**Risk:** pure pixel L1 can be "hacked" (e.g. moving in front of the port to darken pixels). P_H's impedance controller in the residual action prevents this because it locks the tip XYZ near the port entrance.

### 2.2 Piecewise logarithmic force reward (auxiliary shaping, ~±2 / step)

Goal: encourage some contact force (so friction is overcome) without exceeding the 20 N force penalty threshold. Force is a *scalar* signal — easy to log-shape without dominating the image reward.

```
f_low     = 2.0    N    # below = not enough contact
f_optimal = 10.0   N    # sweet spot for SC/SFP seating
f_max     = 20.0   N    # above this = scoring penalty territory
β_low, γ  = 0.5, 0.5

if 0 ≤ f_z < f_low:
    r_force = β_low · log(1 + f_z)
elif f_low ≤ f_z ≤ f_optimal:
    r_force = β_low · log(1 + f_low)        # plateau, just contact
              + γ · (f_z − f_low)           # linear ramp to optimal
elif f_optimal < f_z < f_max:
    r_force = β_low · log(1 + f_low)
              + γ · (f_optimal − f_low)     # ceiling
              − γ · (f_z − f_optimal)       # linear decline
else:  # f_z ≥ f_max
    r_force = β_low · log(1 + f_low)
              + γ · (f_optimal − f_low)
              − γ · (f_max − f_optimal)
              − log²(1 + (f_z − f_max))     # log-square penalty
```

Plot sketch (f_z → r_force, peaking around 10 N, log-square drop after 20 N):

```
       f_z →  0    2   5   10   15   20   25   30  N
                ────────────────
   r_force     0  0.35 0.7  1.0  0.5  0  -1.8 -3.6
                              ↑ plateau    ↑ drops off
                              peak
```

**Why this shape:** piecewise linear is fine but a log penalty on the runaway-force tail avoids the discontinuity slope that sharp caps create. The log-square (`-log²(1+Δ)`) rises steeply but stays bounded near the AIC force limit.

### 2.3 Tip-port XY progress (auxiliary, ~±2 / step)

```
r_xy = -δ_xy · ‖tip_xy − port_xy‖_2
δ_xy = 0.5
```

Once within 1 mm of port, this saturates near zero. Helps bootstrap SAC during the first ~5k steps when the image reward is still noisy and π_θ hasn't learned to read camera features.

### 2.4 Action smoothness (residual regularization, ~±0.5 / step)

```
r_action = -δ_a · ‖a_t − a_{t-1}‖_2
δ_a = 0.1
```

Penalizes jerky residual corrections. Critical because Tier 2 scores linear jerk — A 30 m/s³ can wipe most of the 6 points.

### 2.5 Lateral force penalty (small, ~±0.5 / step)

```
r_lateral = -δ_lat · f_xy
δ_lat = 0.02
```

Lateral forces don't help insertion, they bend cables. Small linear penalty so π_θ doesn't accidentally push sideways.

### 2.6 Termination bonus / penalty (~one-shot)

```
r_done = {
    +50.0   if insertion verified (Tier-3 success)
    -30.0   if AIC 24 N safety abort triggered
    -10.0   if off-limit contact (enclosure/taskboard)
     0.0   otherwise
}
```

---

## 3. Total reward

```
r_t = r_image                                       # main, ±50
    + 0.05 · r_force                                 # auxiliary ±2
    + 0.02 · r_xy                                    # bootstrap ±1 (down-weighted for arbitrary-start)
    + 0.10 · r_action                                # smoothness ±0.5
    + 0.10 · r_lateral                               # small ±0.5
    + r_done                                          # ±50 one-shot
```

**Calibration rule:** the *image reward must be the dominant gradient* — otherwise π_θ will happily stay near the port and ride the force reward without actually inserting. The 0.05 / 0.05 / 0.10 / 0.10 multipliers make auxiliary signals ≤ 4 % of image magnitude.

If after a few runs the policy is **not making progress** (image distance stagnant) but stays alive (no force penalty), increase `α` and/or decrease `r_force`'s weight. If it is making progress but **force oscillates**, increase `δ_lat` and `δ_a`.

---

## 4. Observation (state) design

State passed to the SAC policy network each step. Image + small aux vector:

| Component | Shape | Source | Notes |
|---|---|---|---|
| `image_curr` | `(H, W, 3)` uint8 | left/center/right wrist cameras, 32×32 grayscale, stacked channel-last | Resized from native 1152×1024 |
| `image_goal` | `(H, W, 3)` uint8 | captured once per episode when cable is at fully-inserted pose | Refreshed on every env reset |
| `force_z` | scalar float | `obs.wrist_wrench.wrench.force.z` | Forwarded as-is |
| `force_xy_norm` | scalar float | `‖[fx, fy]‖` | Pre-computed in the env |
| `tcp_pose_err` | `(6,)` | `[Δxyz_port (3), axis-angle of q_rel (3)]` — pose of `gripper/tcp` minus target port entrance pose, in the port frame | CheatCode-style TF lookup |
| `tip_xy_err` | `(2,)` | plug tip XY minus port entrance XY | From `PerceptionInsert` `_plug_tip_world` math |
| `last_action` | `(6,)` | previous residual u | Recurrent-ish context |

Total non-image features: ~19 floats. SAC backbone:

```
VisEncoder(image_curr, image_goal) -> z_v (latent 64-128)
+ MLP(tcp_pose_err, force_*, last_action) -> z_s (latent 32)
fusion = concat(z_v, z_s) -> MLP -> (mean, log_std) of action distribution
```

Image encoder: small CNN, similar to SAC-AE / DrQ-v2. Could borrow from `stable-baselines3`'s `VisualCNN` features extractor.

---

## 5. Force tracking — why include it

Including `force_z` and `force_xy_norm` in the observation lets π_θ learn force-aware behavior:
- "Don't push harder when already at 22 N — back off 0.5 mm to discharge friction, then re-push"
- "Pivot along X when a side scrape registers a force spike"  (5 N peak → mini Z retract)

Without these in obs, π_θ would learn from image reward only and have to *infer* force state by reacting to F/T effects visible in image pixels — slower and less sample-efficient. With force in obs, π_θ becomes a **force-aware corrector** that gets state info via the same wrist F/T sensor the real cable inserter will have.

**Caveat:** force is high-frequency noisy. Apply a 5 Hz low-pass (Savitzky-Golay window=15, polyorder=2) before exposing to obs, matching how `aic_controller` itself smooths F/T for stability. Raw `wrist_wrench.wrench.force.*` is what scoring uses, so keep raw available for the safety abort check.

---

## 6. Termination conditions

```python
def is_done(obs, step):
    # success — image goal match
    if image_l1 < 0.02:
        return "success"
    # safety — force abort (16 N sustained 1 s, or 24 N instant)
    if obs.wrist_wrench.wrench.force.z > 24.0:
        return "force_abort"
    # safety — off-limit contact (sensor/contact plugin flag)
    if obs.gazebo_contacts_off_limit:
        return "off_limit"
    # horizon — 30 s @ 20 Hz = 600 steps
    if step >= 600:
        return "timeout"
    return None
```

---

## 7. Episode structure (matching Phase 1 sim)

For each `/insert_cable` action dispatched by `aic_engine` / `insert_cable_skill`:

| Phase | What happens | Reward |
|---|---|---|
| **Boot (first ~10 s)** | π_H drives the gripper to ~30 mm above the port via its existing approach logic | image reward only (large negative because not inserted) |
| **Approach (next ~5 s)** | π_H descends to port entrance | gradually increasing image reward |
| **Seating (last 5 s)** | π_H attempts insertion (spiral for SC, search for SFP) | π_θ residual takes over, learns corrections |
| **Termination** | success / timeout / abort | `r_done` + final image reward |

We don't need to refactor `PerceptionInsert` to split into π_H + π_θ modes — that happens organically:
- `π_H`'s spiral/search behavior is what we want preserved
- The residual just adds small corrections atop whatever `π_H` commands

---

## 7b. Arbitrary-start last-inch regime (v0.2)

**Setting.** `π_H` brings the plug into the *last inch* envelope but does not
need to land at a fixed handoff pose. `π_θ` takes over from any pose inside
the box and is responsible for the entire last-inch insertion. This matches
real deployment, where the handoff from `π_H` is noisy (perception error,
compliance drift, partial contact).

**Training-time start sampling.** At every `env.reset()` we sample a start
pose for the plug tip relative to the port entrance:

```
Δxy      ~ U(-3 mm, +3 mm)        # sideways miss
Δz       ~ U(-8 mm, +4 mm)        # tip above (negative) or below (positive)
Δyaw     ~ U(-15°, +15°)          # in-plane orientation
Δroll    = 0                      # port-relative, locked
Δpitch   = 0
force_z  ~ U(0 N, 8 N)            # partial pre-contact optional
```

These ranges cover what `π_H` actually delivers in the live sim with noisy
perception. They also mean `π_θ` must learn to *recover*, not just refine:
a tip 3 mm off-centre with 0 N contact is the hardest case the policy will
see, so we weight the random distribution towards that corner during early
training and anneal toward centre-only over 100 k env steps.

**Consequence for the reward.** `r_xy` (tip-port XY progress) must be
down-weighted so `π_θ` is not tempted to hover near `port_xy` from t=0;
the image reward alone provides the gradient to actually insert. We set
`w_xy = 0.02` (down from 0.05).

---

## 7c. Reverse curriculum (v0.3) — start near the goal, expand outward

**Setting.** Random initialisation inside the full last-inch envelope
succeeds only ~0% of the time at the start of training (the policy has no
idea what to do), so the success signal is too sparse to bootstrap. The
**reverse curriculum** (Florensa et al. 2018) starts the policy at *easy*
states (close to the goal) and progressively expands the start distribution
as the policy learns.

**Levels.** The start envelope radius is linearly interpolated between
`level=0` (near-goal) and `level=1` (full last-inch envelope):

| level | Δxy   | Δz          | Δyaw    |
|-------|-------|-------------|---------|
| 0.00  | ±1.5 mm | −1.0 to +0.5 mm | ±5°   |
| 0.25  | ±2.1 mm | −0.75 to +0.5 mm | ±7.5° |
| 0.50  | ±2.75 mm | −0.5 to +0.5 mm  | ±10°  |
| 0.75  | ±3.4 mm | −0.25 to +0.5 mm | ±12.5°|
| 1.00  | ±4.0 mm | 0.0 to +0.5 mm   | ±15°  |

All levels share the same goal pose (fully inserted, image-L1 ≈ 0). The
goal does not change with the level — only the start point moves.

**Advance / retreat rule.** After every `--curriculum-eval-window 100`
completed episodes, compute the success rate. If success rate
≥ `--curriculum-advance-threshold 0.6`, advance `level += step` (default
0.05). If ≤ `--curriculum-retreat-threshold 0.15`, retreat `level -=
step`. The level is also written to
`outputs/<run>/curriculum_level.txt` so `--resume` keeps the progression.

**Why not RSI or HER.** No teleop / base policy is available to seed RSI
rollouts, and HER-style goal relabelling is awkward when the goal is an
image (we'd have to capture a new goal image per relabelled trajectory).
Reverse curriculum uses the *same* goal image for every start pose, which
keeps the CNN's job simple: "drive the image toward this fixed reference,
no matter where you start".

**Convergence expectation.** A well-tuned SAC + this curriculum should
reach `level=1.0` (full envelope) within ~80 k env steps and converge to
≥80 % success rate on the full envelope by ~150 k env steps.

---

## 8. Hyperparameter starter set

```toml
[reward]
α            = 1.0      # image L1 coefficient
β_s          = 50.0     # image success bonus
ε            = 0.02     # image L1 success threshold

f_low        = 2.0      # force low threshold
f_optimal    = 10.0     # force optimum
f_max        = 20.0     # force hard cap
β_low        = 0.5
γ            = 0.5

δ_xy         = 0.5      # xy progress
δ_a          = 0.1      # action smoothness
δ_lat        = 0.02     # lateral force

# Component weights in total
w_image      = 1.0
w_force      = 0.05
w_xy         = 0.05
w_action     = 0.10
w_lateral    = 0.10

[termination]
f_abort_n        = 24.0
f_linger_n       = 16.0
f_linger_sec     = 1.0
max_steps        = 600

[training]
total_timesteps  = 200_000    # ~1-2 GPU-hours on 5090
learning_rate    = 3e-4
batch_size       = 256
buffer_size      = 300_000
warmup_steps     = 5_000
seed             = 42

[observation]
image_h          = 32
image_w          = 32
residual_action_scale = [0.0015, 0.0015, 0.0035, 0.08, 0.08, 0.12]  # from PerceptionInsert
```

---

## 9. Files in this folder

| File | Purpose |
|---|---|
| `REWARD_SPEC.md` | this document |
| `reward.py` | `Reward` class implementing 2.1–2.6 with the multipliers in §3 |
| `observation.py` | `ObservationBuilder` that turns ROS `Observation` → π_θ obs dict |
| (later) `env.py` | gym wrapper around AIC sim for SAC training |
| (later) `train.py` | Stable-Baselines3 + custom feature extractor launch script |

---

## 10. Open questions to settle after first runs

1. Does `α=1.0` make L1 reward too negative per step? If yes, scale to `−α·‖I_t − I_goal‖_1` divided by `H·W·C` (per-pixel per-channel) so range is roughly `[−1, 0]`, then bump `β_s` to compensate.
2. Are we using 32×32 grayscale or 64×64 color? 64×64 color triples the data; probably stick with 32×32 grayscale for first runs.
3. Is the force piece shaping helping or hurting? Compare training curves with and without `r_force` term enabled (set `w_force=0`).
4. Should we include the previous image frame (single-step frame stacking)? Two stacked frames at 32×32×3 doubles image input but helps optical flow / contact dynamics perception.
5. **v0.2 specific:** is `w_xy=0.02` enough? If `π_θ` is collapsing to "sit at port centre and don't move", drop `w_xy` to 0. If it is thrashing with no convergence, raise `w_xy` back to 0.05 for the first 50 k steps and anneal.

---

## 11. Training scene (MuJoCo, fast sim)

`env.py` wraps a small **MuJoCo** scene rather than the full AIC Gazebo
sim. MuJoCo runs ~1000× faster than Gazebo, which is required for
residual SAC to converge in <2 GPU-hours per port type.

### 11.1 Scene contents (v0.3 — corrected)

- A 6-DoF free joint for the plug (box, half-extents 3.5×3.5×5 mm), actuated by
  six **velocity-servo actuators** (`<velocity kv=…>`) — one per DoF (mx, my,
  mz, drx, dry, drz). The action is a per-step pose delta → target velocity;
  velocity servos let contact genuinely resist the plug so force is real.
  (v0.2 overwrote `qvel` directly each step, which killed contact/force.)
- A **real open-top rectangular socket**: 4 side walls + a bottom wall forming
  a cavity sized to the plug + clearance (SC ~0.6 mm, SFP ~0.8 mm). The plug is
  driven in until the bottom wall stops it at `z = −depth`
  (SC 16 mm, SFP 51 mm). **There is no top cap** — v0.2 had a solid top cap +
  a brick-filled interior, so the plug could never enter (the core bug).
- Physics timestep 0.005 s, **10 substeps** per 20 Hz control step
  (`integrator=implicitfast`, gravity 0).
- **Three** overhead virtual cameras (left/centre/right, `±12 mm` x-offset) at
  `z=0.06` looking down `−Z`, rendering `32×32` RGB, stacked on the channel
  axis → `(32,32,9)`, matching the AIC 3-wrist-cam observation.
- Contact force = **`mj_contactForce` summed** over the plug's active contacts,
  rotated to world frame → the `force` obs and `f_z` for the reward
  (v0.2 read `cfrc_int`, which is 0 for a free-joint body).

### 11.2 Contact model

- Plug tip → port entrance: stiff normal contact, low friction (`0.15`).
- Plug sides → port walls: stiffer normal, friction `0.30`.
- Plug tip force read by `mujoco` contact force aggregation, exposed as
  `force_z` in the observation (matches the deployment F/T channel).

### 11.3 Action / step

- Action ∈ ℝ⁶, scaled to `[pos_scale, rot_scale]` from REWARD_SPEC §8.
  These are deltas to the plug pose, applied as `mj_step` velocity targets
  (`qvel = action / dt`).
- Control rate: 20 Hz (`dt = 0.05 s`), matching the AIC sim.
- Episode horizon: 600 steps = 30 s.
- Per-step reward from `reward.compute_reward` against the *rendered*
  current image and a captured goal image (taken once at the fully
  inserted pose after a scripted settle).

### 11.4 Goal image capture

During `env.reset` we render a "ground truth" image by settling the plug
at the exact port centre with `force_z = 10 N` applied for 1 s. That
image is the `image_goal` used by `r_image_l1` for every step of the
episode. This avoids needing any "real" goal photo — it is regenerated
inside the sim.

### 11.5 Why MuJoCo, not Gazebo

- Stepping + rendering at 20 Hz × 600 steps × 5 k episodes = 1.5 × 10⁸
  frames. Gazebo at ~50 ms/step would take 87 days. MuJoCo at 0.1 ms/step
  finishes in 4 hours on one CPU.
- We can randomise plug mass, port friction, camera calibration, and
  lighting with a few lines of XML rather than spinning up a new Gazebo
  world each time.
- Lerobot provides a clean PyTorch dataset / replay buffer we can use
  alongside SB3 for offline evaluation; `mujoco==3.5.0` is already
  pinned in `pixi.toml`.

### 11.6 Reality gap

What MuJoCo does **not** model well:
- Cable flex during the last mm (we treat the plug as rigid).
- Latency between action and rendered observation (we use `dt = 0.05 s`
  exactly, no jitter).
- Vision artefacts from the real wrist cameras (we render idealised
  greyscale).

To close the gap before deployment we evaluate the MuJoCo-trained policy
in the AIC sim for at least 50 episodes per port type, then fine-tune
for another 5–10 k env steps if success rate drops below 80 %.
