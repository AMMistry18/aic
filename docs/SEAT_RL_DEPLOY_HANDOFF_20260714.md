# Seat-RL deploy handoff — 2026-07-14

Handoff for the next agent working on the **script → seat-RL insertion handoff** on
Flowstate. Written after diagnosing two failed Flowstate deploy runs. The RL *training*
is fine (~0.85 sim success); the problem is the **deploy contract** — the data we feed the
policy at handoff does not match what it trained on. Read this top to bottom before
touching anything.

## TL;DR

The seat policy is force-reactive and works in sim. On Flowstate it flails (wanders in/out,
never seats). Root cause: **we hand the trained policy inputs and an action interface it
was never trained on.** Three concrete mismatches, in priority order:

1. **Observation is out-of-distribution.** The script's +7 mm / -7° pre-engage bias parks
   the plug ~8 mm off the TRUE port. RL trained on ≤1 mm lateral. Handoff obs reads
   `lateral=8.16mm rot=7°` → 8× outside the training band → extrapolation garbage.
   **Status:** partial fix committed (entry-gate + sweep), NOT yet enough (sweep is 2 mm,
   offset is 8 mm — see below). NOT yet deployed/tested.
2. **Action integration is wrong (the "wander").** Deploy free-integrates the action
   (`cmd_pos = cmd_pos + dp`) with no clamp and no base pose. Training clamps the action as
   a bounded *residual around a base pose*. Same action → bounded tiny move in sim,
   unbounded drift in deploy. **Status:** NOT fixed. Bug still live at
   `RLInsert.py:1540` (and a second copy at `:1202`).
3. **Action SCALES likely differ.** Deploy uses `DEPLOY_POS_SCALE=[0.0015,0.0015,0.0035]`,
   `DEPLOY_ROT_SCALE=[0.08,0.08,0.12]`. Training scene default is
   `cart_trans_scale_m=0.001`, `cart_rot_scale_rad=0.0175`. If the trainer did NOT override
   the scene scales to the DEPLOY values, deploy moves 1.5–3.5× (pos) / 4–7× (rot) too far
   per step. **Status:** UNVERIFIED — see open question #1. Do NOT change scales until
   verified.

**The user's framing (correct):** "Is RL doing something it's not supposed to because the
data we give it is wrong?" — **Yes.** The policy is doing what it learned; our deploy
interface (obs band + action scale + action integration) is not the interface it trained
on. Fix the interface before judging the policy.

## Where the code lives (IMPORTANT — two worktrees)

- **Main repo / current work:** `/Users/satya_anandh/Developer/aic` (branch
  `flowstate-rl-deploy-and-docs`). Training env lives here:
  `RL/scene_env.py`, `RL/student_teacher/seat_env.py`, `RL/student_teacher/train_seat.py`.
- **Deploy code (the seat-RL handoff) lives on branch `board-search`**, checked out as a
  **separate git worktree** at `/Users/satya_anandh/Developer/aic-board-search`. Codex owns
  this branch and does the Flowstate deploys from it. Edit the deploy handoff THERE, not in
  the main repo. Key file: `aic_model/aic_model/RLInsert.py` and
  `aic_model/aic_model/rl_insert_contract.py`.
- board-search HEAD at handoff: `c05fa53 "Harden Flowstate perception and seat RL handoff"`
  (Codex committed the entry-gate/sweep edits; worktree is clean).

## The failing behavior (from the two Flowstate logs)

Script descends, stalls at ~3.3 mm depth at low force (~1.9 N), +Y nudge (2 mm cap) does
not free it, hands off to seat RL. Seat RL:
```
[seat_rl] handoff: depth=3.28mm lateral=8.16mm rot_deg=7.00 force=1.00N   <- 8mm OOD
step 20: depth 8.0,  lateral 3.29     <- pulls in, gains depth
step 50: depth 12.0, lateral 4.26
step 70: depth 7.9,  lateral 4.22     <- LOSES depth, comes back out
step 110: lateral 7.91                <- wanders far out
step 200: depth 16.4 ... step 240: 10.1  <- oscillates, never seats
[seat_rl] step budget (250) exhausted; holding position -> returned False
```
Diagnostic facts from the log:
- `raw` action column [2] (insert axis) is ~+0.9 almost every step → **the policy IS
  commanding "down."** It is not choosing to retract.
- `delta_port_mm` depth is frequently `-0.0` → the plug is **mechanically wedged**; the
  commanded down produces no motion, and the open-loop `cmd_pos` keeps integrating anyway →
  when it slips it lurches (depth 8→16→8). That is the wander (bug #2).
- Perception is excellent: `7/7 agree, reproj=1.17–1.37px`. Port location is NOT the
  problem — do NOT add re-perception.

## What the policy actually trained on (ground truth — verify before trusting)

`RL/student_teacher/seat_env.py` `_CANONICAL_STAGES["wedge"]` (the hardest/handoff stage):
- lateral offset ~0.65–1.00 mm; **accepted lateral 0.30–1.00 mm** (`accepted_lateral_range_m`)
- depth level_range (0.05, 0.45) → plug already IN the port, anywhere ~5–40 mm deep
- tilt ≤ ~1°, low contact force
- Action application (`seat_env.step` → `scene_env._apply_cartesian_action`, scene_env.py
  ~1277): `SEAT_ACTION_GAIN=0.20` scalar, then **residual accumulate + CLAMP** to
  `cart_pos_limit_m`/`cart_rot_limit_rad` (or `base_script_residual_limit_m=0.01` /
  `_rad=0.10` if base_script mode), commanding `base_pose + clamped_residual`. This is the
  scheme deploy must mirror.

## What has been done (committed on board-search, c05fa53)

1. **+Y stall nudge** in the script descent (`_nudge_to_unstick`, +Y only, 0.4 mm steps,
   2 mm cap) — pre-handoff attempt to drop the plug in. (Does not fix the 8 mm problem.)
2. **Seat-RL entry gate + sweep** in `_run_script`'s handoff block:
   - `_true_frame_pose(Rp, actual_port_pos)` — measures depth/lateral vs the TRUE port.
   - `_is_in_port(true_depth, true_lateral, f_mag)` — gate: `depth ≥ SEAT_ENTRY_MIN_DEPTH_M`
     (4 mm) AND `lateral ≤ SEAT_ENTRY_MAX_LATERAL_M` (1.5 mm) AND low force.
   - `_sweep_into_port(...)` — if not in port, sweep `-Y, +Y, -X, +X` (user chose -Y first),
     0.4 mm steps, `SEAT_SWEEP_MAX_PER_DIR_M=2 mm` per direction, step back between
     directions, hand off the moment `_is_in_port` becomes true.
   - Handoff now only fires when in-port (or after a successful sweep); else holds + fails
     instead of feeding an 8 mm start. Logs `[script] IN-PORT` / `NOT in port -- sweeping`.
   - **This is committed but NOT yet deployed to Flowstate. Both logs above are the OLD
     image (no IN-PORT/SWEEP lines).**

## Why the entry-gate alone is NOT sufficient (the trap)

The plug stalls **8 mm off the true port** (bias). The sweep cap is **2 mm/direction**. 2 mm
cannot recover an 8 mm offset — the sweep will run all four directions and fail. So the
entry-gate as-committed will just convert the flail into a clean "sweep failed, holding."
Better than flailing, but still no seat. **Do NOT just raise the sweep cap blindly** — the
user was emphatic that the +7 mm bias is intentional and load-bearing (it lets the SCRIPT
do the easy pre-seating; removing it is off the table — the user shut down that suggestion
hard). The real fixes are the action-interface bugs (#2, #3) plus reconciling the 8 mm
handoff offset with RL's ≤1 mm training band WITHOUT deleting the bias.

## NEXT STEPS (in order)

### Step 1 — VERIFY the action interface against training (do this FIRST, no code changes)
The whole diagnosis hinges on the deploy action interface differing from training. Confirm:
- **Scales:** find where the trainer builds the scene env for `seat_env` and check whether
  it sets `cart_trans_scale_m` / `cart_rot_scale_rad` to the `DEPLOY_POS_SCALE` /
  `DEPLOY_ROT_SCALE` values. `seat_env.py` imports `DEPLOY_POS_SCALE` and uses it in its
  reset probe math (lines 417, 440), which HINTS the scene is configured to the deploy
  scales — but VERIFY, do not assume. If they match, scale is NOT a bug (drop #3). If they
  differ, that's a real bug.
- **Integration:** confirm training uses clamped-residual-around-base-pose
  (`scene_env._apply_cartesian_action`, ~1296) while deploy uses free `cmd_pos += dp`
  (`RLInsert.py:1540`). This one is confirmed a mismatch from reading the code, but
  re-confirm the deploy really has no clamp/base anchor.

### Step 2 — FIX the action integration in deploy (`_run_seat_rl`, board-search)
Replace the free integrator with training's scheme:
- keep a **base seat pose** = the handoff TCP pose,
- accumulate action into a **residual**, **clamp** it to training's limit
  (`base_script_residual_limit_m`/`_rad` if that's the training mode, else
  `cart_pos_limit_m`/`cart_rot_limit_rad`),
- command `base_pose + clamped_residual` each step.
Simplest robust alternative (matches the SCRIPT, which moves cleanly): re-anchor to the
MEASURED tip each step — `cmd_pos = tip_pos + dp; cmd_R = dR @ R_tip` — instead of
integrating `cmd_pos`. Pick whichever provably matches `scene_env`. There are TWO copies to
fix: `RLInsert.py:1540` (seat RL) and `:1202` (check whether the other loop needs it too).

### Step 3 — reconcile the 8 mm handoff with the ≤1 mm training band (KEEP the bias)
The bias must stay. Options that keep it (get user's call — do NOT unilaterally shrink/drop
the bias, the user rejected that): e.g. the script's LAST move before handoff walks the
plug from the biased pose toward the true port so the handoff lands ≤1.5 mm true (a bounded
"de-bias to entry" move, distinct from deleting the bias); or extend the sweep budget to
cover the real offset in the +Y-toward-true-port direction; or confirm whether the intended
design is that the SCRIPT (not RL) uses the bias to seat and RL only takes over once truly
in-port. This is a DESIGN decision — ask the user, do not guess.

### Step 4 — deploy v37+ and re-test (only after 1–3)
Per `docs/FLOWSTATE_DEPLOY_RECIPE.md`: full `Dockerfile.student_flowstate` (not a thin
overlay — crash-loop gotcha), keep `RL_INSERT_CONTROL_MODE=script_then_seat_rl`, **bump the
asset name** (v37), `inbuild` → `inctl asset install` → `service delete`/`add --name
aic_model`. Success signal in logs: `[seat_rl] handoff:` lateral ≤ ~1.5 mm (NOT 8 mm), and
depth advancing monotonically instead of oscillating.

## Hard-won facts / do-NOT-repeat
- **Do NOT re-perceive the port** — perception is 7/7 / ~1.2 px. The catch is plug-side
  geometry, resolved by feel.
- **Do NOT delete or shrink the +7 mm bias without the user** — they consider it
  load-bearing for the script's pre-seat. (They were very clear.)
- **Do NOT change reward/training** — the policy trains fine; this is a deploy-contract bug.
- The seat-RL "success_mean" curve reads >1.0 because it's a shaped/bonus'd mean, NOT a
  0–1 rate — it is NOT a deploy-readiness number. Get a real eval-success rate separately.
- Deploy the current CHECKPOINT for plumbing smoke-tests; don't read capability from a
  not-yet-fixed-interface run.

## Key file:line anchors (board-search worktree)
- `aic_model/aic_model/RLInsert.py`
  - `_run_seat_rl` ~1435; the wander bug `cmd_pos = cmd_pos + dp` at **1540** (+ **1202**).
  - handoff gate block ~1653 (`IN-PORT` / sweep logic).
  - entry-gate helpers `_true_frame_pose` / `_is_in_port` / `_sweep_into_port` just before
    `_run_seat_rl`.
  - bias: `SCRIPT_BIAS_{X_M,Y_M,RX_RAD}` ~198–200 (-0.8 mm, +7 mm, -7°).
  - entry/sweep knobs: `SEAT_ENTRY_*` / `SEAT_SWEEP_*` ~270–277.
- `aic_model/aic_model/rl_insert_contract.py`: `DEPLOY_POS_SCALE` (18), `DEPLOY_ROT_SCALE`
  (19), `deploy_action_delta` (121).
- Training (main repo): `RL/scene_env.py` `_apply_cartesian_action` (~1277), cart scales
  (218–223), base_script residual limits (242–243). `RL/student_teacher/seat_env.py`
  `step` (734), wedge stage (~147), `SEAT_ACTION_GAIN` (93).

## Related memory / docs
- `docs/SEAT_RL_V2_PLAN.md`, `docs/FLOWSTATE_DEPLOY_RECIPE.md`,
  `docs/reference/nic_card_mount.sdf`.
- Memories: [[seat-rl-v2-physics-recalibration]], [[seat-env-start-depth-and-nic-geometry]],
  [[seat-env-reset-rotation-floor]], [[script-align-seat-findings]],
  [[flowstate-deploy-recipe]].
