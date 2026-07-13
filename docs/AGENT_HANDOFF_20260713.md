# Agent handoff — 2026-07-13 (AIC Phase 1, script align-then-seat)

Branch: `flowstate-rl-deploy-and-docs` (all work pushed; HEAD = `4ca6ae3`).
Read `docs/FLOWSTATE_DEPLOY_RECIPE.md` and `docs/INSERTION_PIPELINE_DESIGN.md` too.

## Where we are (one paragraph)

We built a pure-geometry `CONTROL_MODE=script` last-inch controller in
`aic_model/aic_model/RLInsert.py` to test whether a script (no RL) can align +
seat the SFP plug in Flowstate. Result, proven across many deploys: **the script
ALIGNS perfectly to the port mouth (0.7 mm lateral / 0.6° rotation on the correct
port, every run) but CANNOT SEAT** — it stalls at ~5–6 mm on a LATERAL
chamfer-lip catch (lateral creeps 0.7→1.5 mm) at LOW force (~1.5 N, a geometric
catch, not an axial jam). This matches the connector-insertion literature:
alignment is scriptable, the contact seat is not. **Decision (user, firm): use a
force-reactive RL for the seat.** The immediate open loop is one more Flowstate
test run of a calibration tweak (see "Pending test" below), then start the seat RL.

## Corrected understanding (do NOT re-chase these — they were wrong theories)

- **The stall is LATERAL, not rotational.** Handoff `rot_err` is always tiny
  (0.5–1.5°) no matter the calibration. Orientation is NOT the lever for the
  wedge. Earlier "~19° grasp mis-calibration" theory was WRONG.
- **The grasp is REPEATABLE** (TCP orientation matches ~0.6° run-to-run), not
  variable as first claimed.
- **"Wrong port" is FINE / parked.** `_select_sfp_candidate` picks the port
  nearest the plug tip; when handed off between two ports it can pick the
  neighbor (mount_1 Y≈0.346 vs target mount_0 Y≈0.386). This is correct behavior
  — the fix is the FLOWSTATE handoff position (put the plug near the right port),
  NOT the selector. User explicitly said don't touch it.

## What the script does now (`_run_script` in RLInsert.py)

1. Perceive port (multi-frame consensus, nearest-tip) — unchanged, works.
2. Tare wrench at handoff. `CALIB_DUMP=1` logs TCP+assumed-tip (see below).
3. **Phase 1 align**: cancel lateral + square to port at the HANDOFF depth
   (`align_standoff = min(handoff_depth, -4mm)` — so a deliberately-higher
   handoff just aligns higher; HANDOFF_MAX_DIST=120mm won't abort a +5mm start).
4. **Phase 2 descend**: progressive step (2.0mm far → 0.4mm near). At contact
   (force > `SCRIPT_CONTACT_FORCE_N`=5N) switch to a slow force-limited SEAT push
   (advance `SCRIPT_SEAT_STEP_M`=0.3mm until `SCRIPT_SEAT_FORCE_N`=12N, softer
   STIFFNESS in contact).
5. **On stall (depth stuck + lateral > `SCRIPT_WEDGE_LATERAL_M`=1.5mm) or high
   force: HOLD in place, NO retreat** (user instruction — the plug stays at the
   mouth for the RL to take over). Only the separate rl-mode loop still retreats.

All script knobs are `RL_INSERT_SCRIPT_*` env vars (tune without rebuild).

## Grasp calibration state (IMPORTANT — currently mid-experiment)

`SFP_TIP_IN_TCP_QUAT` in `rl_insert_contract.py` maps TCP→plug-tip. History:
- Original: `[0.9852867415, 0.1688620346, -0.0042579615, -0.0260292145]`.
- v17 tried `+1.16°` (from averaged `q_tcp^-1*q_port`): made handoff tilt WORSE
  (rot_err grew to 1.52°).
- **HEAD (4ca6ae3): applied the SAME 1.16° in the OPPOSITE direction** =
  `[0.9840750466, 0.1756266707, -0.0115567892, -0.0248599222]`. UNTESTED — this
  is the pending Flowstate run. Caveat: rot_err is tiny anyway, so this may not
  change the seat outcome; user wanted to try it.

Grafts: one-shot `RL_INSERT_CALIB_DUMP=1` logs TCP + probes candidate
ground-truth tip TF frames (`RL_INSERT_CALIB_PLUG_FRAMES`) to solve the true
transform. **No ground-truth frame has resolved yet** (sim doesn't publish the
plug pose under any tried name). If you want a rigorous recal, get the real frame
name or a TCP+true-tip pose pair in the same frame at one instant.

## Pending test (hand to Codex to deploy)

Deploy HEAD as Flowstate asset **v18** per `docs/FLOWSTATE_DEPLOY_RECIPE.md`:
build from the FULL `docker/aic_model/Dockerfile.student_flowstate` (already
`CONTROL_MODE=script`, `CALIB_DUMP=1`), tag `student-flowstate-script-v18`, bump
manifest to `aic_model_v18`, install, rebind `--name aic_model`. Org
`tar-2@xfa-prod-aic-us`, solution `582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH`.
Watch: is `rot_err_deg` at the handoff check smaller than v17's
`[-1.36,-0.05,0.67]`? Does descent get past ~5.8mm? On stall it should HOLD (not
lift up).

Deploy gotchas learned: build from the FULL Dockerfile (thin overlay crash-loops
on the old entrypoint); `inctl service add` needs the LITERAL asset id
`ai.intrinsic.aic_model_v18` (empty-var expansion caused "Asset ID cannot be
empty"); ~26 GB docker export is NORMAL (every image in this stack is that big);
auth is browser device-login only, never paste tokens.

## Next real work: the force-reactive seat RL

- Start from the script's aligned-at-mouth pose; RL only does the contact seat
  (wiggle / tilt-to-relieve / search + gentle push) using the TARED wrench.
- Reward on force DIRECTION (transfers sim-to-real) + depth progress, force-cap
  penalty à la FORGE (arXiv 2408.04587). Grasp-angle + lateral-offset domain
  randomization in sim so it learns the chamfer catch.
- The align-only RL env already written (`RL/student_teacher/align_env.py`,
  `train_align.py`, `tacc/train_align.slurm`) is for ALIGNMENT and is now lower
  priority — align is solved by the script. The seat RL is the new focus. Note
  the align RL trains on FROZEN perception, so pose-based align RL ≈ a script;
  the real RL value is the force seat.

## Parallel/background threads (not blocking)

- Align-first RL env + trainer + slurm exist for TACC (`align_env.py` etc.);
  Codex had trouble getting the TACC run going (pixi env: fixed with
  `pixi install --locked` in the slurm). Not the current priority.
- SC-port teacher handoff doc exists (`docs/SC_PORT_TEACHER_HANDOFF.md`) for a
  friend's separate effort.

## Key files

- `aic_model/aic_model/RLInsert.py` — the policy (guided/rl/script modes;
  `_run_script`, `_dump_grasp_calibration`).
- `aic_model/aic_model/rl_insert_contract.py` — `SFP_TIP_IN_TCP_QUAT/POS`, obs69.
- `docker/aic_model/Dockerfile.student_flowstate` — deploy image (script mode).
- `docs/FLOWSTATE_DEPLOY_RECIPE.md`, `docs/INSERTION_PIPELINE_DESIGN.md`,
  `docs/FLOWSTATE_STATUS.md`.
