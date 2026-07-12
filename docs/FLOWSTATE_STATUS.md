# Flowstate status (current)

Updated: 2026-07-12. This is the single live status doc for the Flowstate
deployment. For the step-by-step deploy procedure see
[FLOWSTATE_DEPLOY_RECIPE.md](FLOWSTATE_DEPLOY_RECIPE.md). Superseded handoffs are
in [archive/](archive/).

## What works now

- The `aic_model` service deploys and starts cleanly (no lifecycle-service
  failure, no router crash-loop). `/aic_model/change_state` resolves; configure
  and activate succeed.
- The deployed policy runs `RL_INSERT_CONTROL_MODE=rl` — the LEARNED policy
  (`final_insert_sfp_flowstate_v1.ts`), not the guided scripted fallback.
  Confirmed by `control=rl` in the run log.
- Perception is clean at handoff (reproj ~1.2-1.4 px).

## Current blocker: step-0 handoff offset

The Flowstate macro hands the cable off with a real offset (~6.0 mm lateral,
~8.6 deg rotation) that the student must correct. The contract safety gate used
to abort on step 0 before the policy could act. As of v11/v12 a grace window
(`RL_INSERT_SAFETY_GRACE_STEPS`, default 40) lets the policy correct the handoff
error for the first N steps before the hard 6 mm / 0.20 rad limits apply; the
retreat and force guards stay live from step 0. See
`aic_model/aic_model/RLInsert.py` (search `SAFETY_GRACE_STEPS`).

Open question being tested: the student was trained mostly on near-square
handoffs, so a 6 mm / 8.6 deg start is somewhat out of distribution. The grace
window lets us OBSERVE whether the policy's correction generalizes. Three
outcomes and their next step:

1. Corrects and seats -> done.
2. Reduces error partway then stalls/times out -> train a residual RL correction
   on top of this policy.
3. Makes lateral worse / hits the grace ceiling -> retrain with randomized
   handoff offset (domain randomization of handoff lateral/rotation in sim).

## Deployed versions (history)

- v8: WRONG — guided image bundled under an rl-named asset (mistake).
- v9: WRONG — crash-looped on `AIC_MODEL_ROUTER_ADDR must be provided` (overlay
  kept the old strict entrypoint).
- v10: first correct rl deploy (baked router addr, non-fatal entrypoint);
  `control=rl` confirmed; aborted at step 0 on the 6 mm gate.
- v11/v12: v10 + grace-window safety gate.

Always bump the asset manifest `name` (v10 -> v11 -> ...) per deploy; reinstalling
the same identity does NOT replace the running image.

## Key files

- `aic_model/aic_model/RLInsert.py` — self-contained policy: own SFP perception,
  straight-descent last-inch RL, grace-window safety gate.
- `aic_model/aic_model/rl_insert_contract.py` — 69-value observation contract,
  TCP->SFP-tip calibration, action scaling, guided target (fallback).
- `docker/aic_model/Dockerfile.student_flowstate` — the thin Flowstate image
  (rl mode, baked Zenoh router addr, non-fatal entrypoint).
- `models/final_insert_sfp_flowstate_v1.ts` (+ `.contract.json`) — the policy.
- `docs/FLOWSTATE_DEPLOY_RECIPE.md` — how to build/bundle/install/rebind/verify.
- `docs/FLOWSTATE_MUJOCO_PARITY_20260711.md` — 69-obs contract diagnosis (still
  reference).

## MuJoCo baseline (NOT a Flowstate score)

Epoch-25 held-out: 210/300 success, 88 timeout, 2 bad_collision. This is the
policy that drifts ~30%; relevant to why outcome 2/3 above are plausible.
