# Seat RL v2 — reverse curriculum from a WEDGED start

Created 2026-07-13. Supersedes the first seat-RL env's fixed aligned-at-mouth start.
Driven by new Flowstate deploy evidence (script now pre-engages the plug DEEP, then
hands off at a low-force LATERAL wedge).

## New Flowstate ground truth (the data this plan is calibrated to)

The tuned script (v35: pre-engage bias absolute-shift dX=-0.8mm, dY=+7.0mm, tilt=-7deg,
handoff on depth-stall OR force) now pre-engages the plug cleanly and deep. Three
consecutive identical-setting runs landed at DIFFERENT depths:

| run | depth reached | how it stopped |
|---|---:|---|
| 1 | 26.8 mm | depth-stall, 3.67 N, lateral 0.82 mm |
| 2 |  5.5 mm | depth-stall, 2.66 N, lateral 1.32 mm |
| 3 | 40.5 mm | depth-stall, 2.27 N, lateral 1.85 mm |

Key facts (ALL new vs the original mouth-only calibration):
- The plug descends to 26-40 mm at LOW force (~2 N) with <1 mm lateral most of the way.
- It then catches on a LATERAL wedge (lateral creeps to ~1.9 mm as depth stops), NOT an
  axial jam (force stays ~2 N). A straight-down push does NOT help; it needs a ~0.5-1 mm
  LATERAL correction to unstick, then it slides home.
- Stall depth is VARIABLE run-to-run (5.5 / 26.8 / 40.5 mm) under identical settings ->
  the catch is stochastic, so it CANNOT be scripted. This is the RL's job.

## Why the first seat env is wrong for this

The v1 seat env (`RL/student_teacher/seat_env.py`) resets to an ALIGNED pose at the
mouth (the settle squares it) and pins the start at the ~6 mm ridge only. Two problems:
1. It starts STRAIGHT -> the unstick task is trivial / RL barely needed. The real
   handoff is a plug that is already STUCK (0.5-1 mm off, in contact).
2. It only covers the mouth wedge; the real behavior is a deep descent that can catch
   at any depth 5-40 mm.

## The idea (user, 2026-07-13)

Reverse curriculum from ~90% inserted back to the wedged condition, AND start the plug
already STUCK (lateral+tilt offset in contact), not straight -- because a straight start
makes the task too easy and the RL is never exercised on the real skill (a reactive
lateral unstick). Physics must match the NEW Flowstate data first.

## Plan

### Phase 1 — RE-VALIDATE PHYSICS against the new Flowstate data (DO FIRST)

The existing sentinel `RL/student_teacher/gate0_contact_jam.py` only checks a jam in the
5-9 mm band (the mouth wedge). It does NOT validate the new behavior. Extend/replace it to
confirm the sim reproduces:
- (a) a low-force descent PAST 9 mm to ~40 mm (not just a mouth stall),
- (b) a LATERAL wedge (lateral grows while depth stalls at ~2 N), at VARIABLE depth,
- (c) that a small (~0.5-1 mm) LATERAL correction UNSTICKS it and lets it seat.
Gate: if the sim's wedge is axial (high force) rather than lateral (low force), RL trained
on it will NOT transfer -- fix contact params (see the calibrated ridge in
`RL/scene_env.py`: `compiled_contact_ridge_*`, friction, solref/solimp) until the sim shows
the deep/low-force/lateral/variable pattern. This is a sentinel/diagnostic, NOT training.
Do NOT change calibrated physics blindly -- change it to MATCH the Flowstate traces above.

### Phase 2 — WEDGED-START reset + reverse curriculum (`seat_env.py` rework)

Replace the aligned-at-mouth reset with a start that is already STUCK:
- Reset with a randomized LATERAL offset (~0.3-1.0 mm) + small tilt, placed IN CONTACT at
  the curriculum depth, so the plug begins wedged (not squared). Randomize offset
  magnitude AND direction so RL learns to unstick from many stuck poses, not one.
- Reverse curriculum on BOTH axes together (easy -> hard):
    * depth: near-seated (level ~0.08, ~90% in) -> wedge depths (level up to ~0.45),
    * wedge severity: barely-off (~0.2 mm) -> the real ~1 mm lateral catch.
  Easy = near-seated + barely-off (trivial nudge). Hard = wedge-depth + 1 mm-off (real
  stuck condition). The env already has a `level`-based reverse curriculum in
  `_sample_start_tcp` (scene_env.py) -- reuse it, add the in-contact lateral/tilt offset.
- Success = seated (cfg.seated_depth_m = 45.8 mm) from the stuck start.

### Phase 3 — TRAIN (trainer already built)

Reuse `RL/student_teacher/train_seat.py` (plain SAC + AsymmetricSACPolicy, W&B + video +
success metrics, all verified working). Point it at the new wedged-start reverse-curriculum
env. Verify the FORGE-style reward in seat_env rewards the LATERAL UNSTICK (relieve
side-load + gain depth) -- that is now confirmed as the actual skill, not axial push.

### Phase 4 — DEPLOY as the script->RL handoff

Script (v35+) pre-engages deep and hands off at the wedge; RL does the reactive ~0.5-1 mm
lateral unstick to seat the rest. Wire into RLInsert.py's control-mode router.

## Related
- Script handoff behavior + bias tuning: `aic_model/aic_model/RLInsert.py` (_run_script).
- v1 seat env / reward / trainer: `RL/student_teacher/seat_env.py`, `train_seat.py`.
- Old physics calibration (mouth-only): `RL/student_teacher/MUJOCO_CONTACT_PHYSICS_CALIBRATION_20260712.md`.
- Old sentinel: `RL/student_teacher/gate0_contact_jam.py`.
- SEPARATE blocker: scoring TF timing (`cable_0/sfp_tip_link` absent at trial start) --
  under investigation, unrelated to insertion/RL.
