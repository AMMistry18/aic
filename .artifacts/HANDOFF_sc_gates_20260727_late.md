# Handoff — SC: the depth question is answered, two gate fixes shipped

**Date:** 2026-07-27, late night (supersedes parts of
`HANDOFF_sc_wedge_recovery_20260727_night.md`)
**Branch:** `aic-model-flowstate-20260727` @ `1271ee1`, pushed to origin
**Solution:** `Wseto Copy of satya robert`,
`2b7f0a8e-1995-4b16-b3d7-6770735736df_BRANCH`, org `tar-2@xfa-prod-aic-us`

## The headline

**Nothing in SC's perception or geometry is broken.** Plug pose measures
correctly, port pose triangulates correctly, alignment converges to 0.09 mm, and
depth is honest. Three *thresholds* were measuring quantities unrelated to the
decisions they gated. Two are now fixed; the third is the remaining known
failure mode.

## What is committed and pushed

`1271ee1` — "Stop SC rejecting insertions it has already earned". Three files
(`sc_controller.py`, its overlay copy, `test_sc_controller.py`). Source and
overlay byte-identical. No SFP / `v50_controller.py` change.

**165 passed** on `test_v50_controller.py` + `test_sc_controller.py` (144
before). **263 passed** on the full `aic_model/test/` directory (242 before).

### 1. Event acceptance is no longer depth-gated

`_event_status` deferred any event below `seat_candidate_depth_m` (15.2 mm).
Removed. Any fresh event returns SEATED, matching SFP and
`docs/INSERTION_EVENT_POLICY.md`. `depth_m` is retained only to log
`SC_EVENT_ACCEPTED ... depth_mm=`.

New `_stall_event_dwell` (3.0 s, `RL_INSERT_SC_STALL_EVENT_DWELL_WALL_S`): on
stall the controller holds position and waits for a late event before conceding.
Required, because the 19:31 event arrived 0.66 s *after* `_seat()` returned —
the depth fix alone would not have saved that run.

`seat_candidate_depth_m` is unchanged and still governs when to stop driving, so
partial-insertion depth is unaffected.

### 2. Perception gates degrade instead of aborting

New `_centre_reproj_px`. The select gate and the consensus gate both read
**centre** reprojection and degrade to the best available candidate
(`SC_PERCEPT_DEGRADED`) rather than returning `None`. Consensus collects every
finite sample, prefers those under `SC_MAX_PORT_REPROJ_PX`, and falls back to
the full pool only if none qualify. Alignment timeout returns `ALIGN_TIMED_OUT`
and proceeds to seating instead of discarding the pose.

`RL_INSERT_SC_STRICT_PERCEPTION=1` restores the previous abort behaviour on
every degraded path.

## The depth question — ANSWERED, do not re-open

The previous handoff proposed "log depth at the instant the scoring event fires"
as the one measurement that settles depth. **It is already settled**, and the
answer was in this repo the whole time —
`.artifacts/HANDOFF_sc_insertion.md:426`:

> **Insertion triggers at 1 mm tip proximity, not contact** (since the
> 2026-07-09 assets, aic#593)... **Check against the first successful run.**

The 19:31 run *is* that check: stalled at 1.39 mm, event fired 0.66 s later.
Proximity trigger confirmed. Depth is honest; `SC_INSERT_DEPTH_M` (15.64 mm)
is the bore, not the event threshold. There is **no ~14 mm axial bias** and
nothing to recalibrate.

`SC_EVENT_ACCEPTED ... depth_mm=` now records this on every success, so the
claim stays falsifiable.

## Corrections to the previous handoff

- Its **step 3** ("keep the event dwell live during recovery") was insufficient
  as written. `_wait_for_insertion_event` calls `_event_status`, which carried
  the same 15.2 mm floor — recovery would have logged `SC_EVENT_DEFERRED` and
  discarded its own success anyway. Fixed by removing the floor, not by making
  the dwell reachable.
- Its **perception admission fix** proposed "gate on the centre, corners need
  their own scoped gate". The corner gate is **not needed** — see the yaw facts
  below. Gating on the centre alone is sufficient and is what shipped.

## Facts established from the user this session (not in any repo doc)

1. **The board spawns at randomized poses, but the macro's handoff is always in
   the same place relative to the SC port.** Therefore `SC_PRESERVE_HANDOFF_YAW=
   True` (the deployed default) is **correct**, and vision yaw is unnecessary.
2. **The SC port cannot rotate** — always the same rotation.

**Consequence:** `.artifacts/HANDOFF_sc_insertion.md:415-421` ("Yaw therefore
must be measured from vision every trial, and cannot be replaced by a constant
prior... the yaw-conditioning argument for retraining (6d) stands") is
**SUPERSEDED**. Do not chase a vision-yaw fix or a yaw-conditioned retrain on
the strength of it. That doc line is still uncorrected in the repo.

3. **Blind descent was considered and declined by the user** ("it'll find some
   pose"). If perception ever returns zero candidates across all frames, the run
   still aborts. Do not add a dead-reckoning fallback without revisiting the
   OPEN tunneling bug (#121/#137) first.

## Why the corners were the wrong thing to gate on

Kept here because it is the reasoning that justifies the fix:

- Position is `kp_3d[4]` alone (`sc_controller.py`, `raw_X = kp_3d[4]`).
- `_rolled_kps` permutes only indices 0-3, so **KP4 is identical under all 16
  roll assignments**. Corner error is mathematically independent of centre
  correctness.
- The corners' only rotational output is yaw, which `seat_frame` /
  `SC_PRESERVE_HANDOFF_YAW=True` discards. The insertion axis is hardcoded
  world -Z in `_estimate_sfp_port_orientation`.
- The 24x crop refinement (`AIC_SC_POSE_CROP_REFINE`, on by default for both
  port and plug) was TACC-tuned on **pose-centre** error, which it halved. The
  one piece of validated accuracy work targeted the keypoint the gate ignored.

19:44 numbers: centre 1.54-2.48 px, two corners 8-11 px, mean 5.30-5.38 px
against a 5.0 px gate. All 7 frames rejected; 8.6 s used of 144 s.

## Scoring arithmetic that drives the "degrade, don't abort" policy

`docs/scoring.md:106-114`:

| outcome | points |
|---|---|
| abort at handoff, never approach | proximity only, ~25 max |
| attempt, reach 1.39 mm | partial insertion, **38-50** |
| correct-port seat | 75 |
| **wrong-port seat** | **-12** |

Refusing to try is strictly worse than trying badly — **except** for wrong-port,
which is ~37 points worse than aborting. Hence `SC_MAX_HANDOFF_LATERAL_M`
(10 mm) stays **fatal**: nearest-to-tip selection is the wrong-port defence, not
reprojection. Note the interaction — since `bd1ae9d`, SC accepts wrong-port
events as success, so a wrong-port seat is a silent -12 the model reports as a
win.

## NEXT WORK: the visual-recovery force lockout (item 3)

The remaining known failure mode. Untouched.

- `visual_recovery_min/max_force_n` = 0.4-1.5 N. A wedge sits at 2.8-3.2 N, so
  `SC_VISUAL_RECOVERY_SKIPPED ... light_contact=False` — **SC's only recovery is
  locked out precisely when it is needed.**
- Worse, during recovery it commands `recovery_start_depth +
  visual_recovery_force_n / axial_stiffness_n_m` = **start + 2 mm forward**. It
  pushes while shuffling sideways. A wedge needs the axial load *released* —
  19:31 is direct evidence, the plug seated the instant the push stopped.
- It is single-shot: every exit returns STALLED and `run()` turns that into
  False. SFP has WEDGED-vs-STALLED, a 1 mm unload, unbounded retries, slow
  re-probe and a full retract; SC has none of it.
- Visual recovery answers "am I aimed wrong". The actual failure is "I'm aimed
  right (0.09 mm) and physically caught on the lip". Different problems — do not
  merge them.

Caveat from the previous handoff still stands: SFP's constants are tuned for a
42 mm insertion; SC seats ~1-2 mm past its mouth. Port the structure, derive the
thresholds from SC traces.

**Only 1 of 9 `return STALLED` paths currently has the event dwell** (the
`SC_VISUAL_RECOVERY_SKIPPED` path, which is the one 19:31 hit). The six
visual-recovery exits still bail instantly. Low probability today because
recovery rarely activates at all, but it is a real hole and belongs with this
work.

## Repo gotcha that cost time — read before running tests

`pytest aic_model/test/` (whole directory) imports `aic_model` from a **stale
installed copy** at `.pixi/envs/default/lib/python3.12/site-packages/aic_model/`,
which `pixi run` does NOT rebuild from source. New tests then fail against old
code and it looks like a real regression. `pytest aic_model/test/<file>.py`
(named files) uses the source tree correctly.

```
# check which module is live
pixi run python -c "import aic_model.sc_controller as m; print(m.__file__)"
# unstick (regenerable build artifact)
cp aic_model/aic_model/sc_controller.py \
   .pixi/envs/default/lib/python3.12/site-packages/aic_model/sc_controller.py
```

Old tests pass against either copy, which is why this stays hidden until you add
a test.

## What to watch in the first field logs

- `SC_EVENT_ACCEPTED ... depth_mm=` — depth at event. ~1.4 mm on a scored
  insertion confirms the proximity trigger outright.
- `SC_PERCEPT_DEGRADED` — perception proceeded on a below-gate estimate. Some
  are expected; a run that is *entirely* degraded is worth investigating.
- `SC_ALIGN_TIMEOUT_PROCEEDING` — seating from an unconverged pose.
- `SC_STALL_EVENT_DWELL_START` / `_TIMEOUT` — the late-event dwell firing.

## Repository state

- `aic-model-flowstate-20260727` @ `1271ee1`, pushed. **Not merged to `main`.**
- `origin/main` has moved to `8b5821f` — three commits ahead of this branch's
  base, including "fixed perception skill, still more issues". **Check whether
  that touches the same SC perception path before merging.**
- **NOT DEPLOYED.** The live Wseto pod is the `20260728-wrong-port-accepted-r2`
  bundle (built 19:22). It has the SFP acceptance fix but **not** the SC
  strict-port flip from `bd1ae9d`, and not `1271ee1`. A rebuild + upload is
  required for any SC change to take effect.
- `docker/aic_model/Dockerfile`: the user's local Apple-Silicon build variant
  (COPY-from-local-checkout, to avoid the QEMU/credentials failure) was
  **replaced by the branch version** this session. It is preserved in
  `git stash@{0}` and `/tmp/Dockerfile.local.backup` (the latter is NOT durable
  across reboot). That local variant is **stale**: its `py_compile` list lacks
  `sc_controller.py` and `sc_visual_alignment.py`, and it sets
  `RL_INSERT_ACTION_TIME_BUDGET_S=45` vs the branch's 150. Do not build with it
  as-is.
- Untracked and deliberately left alone: `diagnostics logs.txt`,
  `flowstate/scripts/build_aic_model.sh`, `flowstate/services/`.

## Caveat

None of `1271ee1` has run against a robot. It is verified against tests and the
two field logs of 2026-07-27 (19:31 and 19:44). The 19:44 abort should now
proceed; the 19:31 stall should now be credited. That is inference from the
traces, not observation.
