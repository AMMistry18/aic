# Handoff — SC insertion: event acceptance shipped, wedge recovery is next

**Date:** 2026-07-27, night
**Branch:** `aic-model-flowstate-20260727` at `bd1ae9d` (pushed to origin)
**Worktree:** `/home/Anshul/AIC_Phase_1/aic_0/aic-model-flowstate-20260727`
**Solution:** `Wseto Copy of satya robert`,
`2b7f0a8e-1995-4b16-b3d7-6770735736df_BRANCH`, org `tar-2@xfa-prod-aic-us`

## What is committed and pushed

`bd1ae9d` — accept any fresh insertion event as success on both SFP and SC.

- **SFP** (`v50_controller.py::_event_status`): returns `SEATED` for any fresh
  event, warns on an alternate port instead of `HARD_FAILURE`. This was built
  earlier in the evening and was already live on Wseto; it was uncommitted until
  now.
- **SC** (`sc_controller.py:462`): `SC_STRICT_PORT_EVENT` default flipped
  `True` -> `False`. **This is the new work.** `docs/INSERTION_EVENT_POLICY.md`
  already claimed SC was non-strict, but the code default was still `True`, so
  the SC path kept the wrong-port rejection the policy exists to remove.
- Regression test + `docs/INSERTION_EVENT_POLICY.md` are in the same commit.

Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pixi run pytest -q
aic_model/test/test_v50_controller.py aic_model/test/test_sc_controller.py`
-> **144 passed**. Source and overlay copies are identical in the changed region;
`py_compile` clean on both.

**Not deployed.** The live Wseto pod is the `20260728-wrong-port-accepted-r2`
bundle (built 19:22, pod up 00:24:09Z). It has the SFP fix but **not** the SC
default flip. A rebuild + upload is required for the SC half to take effect.

## The two SC failures from the 19:24 pod

Both from the same build; the user confirmed this explicitly. Do not treat the
difference between them as a regression.

### Attempt 1 (19:31) — inserted, reported failure

Perception passed 7/7, alignment converged (lateral 0.09 mm), seating advanced
freely from -6.8 mm to **1.29 mm** under 0.5 N, then axial force climbed
(-1.27 -> -3.18 N) while depth moved only **1.29 -> 1.39 mm** for the rest of the
run. `wall-time stall` -> `seating ended without confirmation: stalled` ->
`insert_cable() returned False`. The scoring event arrived **0.66 s later**
(`cable_0#1#/sc_port_3/sc_port_base`).

**Reading: the plug wedged on the entrance lip and seated the moment we stopped
pushing.** The user independently confirmed the wedge behaviour is routine.

### Attempt 2 (19:44) — never moved

All 7 frames rejected: raw reprojection 5.30-5.38 px against the 5.0 px gate at
`sc_controller.py:3235`. Aborted in 8.6 s with **135 s of budget unused**.

Root cause is a gate-design mismatch, confirmed in code:

- position is `kp_3d[4]`, the explicit centre keypoint (line 3054 -> line 3138);
- orientation comes from the four corners (line 3097);
- but the gate metric averages **all five** keypoints across cameras
  (`_mean_reproj_px`, line 2863).

Centre residuals were 1.54-2.48 px across the seven frames; two corners sat at
8-11 px because the near-symmetric mouth makes corner identity ambiguous (see
`SC_KEYPOINT_ROLL`). So four keypoints nobody aims with outvote the one that
determines position, and the mean lands either side of 5.0 at random.

## Theories that were tested and killed — do not re-run these

1. **"The new SC mouth-pose model caused a regression."** Killed: both attempts
   are the same build.
2. **"Uncalibrated `SC_TIP_IN_TCP` is causing a ~34 mm axial error."** Killed by
   code. `_tip_pose()` is pure kinematics through the *measured* per-grasp
   transform, `sc_tip_from_tcp` uses the measured transform whenever priming
   succeeded, and `run_sc_insertion` refuses to start without priming. The
   constant never touches a field run. The
   `measured_minus_fixed_grasp_mm=[-4.9, -8.31, 34.2]` log line reports how much
   the measurement *corrected* the constant — it is the system working, not an
   error signal.
3. **"~14 mm axial depth bias, matching the model's 13.55 mm single-view PnP
   error."** Weakly supported and probably wrong. It assumes full SC seating is
   ~15 mm, which was never checked, and the 13.55 mm figure is *single-view PnP*
   from `validate_sc_mouth_pose_test.json` while the deployed path is 3-camera
   DLT triangulation — a different estimator. The force/depth trace shows free
   travel to 1.29 mm then a hard stop, which is a physical wedge, not a
   mis-measured depth.

Still true and worth remembering: **reprojection measures self-consistency, not
accuracy.** Three cameras can agree tightly on a physically wrong point, and
`SC_PERCEPT_AGREE_TOL_M = 0.004` (4 mm, line 544) against a 0.725 mm binding
half-clearance cannot catch that. Cross-frame consensus is not a safety net for
systematic bias.

## The one measurement that settles depth

**Log depth at the instant the scoring event fires.** The event is ground truth
for "seated", so the depth we report at that moment *is* the bias, read directly.

- ~1.4 mm -> depth is honest; this is purely a wedge/compliance problem.
- ~15 mm -> there is a real axial offset after all and perception needs work.

One log line on a path that already fires. This decides between two very
different bodies of work; do it before investing in either.

## Next work: SC wedge recovery

**SFP already has a full wedge recovery system and SC has none of it.**
`v50_controller.py` lines 41-47 distinguish `WEDGED` from `STALLED` — "Only
WEDGED earns a retract-and-retry" — backed by `wedge_retry_enable`, unbounded
retries, `wedge_recovery_unload_m` (1 mm unload), slow re-probe, force- and
moment-guided micro-nudges with lateral/tilt bounds, progress thresholds, and a
full retract to `retract_clear_depth_m`.

SC's only recovery is *visual*, and it is gated at
`visual_recovery_max_force_n = 1.5` (line 630). The wedge sat at 2.81-3.18 N, so
the log shows `SC_VISUAL_RECOVERY_SKIPPED ... light_contact=False` — **SC's one
recovery path is locked out precisely when it is needed.** The stall path then
returns False without ever unloading or retrying.

Proposed order:

1. **Detect WEDGED vs STALLED on SC** — depth pinned while axial force climbs.
   Reuse SFP's distinction rather than inventing one.
2. **Unload first** — back off ~1 mm and let it settle. Attempt 1 is empirical
   evidence this works: the plug seated the moment the push stopped.
3. **Keep the event dwell live during recovery.** Critical, and it must land
   with step 2. `_wait_for_insertion_event` (6 s, line 2069) already exists and
   works, but is only reachable from inside `_seat()` once depth passes
   `seat_candidate_depth_m` — the stall path at line 2665 never calls it. Since
   the plug seats *during* the unload, recovery without a live dwell will
   discard its own successes.
4. **Re-probe slowly, then retract-and-retry**, reusing SFP's bounds.
5. **Separately fix the 1.5 N visual-recovery lockout** — as its own path.
   Visual recovery answers "am I aimed wrong"; wedge recovery answers "I'm aimed
   fine and physically caught". Different problems, do not merge them.

Caveat: SFP's constants are tuned for a 42 mm insertion; SC seats ~1-2 mm past
its mouth. The structure ports cleanly, the thresholds must come from SC traces,
not from scaling SFP's by guess.

## Also outstanding

- **Perception admission fix** (attempt 2's coin flip): gate on the centre
  keypoint, which is what position actually uses. Note the corners also drive
  cross-camera relabelling, size rejection and candidate scoring, so they need
  their own scoped gate — "bad corners, fall back to a yaw prior" needs a
  verified yaw source first.
- `for _ in range(SC_PERCEPT_SAMPLES)` at line 3320 runs all 7 samples
  unconditionally with no early exit at `MIN_AGREE`. Raising
  `RL_INSERT_SC_PERCEPT_SAMPLES` buys perception time for nothing.
- Never abort with most of the budget unused (8.6 s used of 144 s).
- Geometry contract mismatch: triangulated openings measure ~25.9 x 6.06 mm
  against the declared 22.41 x 8.10 mm. Not what fails the gate (the gate is
  model-free) but it biases candidate *ranking* via `residual * 250.0` at line
  3070, and it means corners and contract disagree about which physical feature
  is being detected.

## Repository state

- `aic-model-flowstate-20260727` @ `bd1ae9d`, pushed. **Not merged to `main`.**
- The main worktree `/home/Anshul/AIC_Phase_1/aic_0/aic` is on `main` @ `1005d78`,
  **6 commits behind origin/main**, and dirty with ~1800 lines of SC visual
  alignment / seating stiffness work that is **not** part of this commit. Do not
  reset or overwrite it.
