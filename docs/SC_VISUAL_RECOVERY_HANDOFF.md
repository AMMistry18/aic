# SC camera-guided seating recovery handoff

Updated: 2026-07-27

## Status

The canonical runtime now uses the five-keypoint physical-front-mouth model.
Absolute blue/dark-bore prealignment is therefore disabled by default
(`RL_INSERT_SC_VISUAL_ALIGN_ENABLE=0`) instead of correcting a target that
already names the physical mouth. It remains available for controlled A/B
evaluation. Stall-only relative recovery remains gated and opt-in configurable;
the latest diagnostic produced no recovery motion because only one camera was
usable after contact.

The recovery is **implemented and unit-tested in the current worktree**. It
still requires the simulator completion gate at the end of this document before
it can be called field-validated.

The controller now has:

- multi-camera SC plug pose estimation;
- seven-sample SC port-pose consensus;
- a stationary, one-time visual opening refinement before alignment;
- a 0.30 mm lateral alignment tolerance;
- force/moment-based seating nudges; and
- bounded wall-time stall detection;
- a clean pre-contact blue-housing side-band reference for each camera;
- paired baseline/current support masks, so a stable calibrated gripper mask is
  allowed but a new or changed occlusion still aborts recovery safely;
- rectified blue-side occlusion measurement after a shallow stall;
- two-camera directional agreement before each 0.25 mm recovery step;
- independent 2.0 mm cumulative-path and radial-excursion caps;
- light-contact force and projected-resolution safety gates;
- recovery-offset persistence through resumed seating and event dwell; and
- depth-gated insertion-event confirmation.

The 2026-07-27 reliability update also makes the commanded impedance match the
SC force model: seating/recovery use 90 N/m in the two port-frame lateral axes
and 500 N/m on the port insertion axis.  Thus the 2 mm recovery hold is
actually a 1 N hold, rather than the former 0.18 N produced by a 90 N/m
all-axis command.  In free-space pre-contact alignment, a bounded 1.5 mm
lateral segment is now retained until reached and uses a modestly stronger
isotropic 200 N/m impedance; this removes the prior per-cycle force cap that
made multi-millimetre handoff residuals crawl into the 15 s alignment timeout.

`SC_TIMING`, `SC_PERCEPTION_TIMING`, `SC_ALIGN_TIMEOUT`,
`SC_VISUAL_RECOVERY_SUPPORT`, and `ACTION_DEADLINE_*` now expose phase wall
time, simulation time, remaining action budget, final alignment residuals,
image-frame age, and paired-mask support state.  Use these before changing any
timeout or speed default.

The recovery uses 10 px and 12 px canonical side bands.  Both band widths and
at least two fresh camera views must agree.  Missing gripper masks, stale
frames, weak references, *new or changed* side support, or disagreement all
result in no lateral command.  A stable partial gripper mask is paired with its
own pre-contact reference instead of being rejected wholesale.

The SC visual-alignment baseline first landed in commit `2bf6715` (`Add SC
visual alignment and crop-refined pose evaluation`).  The current focused SC
recovery/policy suite and broader explicit regression suite are listed under
the test contract below; do not quote a global test count when unrelated test
modules have collection prerequisites.

## Latest field evidence

The 2026-07-26 run had stable raw perception:

- SC plug pose: 3 views, 1.47 px reprojection, confidence 0.778.
- SC port pose: 7/7 consensus, approximately 3.00 px reprojection.
- Raw detected rectangle: approximately 8.05-8.13 x 4.46-4.50 mm.
- Rigid 8.8 x 6.0 mm fit shift: approximately 0.30-0.32 mm.

The stationary prealignment refinement behaved safely but contributed no
correction:

```text
SC_VISUAL_NO_CORRECTION phase=prealign reason=view_disagreement
SC_VISUAL_LOCK phase=prealign accepted=0/7 need=4
action=freeze_raw_pose_and_continue
```

The raw target then aligned to 0.20 mm lateral error and 0.07 degrees rotation.
Seating reached the mouth, stopped at about 0.30 mm reported depth, and remained
near 0.7-1.2 N axial load until the stall timeout. Every seating log reported
`nudge_applied_mm=[0.0, 0.0]` because the force-based correction currently
activates at 2.0 N.

The camera image showed that the plug was too low in the physical blue opening:
the gray/white plug geometry overlapped and phased through the blue housing.
The necessary correction in that view was upward in the **port plane**. It was
not a valid insertion.

A matching simulator insertion event arrived after the shallow stall while the
geometry was still phasing. Therefore an event by itself is not sufficient
evidence of SC seating at shallow depth.

## Native-fixture audit of the recovery cue

The original gray-plug/nearest-blue-gap detector was not usable at contact:
all 176 controlled shifted-plug cases from the native SC-pose images became
zero-gap/balanced at the moment the plug touched the blue housing.

The replacement was checked against 51 labelled native empty-port captures in
`/home/rschnurr/aic_perception_data/pose_sc` by rendering the measured
20.0 x 6.4 mm plug into the real images.  The 14 px projected-edge gate kept
48 usable views.  It produced the correct signed recovery direction in all of
the following controlled offline cases:

- 192 four-direction shifts (left, right, high, and low); and
- 1,536 shifts with an additional 0.30 mm common expected-port bias swept
  through eight directions.

This is evidence that the cue is robust to the observed common-centre pose
bias; it is not a substitute for the simulator completion gate below.

## Diagnosis

The 0.30 mm alignment gate was satisfied relative to the estimated port pose,
but the estimate retained a common-center bias. The failure was not alignment
loop accuracy: the controller accurately reached a biased target.

Do not address this by:

- loosening the 1.0 mm cross-camera disagreement gate;
- changing the 0.30 mm final alignment tolerance;
- continuously replacing the target during approach;
- applying a hard-coded global upward offset; or
- treating the late insertion event as success.

The prealignment target must remain stationary/frozen. The missing behavior is
a **relative, stall-time image measurement** that can determine which
port-local direction has physical clearance.  The key insight is to compare
each camera to its own clean view before approach: overlap removes blue pixels,
so the signal remains informative when geometric clearance is exactly zero.

## Required recovery behavior

Add an SC-only recovery state that activates after a shallow seating stall,
while the plug is held at light contact.

1. Freeze axial advance and hold approximately 0.5-1.5 N contact.
2. Acquire fresh usable wrist-camera frames.
3. Project the known SC bore geometry into each image and rectify the ROI into
   the two port-local lateral axes.
4. Before approach motion, capture a fresh masked blue-housing reference for
   each usable camera after the final target is frozen.
5. At the stall, measure current blue coverage in 10 px and 12 px left, right,
   top, and bottom bands, normalized by that camera's clean reference.
6. Move toward the side with more retained blue coverage and away from the
   covered/overlapped edge. In the captured failure, this must produce a
   port-local upward correction.
7. Require the two band widths to agree, then fuse signed evidence from usable
   cameras. Contradictory or weak views must
   result in no lateral command, not an arbitrary search direction.
8. Apply one approximately 0.25 mm port-local lateral step, reacquire imagery,
   and repeat.
9. Cap total visual-recovery excursion at approximately 2.0 mm and keep the
   existing lateral safety envelope.
10. Stop recovery immediately when depth advances meaningfully, reset the
    progress watch, and resume normal seating from the corrected position.

This is deliberately a **relative visibility** measurement. The existing
`ray_to_plane` absolute point estimates disagreed across views in the field run;
loosening that gate would hide the calibration error rather than solve it.

## Suggested integration points

Authoritative source:

- `aic_model/aic_model/sc_visual_alignment.py`
- `aic_model/aic_model/sc_controller.py`
- `aic_model/test/test_sc_visual_alignment.py`
- `aic_model/test/test_sc_controller.py`

Deployment mirrors that must remain byte-for-byte consistent:

- `docker/aic_model/v50_overlay/aic_model/sc_visual_alignment.py`
- `docker/aic_model/v50_overlay/aic_model/sc_controller.py`

`sc_visual_alignment.py` already provides:

- blue-housing association;
- projected duplex bore ROIs;
- gripper-pixel masking at the controller boundary;
- camera projection and port-plane helpers; and
- bounded lateral target updates.

Keep the new image-only side-visibility estimator ROS-free and deterministic in that
module. Keep camera acquisition, force/depth state, port-frame transforms,
commands, and logging in `SCController`.

Add explicit `SC_VISUAL_RECOVERY` logs containing at least:

- activation depth and force;
- cameras accepted/rejected and rejection reasons;
- per-view port-local signed direction/confidence;
- fused direction;
- applied step and cumulative excursion; and
- exit reason (`depth_advanced`, `weak_evidence`, `cap_reached`, or safety).

Use separate recovery configuration from the existing prealignment parameters.
Recommended starting defaults:

```text
enable=true
step=0.00025 m
max_total=0.0020 m
max_axial_force=1.5 N
meaningful_depth_advance=0.0005 m
baseline_samples=3
baseline_min_samples=2
```

Do not lower the general 2.0 N force-nudge threshold merely to make this case
move. The field lateral-force vector changed direction and is not reliable
enough to choose the recovery direction by itself.

## Event confirmation

`SCController._event_status()` currently accepts a matching event immediately,
without considering depth. Preserve wrong-port handling, but ensure a matching
SC event observed at a shallow mouth stall cannot terminate the attempt as
`SEATED`.

The event may remain the final scoring confirmation once the controller has
reached credible seating geometry, such as the existing
`seat_candidate_depth_m` path. Keep the geometry/event rule SC-specific and
cover it with tests so SFP behavior is unchanged.

## Deterministic test contract

Add tests that prove:

1. A synthetic blue housing with the gray plug too low yields a port-local
   upward correction.
2. Mirrored left/right and top/bottom cases yield correspondingly mirrored
   corrections.
3. Camera rotation/perspective does not change the port-local direction after
   rectification.
4. Agreeing views fuse; contradictory views command no motion.
5. Weak blue association or an occluded ROI commands no motion.
6. Missing, stale, masked, or weak pre-contact side references command no
   motion.
7. Each step is capped near 0.25 mm and cumulative recovery near 2.0 mm.
8. Recovery activates only after a shallow seating stall, not during ordinary
   approach or frozen prealignment.
9. Meaningful depth advance exits recovery immediately and resumes seating.
10. A shallow matching insertion event does not report `SEATED`.
11. A credible-depth matching event still reports `SEATED`.

Run at minimum:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="aic_model:${PYTHONPATH}" \
.pixi/envs/default/bin/python -m pytest -q \
  aic_model/test/test_sc_visual_alignment.py \
  aic_model/test/test_sc_controller.py
```

Then run the broader insertion suite from `docs/INSERTION_HANDOFF.md` and verify
the two Docker-overlay files match their source counterparts.

## Completion gate

Do not call this complete based only on unit tests. The next simulator run must
show:

- `SC_VISUAL_RECOVERY` activating at the shallow mouth stall;
- a camera-directed port-local correction matching the visible blue clearance;
- depth advancing after correction;
- no phasing through the blue housing;
- seating reaching credible depth; and
- the correct insertion event occurring at credible seating geometry.
