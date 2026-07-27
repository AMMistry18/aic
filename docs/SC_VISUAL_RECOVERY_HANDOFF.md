# SC camera-guided seating recovery handoff

Updated: 2026-07-26

## Status

This handoff describes the next SC insertion change. It is **not implemented**
at this revision.

The current controller has:

- multi-camera SC plug pose estimation;
- seven-sample SC port-pose consensus;
- a stationary, one-time visual opening refinement before alignment;
- a 0.30 mm lateral alignment tolerance;
- force/moment-based seating nudges; and
- bounded wall-time stall detection.

The current controller does **not** inspect the relative blue-port/gray-plug
margins after seating stalls. A shallow stall therefore ends the SC attempt
without a camera-directed lateral recovery.

Start from current `main`. The SC visual-alignment baseline first landed in
commit `2bf6715` (`Add SC visual alignment and crop-refined pose evaluation`).

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
a **relative, stall-time image measurement** that can determine which port-local
direction has physical clearance.

## Required recovery behavior

Add an SC-only recovery state that activates after a shallow seating stall,
while the plug is held at light contact.

1. Freeze axial advance and hold approximately 0.5-1.5 N contact.
2. Acquire fresh usable wrist-camera frames.
3. Project the known SC bore geometry into each image and rectify the ROI into
   the two port-local lateral axes.
4. Use the high color contrast to identify:
   - blue adapter/housing pixels; and
   - the occluding gray/white plug silhouette.
5. Measure signed blue clearance or overlap on the left, right, top, and bottom
   of the plug silhouette.
6. Move toward the side with more visible blue clearance and away from the
   covered/overlapped edge. In the captured failure, this must produce a
   port-local upward correction.
7. Fuse signed evidence from usable cameras. Contradictory or weak views must
   result in no lateral command, not an arbitrary search direction.
8. Apply one approximately 0.25 mm port-local lateral step, reacquire imagery,
   and repeat.
9. Cap total visual-recovery excursion at approximately 2.0 mm and keep the
   existing lateral safety envelope.
10. Stop recovery immediately when depth advances meaningfully, reset the
    progress watch, and resume normal seating from the corrected position.

This is deliberately a **relative margin** measurement. The existing
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

Keep the new image-only margin estimator ROS-free and deterministic in that
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
6. Each step is capped near 0.25 mm and cumulative recovery near 2.0 mm.
7. Recovery activates only after a shallow seating stall, not during ordinary
   approach or frozen prealignment.
8. Meaningful depth advance exits recovery immediately and resumes seating.
9. A shallow matching insertion event does not report `SEATED`.
10. A credible-depth matching event still reports `SEATED`.

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

