# Handoff — SC insertion reliability + SC pose-label geometry

**Date:** 2026-07-27  
**Workspace:** `/home/rschnurr/satya/aic`  
**Branch / HEAD:** `main` at `1005d78540e78f84f9c565b9b68a9783fde62612` —
`Document SC camera-guided recovery handoff`  
**Worktree:** intentionally dirty; do **not** reset, checkout, or overwrite
uncommitted work.

## User's current priority

The user looked at the SC pose drawing and said it is not around the actual SC
port. They asked for a render, not an inference from logs. The render confirms
the visual complaint is valid: the deployed model draws a virtual label
rectangle rather than a physical SC-mouth outline.

The next chat should decide whether to implement a truthful operator overlay,
then build/evaluate a new physical-mouth SC pose model. Do **not** silently
change the old pose geometry constants while retaining the old weights.

## What is already implemented in this worktree

### 1. Safer visual recovery

`aic_model/aic_model/sc_visual_alignment.py` and its Docker overlay now retain
immutable baseline/current support masks in `ScBlueSideSignature` and compare
only paired support. Stable calibrated gripper masking is allowed; new or
changed support rejects recovery safely. The legacy fraction-only path remains
full-valid only.

`ScInsertionController._prime_visual_recovery_baseline()` stores rich
per-camera/per-band signatures via `aggregate_sc_blue_side_signatures()`.
The two-view and two-band agreement gates remain intact.

### 2. Correct seating/recovery force contract

The SC configuration computes force leads using 500 N/m but previously sent
90 N/m in all translational axes. This meant the intended 1 N recovery hold
was actually about 0.18 N.

`ScInsertionController._seating_stiffness()` now sends a base-frame 6×6
matrix with 90 N/m in port-frame lateral axes and 500 N/m along the port
insertion axis. This is used for normal seating, recovery, and event dwell.
`Policy.set_pose_target()` now accepts legacy six-gain vectors and full 6×6 or
flat-36 Cartesian matrices, validates finite/symmetric/PSD values, and
serializes row-major values to `MotionUpdate`.

### 3. Faster pre-contact alignment and usable timeout telemetry

Before mouth contact, `_align()` retains a bounded 1.5 mm lateral segment until
it is reached, using 200 N/m instead of recomputing a soft 1.5 mm command from
the current tip every 50 ms. Near the mouth it preserves the previous
compliant behavior.

Added:

- `SC_TIMING` / `SC_TIMING_SUMMARY` phase wall time, sim time, and remaining
  action budget;
- `SC_PERCEPTION_TIMING` around costly seven-frame port consensus;
- `SC_ALIGN_TIMEOUT` final depth/lateral/rotation/command count;
- `SC_VISUAL_RECOVERY_SUPPORT` paired-mask diagnostics;
- image header stamp/age and remaining action budget on recovery logs;
- overlay-only `ACTION_DEADLINE_START`, `ACTION_DEADLINE_EXCEEDED`, and a
  clean `INSERTION_ABORT reason=action_deadline` instead of an ambiguous V50
  timeout traceback.

The deployed `RLInsert.py` is intentionally the Docker overlay copy; the
source and overlay versions differ by design. The deadline diagnostics were
added only to the deployed overlay.

## Verification already completed

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pixi run pytest -q \
  aic_model/test/test_sc_visual_alignment.py \
  aic_model/test/test_sc_controller.py \
  aic_model/test/test_policy_impedance.py
# 124 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pixi run pytest -q \
  aic_model/test/test_sc_visual_alignment.py \
  aic_model/test/test_sc_controller.py \
  aic_model/test/test_policy_impedance.py \
  aic_model/test/test_sc_plug_pose.py \
  aic_model/test/test_sc_plug_pose_geometry.py \
  aic_model/test/test_v50_controller.py \
  aic_model/test/test_rl_insert_contract.py
# 200 passed
```

`test_sc_plug_pose_trials.py` has a separate collection prerequisite:
`generate_sc_plug_pose_trials` is not importable in the direct test invocation.
Do not report that as a failure of these changes.

`py_compile`, `git diff --check`, and byte comparisons passed for the source /
overlay copies of:

- `sc_controller.py`
- `sc_visual_alignment.py`
- `policy.py`

## Critical rendered SC-pose finding

### Current convention

The active weight is:

```text
aic_example_policies/aic_example_policies/ros/weights/best_sc_pose.pt
```

It predicts the `DataCollectorScPoseGT` virtual rectangle:

```text
LOCAL_SC_PORT_KPS = 8.8 × 6.0 mm
```

in `aic_model/aic_model/sc_controller.py` around lines 439–489. This is **not
the physical mouth outline**.

The SC CAD asset gives:

| Feature | Dimensions |
|---|---:|
| physical front mouth | 22.407 × 8.10 mm |
| binding throat through the lip | 22.407 × 7.85 mm |
| visible outer blue face label convention | 25.78 × 9.27 mm |
| shipped virtual YOLO target | 8.8 × 6.0 mm |

The virtual target is only ~39% of physical-mouth width and ~29% of its area.
It straddles the central divider and does not reach the two bore centres
(±6.350 mm). It is mechanically centered at the entrance frame: the physical
front face is only 0.076 mm ahead of it; the asymmetric binding-throat centroid
is 0.125 mm away vertically. Therefore the old model is not selecting a
neighboring port, but it is visually misleading and weakly conditioned for
physical scale/yaw geometry.

### Direct visual evidence

Rendered CAD comparison (temporary artifact, same host/session):

```text
/tmp/sc_port_render.xJY5Ds/sc_port_front.png
green  = shipped 8.8×6.0 virtual YOLO label
yellow = physical front mouth
orange = binding throat
```

Rendered active `PerceptionCore.detect_sc_pose()` result:

```text
/tmp/sc_yolo_overlay.png
/tmp/sc_port_overlay_review.h2hX3e/left_live_overlay_crop_4x.png
/tmp/sc_port_comparison.l3vHOR/comparison_crop_4x.png
```

The live model's small quad clearly sits inside the blue port rather than
outlining it. On 51 local `pose_sc` outer-face-labelled observations, active
crop-refined inference matched every labelled port; the root audit measured
median center error 0.517 px, p95 2.675 px, max 6.812 px, and predicted/outer
feature-size ratio 0.416. An independent rendering audit measured comparable
center results (0.85 px median, 3.05 px p95). This confirms: **center is good,
the drawn geometry is the wrong physical feature.**

Important dataset caveat: the local `/home/rschnurr/aic_perception_data/pose_sc`
labels are an older outer-face convention. Three of 51 labels use
`sc_port_link` rather than the entrance frame, so do not treat this set as
unqualified 3D mouth ground truth without filtering/recollecting it.

## Recommendation / next implementation order

1. **Fix the operator visualization first, without changing control.**
   Draw the projected physical mouth/throat alongside the current virtual
   YOLO quad, label them honestly, and make the center/error obvious. This
   lets operators distinguish an intentional small label from a wrong pose.
   There is a small bug to fix while doing this: `draw_sc()` expects
   `d['area']`, but `detect_sc_pose()` returns no `area`, causing direct live
   YOLO visualization to raise `KeyError`. Use `d.get('area', w * h)` or add
   an area field in `_sc_pose_record`.

2. **Build a new physical-mouth pose dataset/model.**
   Prefer four corners of the actual duplex mouth/throat (or outer face if
   those edges prove substantially more visible) plus an explicit center
   keypoint. Generate TF-projected labels from the actual entrance/mouth
   geometry, collect images at deployment handoff angles, train a new
   checkpoint such as `best_sc_pose_mouth.pt`, and retain the old checkpoint
   for A/B comparison. Current crop refinement is already active by default:
   `AIC_SC_POSE_CROP_REFINE=1`, pad 24.

3. **Evaluate before switching.**
   Measure per-keypoint/center and 3D position/yaw against a clean held-out
   entrance-frame dataset. The success condition is lower mouth-center / yaw
   error and improved insertion behavior, not a prettier box alone.

4. **Switch weights and geometry atomically.**
   Update `LOCAL_SC_PORT_KPS`, `SC_OPENING_HYPOTHESES`, any PnP/multiview
   geometry, and deployed weight together. Do **not** substitute 22.407×7.85
   dimensions into the current code while keeping `best_sc_pose.pt`; that
   changes the 3D scale and makes the existing model pose wrong.

The recovery/stiffness work remains valuable, but it should not be expected to
permanently compensate for a virtual/nonphysical pose target.

Useful existing planning document:
`docs/WAYS_TO_MAKE_YOLO_POSE_BETTER.md`. It contains useful crop/PnP/data
augmentation ideas, but its opening claim that `pose_sc` does not exist is now
stale.

## Files intentionally modified / added

```text
M  aic_model/aic_model/policy.py
M  aic_model/aic_model/sc_controller.py
M  aic_model/aic_model/sc_visual_alignment.py
M  aic_model/test/test_sc_controller.py
M  aic_model/test/test_sc_visual_alignment.py
M  docker/aic_model/v50_overlay/aic_model/RLInsert.py
M  docker/aic_model/v50_overlay/aic_model/policy.py
M  docker/aic_model/v50_overlay/aic_model/sc_controller.py
M  docker/aic_model/v50_overlay/aic_model/sc_visual_alignment.py
M  docs/SC_VISUAL_RECOVERY_HANDOFF.md
?? aic_model/test/test_policy_impedance.py
```

`pixi.lock` is also modified as a binary side effect of a local Pixi package
reinstall. It was not part of the SC implementation; inspect it separately and
do not blindly overwrite user work.

