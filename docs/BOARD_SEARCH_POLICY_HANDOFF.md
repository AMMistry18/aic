# Board-search policy handoff

Date: 2026-07-18

Branch: `board-search`

## Delivery status

The implementation source used for the installed bundle is committed and
pushed to `origin/board-search` at:

```text
6b612e6a9cbce7caf04cb3ddb072b054bea2386b
```

It was built and installed in the running **Work on this** solution on cluster
`vmp-efe2-hv8d2ahu` as the existing v4 skill:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+9090ff0afdffa69d0e35897dcef204c82c744d663f793ceddffd935592b3c960
```

The install completed on 2026-07-18 at approximately 22:33 CDT. The solution
remained `SOLUTION_STATE_RUNNING_IN_SIM`. No other skill or service was rebound.

```text
image tar SHA-256:  09b8377b6991f7f8c8f7b4bd534cb9569a7f0663a5da36cc29987981e4b04b1d
bundle tar SHA-256: 9332ed00cab8129e69e8c4557309d002210fcbe8c4d3026dee17cc582ff3cc98
```

## Required behavior

The policy is a measured, camera-driven phase machine:

```text
ACQUIRE/CENTER (J1)
        -> ALIGN LONG EDGE (J6, <= 2 degrees)
        -> LEVEL CENTER CAMERA STRAIGHT DOWN (primarily J2-J4)
        -> CLEAR THE GRIPPER MASK + SET 26-36% SCALE (J2-J4)
        -> DONE after two strict center-camera survey frames
```

Left and right cameras are acquisition hints. They may help select the initial
J1 direction, but they never satisfy completion. Completion is based on the
center camera only, after J6 alignment and a fresh physical top-down TF check.

The gripper masks still exclude calibrated robot pixels before board component
selection, but they are no longer diagnostic-only. The selected board convex
hull is expanded by the component-context padding to create a protected survey
envelope. Completion requires zero overlap between that envelope and the mask
and at least 20 pixels of separation. The report's measured escape vector
drives J2-J4 away from the mask.

## Root cause of the no-motion run

The failing trace requested this valid correction:

```text
joint1=-0.5760
target_joint1=-0.5132
delta=+0.0628 rad
```

At the old `0.12 rad/s` profile rate, the minimum-jerk command required:

```text
1.875 * 0.0628 / 0.12 = 0.981 seconds
```

`RobotMotion` embedded the latest controller target mode inside each
asynchronously received `/joint_states` reference and expired that mode after
`0.5 seconds`. The per-sample guard then required both a fresh joint vector and
that still-fresh embedded mode. At `0.525 seconds`, it confused an aged mode
timestamp with stale measured joints, stopped the profile, and reversed the
small partial motion. This exactly explains why the UI appeared not to move.

The wrapper also selected fallbacks by searching message strings. The stale
feedback wording was not on the J1 fallback list, so the policy terminated
instead of trying its Cartesian base-yaw route.

## Implemented source changes

### Direct-joint transactions

File: `flowstate/aic_perception/aic_perception/robot_motion.py`

- Added machine-readable `MotionFailure` values.
- Joint feedback freshness is now checked independently from controller-mode
  freshness.
- Joint mode must still be confirmed on a newer measured sample before a
  profile starts.
- During the profile, a genuinely stale measured joint vector aborts.
- A fresh explicit controller report that mode changed away from joint mode
  aborts and reverses.
- A temporarily unknown or aged mode report does not invalidate healthy
  measured joints.
- Settling accepts newer measured joints without requiring every sample to
  carry a synchronized mode timestamp.
- Direct J1/J6 profiles accept up to `0.20 rad/s`; the controller continues to
  enforce its native URDF position and velocity limits.
- The old artificial `0.30 rad` per-command rejection was removed. The planner
  still emits incremental, remeasured commands.

### Wrapper fallbacks and timing

File: `flowstate/aic_perception/check_board_visibility_skill.py`

- Direct J1 and J6 fallbacks now use `MotionFailure`, not log-message parsing.
- Default Cartesian speed increased from `0.04` to `0.05 m/s`.
- Default Cartesian angular speed increased from `0.20` to `0.30 rad/s`.
- Default direct-joint speed can reach `0.20 rad/s`.
- Default per-command timeout changed from `8` to `6 seconds`.
- Default workflow deadline changed from `150` to `90 seconds`.
- Legacy cumulative/start-relative motion-envelope proto fields remain
  accepted for existing Flowstate node compatibility, but the viewpoint policy
  no longer uses them as termination conditions.
- Wrist force remains the hard physical runtime stop. Fresh feedback is
  required before and throughout every movement.
- Cartesian leveling/clearance that redistributes motion through J1 or J6 now
  returns to visual centering and J6 alignment instead of failing on a drift
  envelope.
- Leveling continues while it makes measured top-down progress; it no longer
  fails after an arbitrary five-stage count.

### J6 accuracy and strict completion

File: `flowstate/aic_perception/aic_perception/viewpoint_search.py`

- Long-axis alignment tolerance is `2.0 degrees`, inclusive.
- Long-axis ratio must be at least `1.15` before an orientation estimate is
  trusted.
- Two fresh frames must agree on the signed correction before J6 moves.
- The old minimum J6 correction was `0.15 rad` (`8.6 degrees`), which could
  never settle accurately near a two-degree target. The minimum is now
  `0.02 rad` (`1.15 degrees`).
- A correction transaction may cover up to `0.45 rad` (`25.8 degrees`) before
  the next measured image, reducing unnecessary mode switches on large errors.
- After J1, J6, and top-down leveling, terminal evidence is deliberately
  rechecked in two consecutive fresh center-camera frames. Both must contain
  the full board and component context, logo identity, rectangularity >= 0.72,
  26-36% board area, long-axis ratio >= 1.15, orientation error <= 2 degrees,
  top-down TF, zero protected-envelope mask overlap, and >= 20 px mask
  separation.

### Vertical J2-J4 visual servo

Files:

- `flowstate/aic_perception/check_board_visibility_skill.py`
- `flowstate/aic_perception/aic_perception/viewpoint_search.py`

The 21:52 live trace proved that geometric top-down leveling reduced camera
tilt correctly but moved the board the wrong way in the image:

```text
before leveling: center_y=+0.241, tilt=0.202 rad
after stage 1:   center_y=+0.344, tilt=0.109 rad
after stage 2:   center_y=+0.423, tilt=0.027 rad
```

The old ASCEND ordering then selected optical-axis backoff before vertical
camera-plane centering. Backoff preserved the bad lower-edge projection and
introduced enough J1/J6 IK drift to restart alignment.

The revised policy now:

- adds signed image-plane vertical correction to each top-down leveling stage;
- interprets positive center Y as a board low in frame and negative center Y
  as a board high in frame;
- prioritizes bounded bidirectional camera-plane centering through J2-J4 before
  optical backoff;
- compares the next fresh center frame with the pre-move vertical error; and
- reverses the learned image-Y polarity when the absolute error worsens toward
  the same edge, preventing repeated wrong-way J4/arm compensation.

The camera orientation target remains physically top-down during these moves;
vertical centering is not achieved by accepting a slanted terminal camera.

### Deterministic gripper-clear survey controller

The 22:10 live run proved that the old terminal predicate was too weak. It
accepted a geometrically complete and top-down center view at area `0.291`,
rectangularity `0.77`, orientation error `0.7 degrees`, and long-axis ratio
`1.56`, but it also logged `gripper_mask_contact=True`. IVM returned estimates,
yet the NIC filter found only three physical rails because the gripper still
occluded the lower-right detail region. The rail filter correctly refused to
invent the two missing cards; score/count thresholds were not relaxed.

The revised controller adds these measured quantities to every center-camera
report and log line:

- `gripper_overlap_px`: overlap between the calibrated mask and the protected
  board/component envelope;
- `gripper_clearance_px`: minimum separation after overlap reaches zero; and
- `gripper_escape_direction`: normalized desired board-image displacement away
  from the mask.

While leveling and during the final survey phase, the camera moves opposite
that image escape vector through J2-J4. The next fresh frame validates overlap
and clearance; if separation worsens, the image-Y polarity reverses. Once the
mask is clear, the old generic vertical-centering servo no longer pulls the
board back behind the gripper unless a physical/context edge is actually
clipped or the vertical displacement exceeds 35% of the image.

The final scale controller backs away above 36% board area and makes up to
three bounded approaches below 26%. Expanded component context must clear all
four physical image edges by at least `1.25 * context_pad_px`. These constraints
keep all five NIC rails and both SC rail/port regions usable without guessing
from partial detections.

J1 centering also learns the observed horizontal image response per signed
command scale. After the first measured correction, later J1 steps use that
live response with a bounded 85% correction target, reducing repeated small
motions while retaining fresh-frame verification.

## Limits that remain

The policy intentionally retains execution correctness requirements:

- finite numeric commands;
- an available controller subscriber;
- confirmed controller mode at direct-joint transaction start;
- fresh measured feedback;
- cancellation and an overall workflow deadline;
- per-command timeouts so an unresponsive controller cannot block forever;
- native controller joint/velocity limits; and
- fresh wrist-force feedback with the configured absolute/delta force stop.

Arbitrary cumulative translation, cumulative rotation, start-relative
workspace, start-relative joint, and fixed leveling-stage limits no longer stop
the search.

## Verification

Run from the repository root:

```powershell
$env:PYTHONPATH=(Resolve-Path 'flowstate/aic_perception').Path
python -m pytest -q flowstate/aic_perception/test
```

Result for this revision:

```text
140 passed
```

The tests include regressions for:

- the exact `0.0628 rad`, `0.981 second` J1 profile continuing when the
  independent mode timestamp has aged out;
- explicit controller mode change still reversing the transaction;
- preserving all non-requested joints during direct J1 and J6 motion;
- two-degree J6 tolerance and fine correction size;
- strict phase order;
- side cameras never finishing the search;
- two consecutive strict post-level center frames being required;
- protected-envelope mask overlap, clearance, and escape direction;
- mask-escape polarity reversal when a fresh frame gets worse;
- bounded approach when a complete board is too small for IVM detail;
- learned J1 image response increasing centering efficiency;
- top-down completion being checked from fresh TF; and
- high/low vertical correction, fresh-frame polarity validation, and reversal
  after a wrong-way response; and
- controller handoff finalization order.

Also run before any future bundle:

```powershell
python -m py_compile `
  flowstate/aic_perception/aic_perception/robot_motion.py `
  flowstate/aic_perception/aic_perception/viewpoint_search.py `
  flowstate/aic_perception/check_board_visibility_skill.py
git diff --check
```

## Expected next live trace

The first retest should show:

1. J1 action planned with `control=direct_shoulder_pan_joint`.
2. Direct J1 completion with a measured joint delta close to the request.
3. A fresh center frame and further J1 correction or centering confirmation.
4. J6 sign confirmation, then direct J6 corrections.
5. J6 accepted only at `<= 2.0deg`.
6. J2-J4 top-down leveling with a logged signed vertical correction.
7. If the board is still high/low, `action=translate` occurs before backoff.
8. The next fresh frame either validates the direction or logs
   `reversing image-y polarity` and commands the opposite direction.
9. `gripper_overlap` decreases to zero and `gripper_clearance` reaches at least
   20 px while physical component context remains clear.
10. Two consecutive strict survey candidates are observed, then the skill
    succeeds and releases its controller/arm resources in the existing
    finalizer.

The old message below must not recur merely at the 0.5-second mark:

```text
measured joint feedback became stale; joint 1 yaw reversed
```

If it does recur, compare actual `/joint_states` receipt timestamps rather than
loosening mode timing again; the revised message now means the measured joint
stream itself was unavailable.

## Flowstate wiring

Keep the behavior-tree controller handoff serial:

```text
Switch To AIC Controller
Check Board Visibility v4
Switch To Default Controller
Require board_result.success && board_result.done
IVM / Move Robot
```

The board-search skill publishes a measured-state hold before returning, but
`Switch To Default Controller` is still the node that releases the shared
Flowstate `arm` lease. Do not put result validation, IVM, or Move Robot before
that switch.

Before building, inspect the node's explicitly stored parameter values. A
nonzero value saved in Flowstate overrides the new source default even though
the manifest and code changed.

## Known limitations and next-agent guidance

- The bundle/install and service-start smoke checks passed. The new
  protected-envelope controller still requires a task run to validate its
  physical mask-clear convergence and downstream five-NIC/SC detection yield.
- The wrapper still contains legacy envelope branches. They are inert because
  runtime envelope values are infinite after parameter validation. Removing
  that dead compatibility code is optional cleanup, not required for the next
  test.
- Do not relax the NIC filter's five-rail requirement. A missing rail now means
  the survey view still failed to expose all physical cards; inspect the new
  overlap/clearance/scale diagnostics first.
- Do not relax the two-degree J6 goal. If convergence oscillates, inspect the
  signed angle and measured J6 delta first.
- Do not reinstall the last asset identity expecting new code. Flowstate assets
  are content-addressed; a future authorized build must produce and install a
  new digest.

## Documentation cleanup performed

Documents tracked on `origin/main` were retained. Custom Flowstate deployment,
board-search, IVM, and perception documents were retained. Two stale custom
training handoffs unrelated to the current Flowstate policy were removed:

- `docs/ALIGN_RL_CODEX_HANDOFF.md`
- `docs/SC_PORT_TEACHER_HANDOFF.md`
