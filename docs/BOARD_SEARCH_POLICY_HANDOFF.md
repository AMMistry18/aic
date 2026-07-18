# Board-search policy handoff

Date: 2026-07-18

Branch: `board-search`

## Delivery status

The current source is committed and pushed to `origin/board-search` at:

```text
9c9765043702514b5785a927bc6913b7fe6ed43f
```

It was built and installed in the running **Work on this** solution on cluster
`vmp-efe2-hv8d2ahu` as the existing v4 skill:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+56b2c76bb0b225bb32ceb98fb684d2365bb604313491ab9cd6513e859243e2be
```

The install completed on 2026-07-18 at approximately 22:06 CDT. The solution
remained `SOLUTION_STATE_RUNNING_IN_SIM`. No other skill or service was rebound.

## Required behavior

The policy is a measured, camera-driven phase machine:

```text
ACQUIRE/CENTER (J1)
        -> ALIGN LONG EDGE (J6, <= 2 degrees)
        -> LEVEL CENTER CAMERA STRAIGHT DOWN (primarily J2-J4)
        -> FRAME THE COMPLETE BOARD (J2-J4)
        -> DONE on the first full center-camera board mask
```

Left and right cameras are acquisition hints. They may help select the initial
J1 direction, but they never satisfy completion. Completion is based on the
center camera only, after J6 alignment and a fresh physical top-down TF check.

The gripper masks exclude calibrated robot pixels before board component
selection. Contact with the conservative mask boundary remains diagnostic; it
does not veto a complete real board boundary.

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

### J6 accuracy and completion

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
- After J1, J6, and top-down leveling are established, the terminal predicate
  is exactly a full center-camera board mask. It does not rerun the noisy
  long-axis estimator on the terminal image and does not add a stability delay.

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
134 passed
```

The tests include regressions for:

- the exact `0.0628 rad`, `0.981 second` J1 profile continuing when the
  independent mode timestamp has aged out;
- explicit controller mode change still reversing the transaction;
- preserving all non-requested joints during direct J1 and J6 motion;
- two-degree J6 tolerance and fine correction size;
- strict phase order;
- side cameras never finishing the search;
- first full post-level center frame finishing immediately;
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
9. Immediate success on the first full center-camera board mask.

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

- The bundle/install and service-start smoke checks passed. The new vertical
  response controller still requires a task run to validate its physical
  direction and convergence in the simulator.
- The wrapper still contains legacy envelope branches. They are inert because
  runtime envelope values are infinite after parameter validation. Removing
  that dead compatibility code is optional cleanup, not required for the next
  test.
- Do not change masking or segmentation thresholds in response to the no-motion
  trace. Perception successfully selected the center board and the planner
  requested the correct J1 direction.
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
