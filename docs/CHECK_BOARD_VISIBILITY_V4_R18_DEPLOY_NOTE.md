# Check Board Visibility v4 r18 deployment

Date: 2026-07-17

Branch: `board-search`

## Target

```text
organization:  tar-2@xfa-prod-aic-us
solution:      9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH (Work on this)
cluster:       vmp-efe2-fz3vn7q3
asset:         ai.tar2.check_board_visibility_skill_v4
installed:     ai.tar2.check_board_visibility_skill_v4.0.0.1+bd2899881538160197a732eb6eedbe75d0c943069e19295ed62c9dd7777b4a24
local image:   flowstate:check-board-visibility-v4-r18
image digest:  sha256:7efe12facdd2eac935c1db7d168621f1bfa10bde8f885744c2ac02285b3b3d99
image SHA256:  6619fdf14ff68ffc7d4a508b15150c6c2e5914084b39d1f77e8d4eeb10190414
bundle SHA256: c74c564c9712f504ac6b7ed03cac0e2337710e330a9b02c5645d08e1bb273b01
```

## Live-trace fixes

- Calibrated gripper-mask pixels remain excluded from segmentation and cannot
  be selected as the task board. Mask-boundary contact is now diagnostic only;
  it no longer vetoes a board whose actual boundary, component envelope,
  detail, and shape are all complete.
- This addresses the right-camera trace which had no clipped edges from
  iteration 5 onward but remained blocked solely by
  `artificial_bottom_contact`. It can now complete after two fresh right-camera
  frames instead of taking four later 60 mm clearance moves.
- Before every return, the skill publishes one final measured-state hold so no
  board-search trajectory remains in flight during controller handoff.
- Missing wrist force during the initial controller transition waits up to the
  configured sensor budget (10 seconds by default) rather than failing after a
  hard-coded two seconds.
- Expected sensor/search failures return `success=false` normally instead of
  throwing. This lets a serial behavior tree execute its controller cleanup on
  either outcome.

## Required process wiring

The installed `ai.tar2.aic_phase_1_current` process was inspected during this
deployment. Its board-search node is followed by pose estimation and MoveRobot,
while its only `switch_to_default_controller_skill` node is much later, after
`insert_cable_skill`. That wiring causes MoveRobot to report `Part: 'arm' is
already in use.`

The process must be edited to run:

```text
Switch To AIC Controller
Check Board Visibility v4
Switch To Default Controller
Require board_result.success && board_result.done
IVM / Move Robot (including move to pregrasp)
```

The AIC controller bridge owns the Flowstate `arm` lease. A custom ROS skill
can stop its own command stream, but only `Switch To Default Controller`
releases that lease. TF lookup and J1/J6 target-mode changes do not own or
release it.

## Verification

```text
Windows source tests:       141 passed
Linux/amd64 baked tests:    141 passed
Python byte compilation:    passed
git diff --check:           passed
runtime startup:            gRPC listening on port 8003
Flowstate skill list:       exact r18 installed asset confirmed
solution state:             SOLUTION_STATE_RUNNING_IN_SIM
```

The source changes remain uncommitted in the local working tree. No existing
user changes were reset, stashed, or overwritten.
