# Check Board Visibility v4 r19 deployment

Date: 2026-07-17

Branch: `board-search`

## Target

```text
organization:          tar-2@xfa-prod-aic-us
solution:              9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH (Work on this)
current cluster:       vmp-efe2-acv4u3yz
asset:                 ai.tar2.check_board_visibility_skill_v4
installed:             ai.tar2.check_board_visibility_skill_v4.0.0.1+67440e360b6107cb1a863e8a57c050a5f9526dbe3f6dbef8706908373b464826
local image:           flowstate:check-board-visibility-v4-r19-final
local image digest:    sha256:4549862d8bc7b3b70e82cfb98906b53ed66f4afa1514f78342c93fad239f709b
uploaded image digest: sha256:63aacdf15983d687ba45cc3fdaa969c502488625ffee70c92167ece1f08b432d
image SHA256:          c2ba49d990d2bc0087c5d2d80ca01aaf14b5755b2449f47274f2ff703e327e2e
descriptor SHA256:     5b18ece033a5969d002ae66e4052f7e3576cb3b03853021ae0f5b71b5a15b9fc
bundle SHA256:         77ad15bca3b59370eedf9e10b781d583343b27bae32ad970f81cb47931d28292
```

The solution had moved from the r18 cluster
`vmp-efe2-fz3vn7q3` to `vmp-efe2-acv4u3yz`. The r19 bundle was installed
through the solution ID so the upload targeted the current cluster.

## Live-trace fixes

- Completion is no longer accepted from an oblique side camera. Left and right
  camera reports remain steering evidence, but only the center camera can
  release downstream IVM.
- The center camera must hold the strict IVM survey predicate for two fresh
  frames: full plate and logo, area no greater than 0.45, rectangularity at
  least 0.70, bounded horizontal and vertical centering error, long-axis ratio
  at least 1.15, long-axis error no greater than 12 degrees, no calibrated
  gripper-mask contact, and component clearance at least 1.5 times the normal
  context pad.
- This specifically rejects the logged false exit in which the right camera
  had a 65-degree long-axis error, 0.63 rectangularity, and gripper contact
  while the center camera was still clipped. That view left IVM with only four
  of five NIC rails.
- A center-camera view that is merely full now continues through bounded
  recenter, yaw, zoom-out, and clearance recovery until it has the context
  needed for NIC, SFP, and SC pose estimation.
- The runtime launcher creates a per-process ROS log directory under `/tmp`
  before starting the skill, avoiding shared log-directory permission
  failures.
- The asset manifest now describes center-camera survey completion accurately.
  Flowstate's installed skill metadata was checked for the new wording.

## Required process wiring

Keep the controller handoff serial:

```text
Switch To AIC Controller
Check Board Visibility v4
Switch To Default Controller
Require board_result.success && board_result.done
IVM / Move Robot (including move to pregrasp)
```

`Switch To Default Controller` must run before result validation and before
MoveRobot. The custom board-search skill stops its measured-state hold before
returning, but the controller-switch node is what releases the Flowstate
`arm` lease.

## Verification

```text
Windows source tests:       144 passed
Linux/amd64 image tests:    144 passed
Python byte compilation:    passed
git diff --check:           passed
runtime startup:            gRPC listening on port 8003
Flowstate skill list:       exact r19 installed asset confirmed
deployed metadata:          strict center-camera survey wording confirmed
solution state:             SOLUTION_STATE_RUNNING_IN_SIM
```

No existing user changes were reset, stashed, or overwritten while resolving
and publishing the branch.
