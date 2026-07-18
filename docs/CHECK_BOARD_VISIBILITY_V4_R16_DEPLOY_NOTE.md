# Check Board Visibility v4 r16 deployment

Date: 2026-07-17

Branch: `board-search`

## Target

```text
organization:  tar-2@xfa-prod-aic-us
solution:      9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH (Work on this)
cluster:       vmp-efe2-fz3vn7q3
asset:         ai.tar2.check_board_visibility_skill_v4
installed:     ai.tar2.check_board_visibility_skill_v4.0.0.1+ed45ed3b042f83da96c2f959d74d7fac9b90a5ce37495efae705bbe8911c8582
local image:   flowstate:check-board-visibility-v4-r16
image digest:  sha256:dab2b508a2ce7abee7cd0b652fc4859044ceef0cb0d6881435199a023bdb00ab
image SHA256:  e2b5c1f9b135997612cfe4e63421f05b5ef4aee059411df7664179db1bbb913b
bundle SHA256: 4380e9eebab91151840b5dc3cd2f16c5095287ed5241cd6ecd918d974eb70b72
```

## Motion fixes

- J6 yaw is a direct joint-mode command to UR5e `wrist_3_joint`; the other
  five measured arm targets are preserved and Cartesian mode is restored.
- Long-edge alignment uses up to six bounded 0.30-rad J6 corrections rather
  than a Cartesian IK roll that can redistribute motion across the arm.
- When direct J6 is unavailable, safely rejected, or safely reversed after a
  settling failure, the planner falls back to measured J1 centering or the
  joints 2-4 zoom/clearance path.
- Straight-down TCP pitch leveling happens only after the board is framed. It
  uses bounded 0.12-rad stages with base-Z clearance instead of blocking the
  entire search with a fixed-position orientation solve.
- Cartesian settling failures now include measured position, orientation,
  linear-speed, and angular-speed residuals.

Force-triggered aborts, stale force/controller feedback, cancellation, and a
failure to restore Cartesian mode remain terminal safety failures.

## Verification

```text
Windows source tests:       131 passed
Linux/amd64 baked tests:    131 passed
Python byte compilation:    passed
git diff --check:           passed
runtime smoke:              gRPC listening on port 8003
Flowstate skill list:       exact r16 installed asset confirmed
manifest markers:           direct joint mode and bounded pitch stages confirmed
solution state:             SOLUTION_STATE_RUNNING_IN_SIM
```

The source changes remain uncommitted in the local working tree. No existing
user changes were reset, stashed, or overwritten.
