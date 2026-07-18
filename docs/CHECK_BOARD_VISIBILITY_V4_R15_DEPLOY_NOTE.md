# Check Board Visibility v4 r15 deployment

Date: 2026-07-16

Branch: `board-search`

## Result

The v4 policy was rebuilt and installed in the Flowstate **Work on this**
solution. The solution was started in simulation for the installation and is
running on `vmp-efe2-528fje20`.

```text
organization:  tar-2@xfa-prod-aic-us
solution:      9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
asset:         ai.tar2.check_board_visibility_skill_v4
installed:     ai.tar2.check_board_visibility_skill_v4.0.0.1+25de25defc75973294a59dd7b33b927ffaf35e4b47cbcf1874f717102a806ff5
local image:   flowstate:check-board-visibility-v4-r15
image digest:  sha256:f8924f8bfab4bb66da9c6080fb8fa17349fc91c446ab0e43ce6db920fb6c20e9
bundle SHA256: 0bc3d6975a6943fb8cefe3952adbd3bf8f9a1341382965ec6c6e66f04fdaedc8
```

If an existing Flowstate process node remains pinned to r14, refresh the skill
catalog or remove and re-add the v4 node so it resolves the r15 asset version.

## Changes

- The TCP/tool axis, rather than the center camera optical ray, is leveled to
  base `-Z`, keeping the physical camera module straight down.
- J6-equivalent yaw rotates about TCP/tool `Z` and aligns the taskboard's
  physical long edge to the image's longer pixel dimension.
- Long-edge estimation uses the broad plate core after removing narrow
  arm/gripper bridges and refuses ambiguous near-square observations.
- When J6 alignment is unavailable, horizontal error falls back to proportional
  J1 yaw. Otherwise, the policy performs at most two optical-axis zoom-outs,
  then changes to base `+Z` clearance rather than repeating backoff until the
  search deadline.
- The exact left, center, and right gripper masks from upstream commit
  `c0a686b` are integrated before component selection. A broad taskboard region
  touching a gripper mask is still marked incomplete, preventing a false pass
  under occlusion.
- A missing force sample at camera-capture time waits briefly for the next fresh
  wrench callback. Motion still refuses to start without fresh force and still
  aborts if feedback becomes stale during motion.

The unrelated upstream `move_to_board_skill` scaffold and CMake refactor were
not imported; the scaffold still uses the old immediate force refusal and was
not needed by the deployed v4 policy.

## Verification

```text
Windows source tests:       125 passed
Linux/amd64 runtime tests:  125 passed
Python byte compilation:    passed
git diff --check:           passed
runtime smoke:              gRPC listening on port 8003
Flowstate skill list:       exact r15 installed asset confirmed
```

The source changes remain uncommitted in the local working tree; no user changes
were reset, stashed, or overwritten.
