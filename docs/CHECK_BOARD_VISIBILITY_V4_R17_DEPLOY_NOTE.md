# Check Board Visibility v4 r17 deployment

Date: 2026-07-17

Branch: `board-search`

## Target

```text
organization:  tar-2@xfa-prod-aic-us
solution:      9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH (Work on this)
cluster:       vmp-efe2-fz3vn7q3
asset:         ai.tar2.check_board_visibility_skill_v4
installed:     ai.tar2.check_board_visibility_skill_v4.0.0.1+446598597c9eef917773f70fd297f783e49f71925d0ea227b040b1c1feb006f6
local image:   flowstate:check-board-visibility-v4-r17
image digest:  sha256:3c487b96a49efbc78a5976087e07c59384edcd2b3c220fe42d30e46f9eb2ef91
image SHA256:  b65c72937a779e1dc4d30ea7d660563932ea63898f0e3d990d0f0d02179e7721
bundle SHA256: 5a5e1bf1345a6b580086328faa9b6b33e6c7b4bee16eb8fd459a3aeaaf62678c
```

## Live-trace fixes

- The center-camera long-axis reliability threshold is 1.15 instead of 1.25.
  Two consecutive fresh frames must agree on the signed orientation before J6
  moves, so stable 1.16-1.24 observations are usable without trusting a
  one-frame edge flip.
- J6 alignment is an explicit phase after confirmed J1 centering and before
  clearance. Replaying iterations 11-12 from the attached trace now commands
  negative J6 at the second 1.23-ratio, approximately -46-degree frame.
- After the two bounded optical-axis retreats are exhausted, base +Z clearance
  uses a 2.5 scale (100 mm at the default 40 mm step) instead of 60 mm.
- The default overall deadline is 150 seconds. Per-move timeout, workspace,
  cumulative travel, force, feedback freshness, and cancellation bounds remain
  independent and unchanged.
- Completion remains any-camera; all three images are evaluated while the
  center camera alone supplies deterministic steering feedback.

## Verification

```text
Windows source tests:       140 passed
Linux/amd64 baked tests:    140 passed
Python byte compilation:    passed
git diff --check:           passed
runtime smoke:              gRPC listening on port 8003
Flowstate skill list:       exact r17 installed asset confirmed
deployed manifest markers:  ratio, confirmation, deadline, J1, and J6 confirmed
solution state:             SOLUTION_STATE_RUNNING_IN_SIM
```

The first r17 image candidate exposed a wrapper-scope `NameError` during the
runtime smoke test. It was fixed and the image was rebuilt before bundling or
uploading; only the corrected image above was installed.

The source changes remain uncommitted in the local working tree. No existing
user changes were reset, stashed, or overwritten.
