# Perception fixes — TODO (deferred)

Created 2026-07-12. These are DEFERRED code changes to `aic_model/aic_model/RLInsert.py`.
Do them AFTER the Flowstate-side positioning / offset work is settled. Not yet
applied — the working tree currently uses the v10/v12 baseline perception
(single-frame, all 3 cameras, nearest-tip).

## Why (evidence)

A 3x repeatability probe on 2026-07-12 (offset X=160, Z=-3, unchanged between
runs) produced these perceived port positions:

| run | perceived port Y | reproj | note |
| --- | --- | --- | --- |
| 1 | 0.38602 | 1.4px | correct port |
| 2 | **0.34636** | 1.2px | **WRONG port — ~40mm off in Y** |
| 3 | 0.38626 | 0.9px | correct port |

Run 2 locked onto a neighboring NIC/port (screenshot showed `nic_card_mount_1`)
with CLEAN reproj (1.2px). So ~1/3 of runs silently target the wrong port. The
plug then descends onto the wrong cage and aic_engine reports "Plugs are not
within max bounding radius from target ports."

Root cause: `_select_sfp_candidate` picks purely `min(distance-to-tip)` across
all detected candidates and does NOT filter by (a) a max plausible handoff
distance or (b) cross-frame agreement. Reproj cannot catch this because a wrong
port is still a real, cleanly-detected port. `target_idx` (from
`task.target_module_name`) is computed but only used in the log string, not to
filter.

This is separate from the contact-time lateral blowout on the CORRECT port
(runs 1 & 3 hit the right port but still aborted on lateral ~6.4-6.8mm) — that
is the residual-RL / contact-skill problem, tracked elsewhere.

## Fix (chosen: BOTH — median consensus + distance gate)

Apply in `RLInsert.py`. User selected doing both together.

1. **Multi-frame consensus** (re-add the reverted `perceive_port_pose_consensus`,
   improved): sample ~5-7 frames (arm static), run detection each, keep
   reproj-passing poses, take the MEDIAN port position, and REJECT outliers
   more than a few mm from the median cluster. Require >=3 frames to agree.
   Kills both the ~40mm wrong-port jump and the ~1mm run-to-run wander.

2. **Hard max-distance gate**: in `_select_sfp_candidate`, reject any candidate
   whose distance-to-tip exceeds a plausible-handoff threshold (handoff is ~9mm;
   gate at e.g. 20mm via a new `RL_INSERT_MAX_HANDOFF_SELECT_M`). Run 2's 40mm
   pick would be rejected -> re-perceive instead of committing.

Suggested env knobs (defaults):
```
RL_INSERT_PERCEPT_SAMPLES=7
RL_INSERT_PERCEPT_MIN_AGREE=3
RL_INSERT_PERCEPT_AGREE_TOL_M=0.004
RL_INSERT_MAX_HANDOFF_SELECT_M=0.020
```

Note: an earlier consensus attempt this session was reverted, but for an
UNRELATED reason (a bad "drop the center camera" change contaminated it). The
consensus logic itself is sound — re-add it WITH all 3 cameras and the
best-subset triangulation left at the v10/v12 default (do not restrict cameras).

## Verification after applying

- Re-run the 3x probe: all three perceived ports should agree within a few mm
  (no 40mm outlier), and any genuine wrong-port frame should be rejected/re-tried.
- Then measure the true correct-port success/abort rate, so contact-skill
  (residual-RL) results are no longer polluted by wrong-port runs.

## Related open items

- Contact-time lateral blowout on the correct port (runs 1&3) -> residual-RL.
- Is there ALSO a small systematic calibration bias (SFP_TIP_IN_TCP_POS or camera
  extrinsics)? The correct-port runs cluster tightly (Y 0.386, ~0.2mm), so the
  dominant error is wrong-port selection, not calibration drift — but re-check
  after consensus is in.
