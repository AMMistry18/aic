# Board-search handoff

Updated: 2026-07-24

## Pinned implementation

The current implementation is the **insignia-driven deterministic survey**. It
supersedes the outline-PnP geometric Stage 2 pinned at `525eb40`. Record the
exact commit here when this change merges; until then the pin is the working
tree on `main`. Legacy adaptive search is now Stage 1 only -- the fallback that
runs when the insignia is not in view at all. All deployed `survey_target`
values (SFP, NIC, SC) route to the geometric sector survey.

Note on branches: the sector-survey work was authored on
`agent/sync-model-overlay-and-skill-bundling` (`252e8ac`, `f33ce83`) while the
fixed build script and the current teammate work landed on `main`. The working
tree carries the union; reconcile the branches before pinning a commit here.

Authoritative modules:
- `flowstate/aic_perception/aic_perception/board_visibility.py`
  (`detect_insignia_polygon`)
- `flowstate/aic_perception/aic_perception/board_stage2.py`
  (`estimate_board_pose_from_insignia`, `board_coverage_corners`,
  `module_coverage_corners`, `sfp_sector_corners`, `sc_sector_corners`,
  `nic_sector_corners`, `search_survey_pose`, `verify_survey_view`)
- `flowstate/aic_perception/check_board_visibility_skill.py`
  (`_stage2_landmarks`, `_execute_inner`, `_uses_geometric_survey`,
  `_sector_for_target`, `_run_sfp_geometric_stage2`)

## Behavior contract

Stage 1 is a short, low-constraint exposure search. It has **no wall-clock
deadline**: the planner terminates on its own stall condition and every move is
force- and per-move-timeout-guarded. For staged SFP modules, as soon as the
insignia is cleanly visible in a calibrated camera (or on the planner's own
`DONE`/terminal), Stage 1 hands its freshest triplet to Stage 2. This
guarantees Stage 2 always runs; it is never pre-empted by a timeout.

Stage 2 is deterministic:

- it consumes exact CameraInfo intrinsics and image-timestamped TCP/camera TF;
- it estimates the full 6-DoF board pose by planar PnP of the **asymmetric
  purple insignia** (bracket corners against `INSIGNIA_RECT_CORNERS`, the mask
  centroid resolving the rectangle ambiguity). This is clip-proof: it does not
  require a fully visible plate outline or a "full" Stage-1 report;
- it computes one board-relative TCP survey pose by inverting the production
  three-camera URDF rig, searching standoff, both board-plane offsets, look
  direction, and roll, filtered by the execution workspace (reach 0.85 m, the
  UR5e envelope; height 0.02 m) and sampled Cartesian path clearance. Height
  and lateral placement both fall out of the estimated `base_T_board`, so the
  pose tracks a board that moves or tilts;
- coverage is **per sector**, selected by `survey_target`: SFP modules (0/1),
  NIC cards (2), SC ports (3). Each sector is a board-frame box covering that
  component group's full rail travel; the whole sector must be framed in all
  three cameras, because IVM pose estimation needs every camera to see it. The
  only per-camera acceptance is target-in-frame plus positive gripper
  clearance;
- selection is tuned for **separability of adjacent parts**. Standoff dominates
  the objective (closest feasible pose wins, ~0.65 m rather than ~0.85 m), ties
  break towards the most overhead view, and two hard rejects back this up: the
  reference optical axis stays within 20 degrees of the board normal, and every
  camera must hold at least 40 px of clearance. A raking or distant view
  foreshortens the along-rail spacing of tall parts, which is what stopped IVM
  telling adjacent NIC cards apart;
- it allows at most 45 degrees of orientation change and performs any
  meaningful wrist reorientation only after retreating beyond a conservative
  0.40 m rig sweep radius; and
- the skill is **perception-only**: it publishes the result as a native
  `intrinsic_proto.Pose` on `result.survey_pose` (with `result.target_frame =
  base_link`) for a downstream Move Robot Cartesian target, and does not move
  to the survey pose itself. There is **no** aggregate Stage-2 time budget and
  **no** two-triplet consistency gauntlet.

Any calibration, geometry, reach, path, or confirmation failure returns
`success=true, done=false` so the Flowstate process can decide whether to retry.
Cancellation still uses the process cancellation path; every motion is
force-guarded.

## Latest deployment

The insignia-driven implementation was built and installed **in place** as the
existing `check_board_visibility_skill_v4` asset (not a new skill) into Flowstate
solution `9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH` ("Work on this",
org `tar-2@xfa-prod-aic-us`) on 2026-07-23 as:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+aac828ea836a05056fc5ab0b1fe10a6fcc112ec7c18ce7bb0a317b81c9dc99f6
```

Provenance: built from the edited working tree via
`flowstate/scripts/build_check_board_visibility_skill.sh` (full colcon rebuild,
gRPC smoke test passed). Bundle SHA-256
`ca45a13cac0ce099f664f1dbcddd9a39754fd93a4284ff5a694dab0dffaade44`. It replaces
the prior outline-PnP build `...+03b1f018...`; the outline-clip failure mode is
eliminated because pose is driven by the insignia rather than the plate outline.
The change is not yet committed to git -- record the commit under Pinned
implementation when it lands. Build note: strip CRLF from
`flowstate/scripts/*.sh` before running the build on a Windows-checkout workspace.

## Authoritative source

- `aic_model/aic_model/board_search.py`
- `aic_model/test/test_board_search.py`
- `flowstate/aic_perception/`
- `flowstate/resources/`
- `flowstate/scripts/`
- `deploy/flowstate/aic_model_v38.manifest.textproto`
- `scripts/flowstate/inctl.sh`

To verify that the implementation files have not drifted, diff against the
commit recorded under Pinned implementation once this change has merged:

```bash
git diff --exit-code <pinned-commit> -- flowstate/aic_perception
```

## Build and install

Use a Linux/AMD64 workspace with `src/aic` and `src/sdk-ros`:

```bash
cd ~/ws_aic_phase1
bash src/aic/flowstate/scripts/build_check_board_visibility_skill.sh
```

Install the generated bundle only after re-reading the active cluster:

```bash
inctl asset install \
  --org tar-2@xfa-prod-aic-us \
  --cluster "$CLUSTER" \
  images/check_board_visibility_skill/check_board_visibility_skill.bundle.tar
```

Recommended serial wiring is:

```text
Move Robot
-> Switch To AIC Controller
-> Check Board Visibility
-> Switch To Default Controller
-> require result.success && result.done
-> downstream IVM
```

Do not run another motion session in parallel with this skill.

## Validation

Run the model helper test and the complete Flowstate package suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="aic_model:${PYTHONPATH}" \
  .pixi/envs/default/bin/python -m pytest -q \
  aic_model/test/test_board_search.py \
  flowstate/aic_perception/test
```

The insignia-driven implementation passes 224 Flowstate perception tests
(216 prior + 8 new insignia-PnP / two-tier-coverage cases). Any intentional
board-search change must update this handoff and its pinned implementation.
