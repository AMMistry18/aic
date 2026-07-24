# Board-search handoff

Updated: 2026-07-23

## Pinned implementation

The current implementation is the **insignia-driven deterministic survey**. It
supersedes the outline-PnP geometric Stage 2 pinned at `525eb40`. Record the
exact commit here when this change merges; until then the pin is the working
tree on `main`. NIC and SC preserve the legacy Stage-1 completion behavior.

Authoritative modules:
- `flowstate/aic_perception/aic_perception/board_visibility.py`
  (`detect_insignia_polygon`)
- `flowstate/aic_perception/aic_perception/board_stage2.py`
  (`estimate_board_pose_from_insignia`, `board_coverage_corners`,
  `module_coverage_corners`, two-tier `search_survey_pose`, `verify_survey_view`)
- `flowstate/aic_perception/check_board_visibility_skill.py`
  (`_stage2_landmarks`, `_execute_inner`, `_run_sfp_geometric_stage2`)

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
  three-camera URDF rig, searching standoff, both board-plane offsets, oblique
  look direction, and roll, filtered by the execution workspace (reach 1.20 m,
  height 0.02 m) and sampled Cartesian path clearance;
- coverage is **two-tier**: it prefers a pose that frames the whole board face
  in all three cameras, and otherwise falls back to a pose that frames the
  SFP/SC module region (both Y-side LC/SFP/SC rails over full travel). The only
  per-camera acceptance is target-in-frame plus positive gripper clearance;
- it allows at most 45 degrees of orientation change and performs any
  meaningful wrist reorientation only after retreating beyond a conservative
  0.40 m rig sweep radius; and
- after the move, one fresh triplet (<=50 ms skew) must show the chosen
  coverage target framed in all three cameras. A single bounded corrective
  re-solve/move absorbs any residual before `done=true`. There is **no**
  aggregate Stage-2 time budget and **no** two-triplet consistency gauntlet.

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
