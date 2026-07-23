# Board-search handoff

Updated: 2026-07-23

## Pinned implementation

The geometric staged-SFP implementation is pinned to:

```text
525eb40
Do not gate SFP Stage 2 on Stage 1 board fullness
```

This is the SFP-only follow-up to `b269872`. NIC and SC preserve the legacy
Stage-1 completion behavior.

## Behavior contract

Stage 1 is the original rough board-acquisition planner. For staged SFP
modules, both its normal `DONE` and terminal/no-progress results hand their
final synchronized triplet directly to Stage 2. Stage-1 board fullness is not
a Stage-2 handoff requirement. Stage 2 then:

- consumes exact CameraInfo intrinsics and image-timestamped TCP/camera TF;
- derives its geometric seed from the observed board outline and asymmetric
  purple material landmark, without requiring the Stage-1 report to call the
  board full;
- searches board-relative standoff, two board-plane offsets, oblique look
  direction, and roll using the production three-camera URDF geometry;
- requires the complete conservative staged-SFP envelope and all six legal
  module-seat detail probes inside every camera;
- requires zero overlap and at least 32 pixels of clearance from each
  conservative gripper mask;
- allows at most 45 degrees of orientation change, and performs any meaningful
  wrist reorientation only after retreating beyond a conservative 0.40 m rig
  sweep radius; and
- requires two fresh triplets with at most 50 ms timestamp skew, complete
  per-camera PnP, pairwise/plan pose consistency, and all-camera projection
  verification before `done=true`.

Expected calibration, geometry, reach, timeout, or verification failure returns
`success=true, done=false` so the Flowstate process can decide whether to retry.
Cancellation still uses the process cancellation path. The complete invocation
is capped at 60 seconds and every motion remains force-guarded.

## Latest deployment

Installed into Flowstate solution
`9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH` as:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+03b1f0186844246ba00990c94fd7cf44067e55ce051bf47dbf1a7722e1d1aa02
```

The immediately preceding run proved that the Stage-2 call occurs, but it
still returns before motion if its own geometric seed rejects a clipped board
outline or logo/gripper overlap. That remaining Stage-2 seed behavior is the
next issue to address; it is distinct from the removed Stage-1 fullness gate.

## Authoritative source

- `aic_model/aic_model/board_search.py`
- `aic_model/test/test_board_search.py`
- `flowstate/aic_perception/`
- `flowstate/resources/`
- `flowstate/scripts/`
- `deploy/flowstate/aic_model_v38.manifest.textproto`
- `scripts/flowstate/inctl.sh`

To verify that the implementation files have not drifted:

```bash
git diff --exit-code 525eb40 -- flowstate/aic_perception
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

Commit `525eb40` passes 216 Flowstate perception tests. Any intentional
board-search change must update this handoff and its pinned implementation.
