# Board-search handoff

Updated: 2026-07-22

## Pinned baseline

The board-search implementation on `main` is pinned byte-for-byte to:

```text
b269872eb6f0a4a49edc6334c6985e4b00238a5b
Record three-camera v4 deployment
```

The pin covers the model helper and test, Flowstate package, calibrated masks,
skill resources and build scripts, deployment manifest, and `inctl` helper.
Later board-search, simplified-search, and gripper-gate commits are intentionally
excluded.

## Behavior contract

The V4 terminal contract uses synchronized evidence from all three wrist
cameras:

- the center camera owns board identity, J1/J6 alignment, top-down geometry,
  26–36% image area, and the strict two-degree orientation gate;
- left and right cameras provide mandatory supporting context and gripper
  separation, but cannot complete the search independently;
- all three views must have zero protected-envelope mask overlap;
- two consecutive fresh synchronized snapshots must satisfy the terminal
  contract; and
- a failing side view drives a small translation in that camera's image axes,
  after which center geometry is rechecked.

Motion remains bounded by deadline, force, displacement, and cumulative-travel
limits. Expected sensor or search failure returns `success=false`; cancellation
uses the process cancellation path. The Flowstate process must always switch
back to the default controller after the skill, on both success and failure.

## Authoritative source

- `aic_model/aic_model/board_search.py`
- `aic_model/test/test_board_search.py`
- `flowstate/aic_perception/`
- `flowstate/resources/`
- `flowstate/scripts/`
- `deploy/flowstate/aic_model_v38.manifest.textproto`
- `scripts/flowstate/inctl.sh`

To verify that the pinned files have not drifted:

```bash
git diff --exit-code b269872 -- \
  aic_model/aic_model/board_search.py \
  aic_model/test/test_board_search.py \
  flowstate \
  deploy/flowstate \
  scripts/flowstate/inctl.sh
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

Commit `b269872` recorded 144 passing Flowstate tests. Any intentional board
change must update this handoff and the pinned commit statement in the same
change.
