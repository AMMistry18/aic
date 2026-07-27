# Insertion handoff

Updated: 2026-07-26

> **Pending SC work:** The next SC change is the stall-time, camera-guided
> blue-housing/gray-plug margin recovery documented in
> [SC camera-guided seating recovery handoff](SC_VISUAL_RECOVERY_HANDOFF.md).
> It is not implemented at this revision.

## Active behavior

The active insertion baseline is SFP V50 with learned insertion disabled.
`aic_model/aic_model/RLInsert.py` forces `CONTROL_MODE = "script"`, skips the
Torch actor import/load path, and keeps direct script control during a detected
stall. Setting `RL_INSERT_CONTROL_MODE` at runtime does not re-enable the actor.

The final deployment overlay is
`docker/aic_model/Dockerfile.plug_relative_v50`. It patches the exact live V49
runtime, adds plug-relative visual pose estimation and the V50 controller, and
sets both `RL_INSERT_MODEL` and `RL_INSERT_SEAT_MODEL` to empty values. The
release budget is 45 wall-clock seconds, with bounded visual and lift recovery.

Physical success is only the correct scoring insertion event. A returned action
result, partial depth, or absence of an exception is not success.

## Authoritative source

- `aic_model/aic_model/RLInsert.py`: script-only checkout implementation.
- `aic_model/aic_model/v50_controller.py`: plug-relative V50 control and
  bounded recovery.
- `aic_model/aic_model/sfp_plug_pose.py`: multi-camera plug pose estimator.
- `aic_model/aic_model/sfp_plug_pose_geometry.py`: rigid geometry helpers.
- `aic_model/aic_model/sc_plug_pose_geometry.py`: SC pose geometry helpers.
- `docker/aic_model/patch_v49_plug_relative_v50.py`: deterministic live-image
  patch.
- `docker/aic_model/Dockerfile.plug_relative_v50`: final overlay recipe.
- `aic_example_policies/aic_example_policies/ros/weights/best_sfp_plug_pose.pt`:
  active plug-pose weights.
- `testing/sfp_v50_validation/`: frozen-gate generation, observation, and
  evaluation tooling.

The Dockerfile deliberately checks hashes of the expected V49 base files. A
hash failure means the base image changed; update and requalify the overlay
instead of weakening the check.

## Build

Build from the repository root on Linux/AMD64, using the exact qualified V49
image unless a replacement has been deliberately validated:

```bash
docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE=my-solution:student-flowstate-v49-wedge-only-visual \
  -f docker/aic_model/Dockerfile.plug_relative_v50 \
  -t my-solution:student-flowstate-v50 .
```

## Plug-pose data and training

Generate randomized SFP grasp trials, collect simulator-ground-truth labels,
then train the isolated Apple-silicon environment when needed:

```bash
.pixi/envs/default/bin/python generate_sfp_plug_pose_trials.py \
  --trials 450 \
  --seed 20260718 \
  --out "$HOME/aic_perception_data/sfp_plug_pose/sfp_plug_pose_trials.yaml"

./setup_sfp_plug_pose_m5.sh
"$HOME/.venvs/aic-sfp-plug-pose-m5/bin/python" train_sfp_plug_pose.py
```

Evaluate the untouched pose test split:

```bash
"$HOME/.venvs/aic-sfp-plug-pose-m5/bin/python" \
  eval_sfp_plug_pose_model.py \
  --weights aic_example_policies/aic_example_policies/ros/weights/best_sfp_plug_pose.pt \
  --split test \
  --enforce
```

The pose estimator fails closed for missing weights, stale or single-camera
images, weak confidence, and inconsistent rigid geometry. There is no
fixed-bias pose fallback.

## Validation

Run the fast source suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="aic_model:${PYTHONPATH}" \
  .pixi/envs/default/bin/python -m pytest -q \
  aic_model/test/test_sc_plug_pose_geometry.py \
  aic_model/test/test_sc_plug_pose_trials.py \
  aic_model/test/test_sfp_plug_pose.py \
  aic_model/test/test_sfp_plug_pose_trials.py \
  aic_model/test/test_v50_controller.py \
  testing/sfp_v50_validation/tests
```

For release qualification, follow `testing/sfp_v50_validation/README.md`.
The frozen gate requires exactly 300 unique trials, 300 correct events within
45 seconds, and zero wrong-port, off-limit-contact, or force-penalty trials.
Write all generated evidence under `results/`; that directory is ignored.
