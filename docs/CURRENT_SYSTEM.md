# Current AIC system

Updated: 2026-08-08

This document is the source of truth for the solution on `main`. It describes
what the active runtime does, where each component lives, and how to validate
changes. Historical handoffs, abandoned plans, and session notes are available
through Git history and are not kept in the working tree.

## System overview

The solution has three cooperating parts:

1. **Board search** uses the three wrist cameras to find the board and produce a
   safe survey pose.
2. **Pose storage** records the labelled SFP, NIC, SC, and home poses during
   Participant Initialize and retrieves the required pose in later trials.
3. **Insertion** uses camera-derived plug/port geometry and deterministic,
   force-limited control to insert SFP and SC connectors.

Learned last-inch insertion has been removed from the active image. The
`InsertionPolicy` runs the deterministic controller path; the existing
`RL_INSERT_*` environment-variable prefix is retained for deployment
compatibility.

## Active components

| Component | Runtime identity | Main source |
| --- | --- | --- |
| Participant model | `aic_model.insertion.InsertionPolicy` | `aic_model/aic_model/insertion/` |
| Board survey | `ai.tar2.check_board_visibility_skill_v4` | `flowstate/aic_perception/` |
| Guarded board move | `ai.tar2.move_to_board_skill_v1` | `flowstate/aic_perception/` |
| Pose storage | `ai.tar2.pose_kv_store_skill_v1` | `flowstate/aic_kv_store/` |
| Deployment probe | `ai.tar2.test_skill_v1` | `flowstate/aic_perception/test_skill.*` |

`docker/aic_model/Dockerfile` is the complete current model build. It starts
from the pinned upstream Phase 1 toolkit revision, installs the frozen Pixi
environment, and copies the canonical `aic_model/aic_model/` package plus the
required pose weights into both the source checkout and installed Python
package. There is no separate Docker policy overlay.

## Board search and survey

The board-search skill uses only permitted participant data: synchronized
left, center, and right wrist-camera images and calibration; measured joint and
controller state; wrist force; and robot-mounted TF. It does not read board,
port, cable, Gazebo, scoring, or other ground-truth poses.

For staged SFP surveys, Stage 1 captures a fresh three-camera observation. If
the asymmetric purple insignia is not sufficiently visible, it executes one
prevalidated, bounded observation path and captures one more triplet. Motion is
force-gated, checked against joint/travel/reach constraints, split into
minimum-jerk transactions, and reversed to the start of the current transaction
when a safety check fails.

Stage 2 estimates the board pose from the purple landmark using planar PnP,
evaluates target-specific survey poses with live-relative IK and camera
framing, and returns a base-frame Cartesian pose. It does not execute the final
survey move. Flowstate must switch back to the default controller and pass the
returned pose to Move Robot only when `success`, `done`, and `target_valid` are
true.

The complete parameters, process wiring, safety limits, build labels, and
installation command are maintained next to the implementation in
[`flowstate/README.md`](../flowstate/README.md).

## Pose KV store

Participant Initialize writes five labelled poses for each indexed object type
and one home pose:

```text
aic/phase1/sfp/0 .. /4
aic/phase1/nic/0 .. /4
aic/phase1/sc/0  .. /4
aic/phase1/home
```

Later trials read one pose by target name or explicit type/index. A NIC target
name can intentionally supply the index for an SFP read so each trial selects a
distinct cable without a counter. Reads fail when a key is missing instead of
returning a zero pose.

Values survive process runs but not a solution teardown. Run Participant
Initialize again after restarting the solution. Build and install details are
in [`flowstate/README.md`](../flowstate/README.md).

## Insertion

Both connector paths share a 720-second wall-clock action deadline and an 18 N
force-abort limit. The Docker image loads
`aic_model.insertion.InsertionPolicy`, branches on the requested plug type, and uses
deterministic controllers:

- **SFP:** `sfp_controller.py` estimates the plug relative to the requested
  port, aligns in the port frame, applies a force-limited seating trajectory,
  and performs bounded visual/lift recovery when stalled.
- **SC:** `sc_controller.py` primes the SC grasp transform, estimates the port
  and physical mouth geometry, aligns the connector, and retries seating until
  success, a hard safety failure, or the action deadline. The retry ladder can
  unload, re-prime the grasp, refresh the port pose, realign, use bounded visual
  recovery, and run a small force-limited spiral search.

Fresh insertion events are the physical-success signal. The controller accepts
a fresh event without applying a second depth gate because the scoring event is
a proximity trigger. A failed SC insertion can be reported as a successful
action result so the enclosing Flowstate process continues; that process result
must not be interpreted as a physical insertion event.

Important implementation paths:

```text
aic_model/aic_model/insertion/InsertionPolicy.py
aic_model/aic_model/insertion/sfp_controller.py
aic_model/aic_model/insertion/sc_controller.py
aic_model/aic_model/insertion/sc_visual_alignment.py
aic_model/aic_model/insertion/sfp_plug_pose.py
aic_model/aic_model/insertion/sc_plug_pose.py
docker/aic_model/Dockerfile
aic_example_policies/aic_example_policies/ros/weights/
```

The insertion-event interpretation is recorded in
[`INSERTION_EVENT_POLICY.md`](INSERTION_EVENT_POLICY.md). Perception training
and measured results remain in the SC/SFP perception documents in this folder.

## Build

Build the current participant model from the repository root on Linux/AMD64:

```bash
docker build --platform linux/amd64 \
  --file docker/aic_model/Dockerfile \
  --tag my-solution:v1 \
  .
```

Flowstate skill builds require the workspace layout below, a compatible
`sdk-ros` checkout, Docker, and the Intrinsic `inbuild` executable:

```text
ws_aic_phase1/
  src/aic/
  src/sdk-ros/
```

```bash
bash src/aic/flowstate/scripts/build_check_board_visibility_skill.sh
bash src/aic/flowstate/scripts/build_move_to_board_skill.sh
bash src/aic/flowstate/scripts/build_pose_kv_store_skill.sh
bash src/aic/flowstate/scripts/build_test_skill.sh
```

The scripts validate the expected Flowstate asset and image labels before
producing bundles under the workspace `images/` directory.

## Validation

Run the source-level insertion suite from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="aic_model:${PYTHONPATH}" \
.pixi/envs/default/bin/python -m pytest -q \
  aic_model/test/test_sc_controller.py \
  aic_model/test/test_sc_mouth_pose_geometry.py \
  aic_model/test/test_sc_plug_pose.py \
  aic_model/test/test_sc_plug_pose_geometry.py \
  aic_model/test/test_sc_visual_alignment.py \
  aic_model/test/test_sfp_plug_pose.py \
  aic_model/test/test_sfp_plug_pose_trials.py \
  aic_model/test/test_sfp_controller.py \
  testing/sfp_v50_validation/tests
```

Run the Python Flowstate perception suite with the package on `PYTHONPATH`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="flowstate/aic_perception:${PYTHONPATH}" \
.pixi/envs/default/bin/python -m pytest -q \
  flowstate/aic_perception/test
```

The C++ KV-store test is built through the Flowstate skill workspace when
`ament_cmake_gtest` is available. A complete release check also builds the
participant image and all Flowstate bundles in their Linux/AMD64 environment.

## Repository boundaries

The `aic_*` packages, assets, interfaces, controller, engine, Gazebo, and
scoring directories are the underlying AIC toolkit. They are indirect build
and simulation dependencies even when the participant model does not import
them directly.

Current perception-training jobs are grouped by target under [`.tacc/`](../.tacc/).
Developer-facing trial generation, evaluation, and perception-model utilities
are organized under [`tools/`](../tools/); they are not imported by the active
runtime.
The earlier LeRobot student-teacher integration is preserved under
[`legacy/lerobot_student_teacher/`](../legacy/lerobot_student_teacher/) because
it remains useful for teleoperation and dataset recording, but it is not part
of the active runtime.

Obsolete learned-insertion experiments, checkpoints, incremental deployment
patches, and Isaac RL prototypes were removed in Phase 3. Git history remains
the source for reproducing those abandoned experiments if they are ever needed.

Generated runs, logs, checkpoints, notebook checkpoints, local worktrees, and
session artifacts are ignored. Store large experiment evidence outside the
repository or in an artifact service.
