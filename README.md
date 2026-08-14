# AIC Cable Insertion Solution

This repository contains a participant solution for the Intrinsic AI for
Industry Challenge. The system uses three wrist-mounted cameras and wrist
force/torque feedback to locate a task board and insert two connector types:

- **SFP** transceiver connectors
- **SC** duplex fibre connectors

The active solution is deterministic. Neural networks estimate plug and port
geometry, while force-limited state machines handle alignment, insertion, and
bounded recovery. Learned last-inch reinforcement-learning control is not part
of the deployed runtime.

The repository also retains the upstream AIC simulation toolkit required to
build, test, and package the participant model.

## How the system works

The solution has three cooperating parts:

1. **Board search** finds the task board from the wrist cameras and produces a
   safe survey pose.
2. **Pose storage** records labelled SFP, NIC, SC, and home poses during
   Participant Initialize and retrieves the requested pose in later trials.
3. **Cable insertion** estimates the plug and target port, aligns the connector,
   and seats it using force-limited SFP- or SC-specific control.

```text
Wrist cameras + robot state + wrist force
                    |
                    v
        Board search and pose storage
                    |
                    v
     aic_model.insertion.InsertionPolicy
              /                 \
             v                   v
      SFP controller       SC controller
             \                   /
              +----> Robot motion
```

The policy uses only participant-accessible observations. It does not depend on
simulator ground-truth board, cable, or port poses.

## Active runtime

| Component | Location | Purpose |
| --- | --- | --- |
| Policy entry point | [`aic_model/aic_model/insertion/InsertionPolicy.py`](aic_model/aic_model/insertion/InsertionPolicy.py) | Loads perception and dispatches by connector type |
| SFP controller | [`sfp_controller.py`](aic_model/aic_model/insertion/sfp_controller.py) | SFP alignment, seating, and bounded recovery |
| SC controller | [`sc_controller.py`](aic_model/aic_model/insertion/sc_controller.py) | SC perception, alignment, retry ladder, and spiral recovery |
| Pose estimation | [`aic_model/aic_model/insertion/`](aic_model/aic_model/insertion/) | SFP/SC plug and port geometry |
| Board-search skills | [`flowstate/aic_perception/`](flowstate/aic_perception/) | Board framing and guarded survey motion |
| Pose KV store | [`flowstate/aic_kv_store/`](flowstate/aic_kv_store/) | Labelled pose persistence between trial steps |
| Model image | [`docker/aic_model/Dockerfile`](docker/aic_model/Dockerfile) | Deployable Linux/AMD64 participant image |

The ROS policy name used by the model container is:

```text
aic_model.insertion.InsertionPolicy
```

## Repository layout

```text
aic_model/            Participant ROS model and insertion controllers
flowstate/            Board-search, guarded-motion, and pose-store skills
docker/               Participant and evaluation container definitions
testing/              Frozen validation suites and test data
tools/                Trial, evaluation, and perception-training utilities
.tacc/                 TACC perception-training jobs grouped by target
docs/                  System, deployment, and challenge documentation
legacy/                Archived student-teacher/teleoperation integration
aic_*/                 Upstream AIC toolkit, simulator, engine, and interfaces
```

Generated runs, checkpoints, bundles, videos, and article/site output do not
belong in this repository. Store large experiment evidence in an artifact
service or external storage.

## Requirements

The official deployment target is **Ubuntu 24.04, Linux/AMD64, and ROS 2
Kilted**. Local source tests also run on macOS ARM64 through Pixi, but deployable
images and Flowstate bundles must be built for Linux/AMD64.

Install:

- [Git](https://git-scm.com/)
- [Pixi](https://pixi.prefix.dev/) 0.67.2
- [Docker](https://docs.docker.com/engine/install/)
- NVIDIA Container Toolkit when GPU acceleration is required
- Intrinsic build/install tooling for Flowstate asset deployment

The contest environment requires Pixi 0.67.2:

```bash
pixi self-update --version 0.67.2
```

## Set up the repository

```bash
git clone https://github.com/AMMistry18/aic.git
cd aic
pixi install
```

The first install is large because it includes ROS, simulation, perception, and
model dependencies. Pixi creates the local environment under `.pixi/`.

For the complete upstream simulator setup, including the evaluation container
and Zenoh networking, see [`docs/getting_started.md`](docs/getting_started.md).

## Run tests

Run the participant-model and frozen insertion validation suites from the
repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="aic_model:${PYTHONPATH}" \
.pixi/envs/default/bin/python -m pytest -q \
  aic_model/test \
  testing/sfp_v50_validation/tests
```

Run the Flowstate perception tests separately:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="flowstate/aic_perception:${PYTHONPATH}" \
.pixi/envs/default/bin/python -m pytest -q \
  flowstate/aic_perception/test
```

To write reproducible source-validation evidence for both canonical suites,
including the exact commands, test counts, durations, Python/platform metadata,
and Git revision/dirty state, run:

```bash
.pixi/envs/default/bin/python tools/validation/run_source_validation.py \
  --output-dir results/source_validation
```

This runner invokes the two commands above with their documented
`PYTHONPATH` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` settings. Its JSON and
Markdown reports record source-test validation only; they are not evidence of
end-to-end simulated or physical cable-insertion success.

The C++ pose-store tests run as part of the Flowstate skill build when
`ament_cmake_gtest` is available.

## Build the participant model

Build the current insertion-policy image from the repository root:

```bash
docker build --platform linux/amd64 \
  --file docker/aic_model/Dockerfile \
  --tag my-solution:v1 \
  .
```

The Dockerfile copies the canonical `aic_model/aic_model/` package and required
perception weights directly into the image. There is no separate policy overlay.

Detailed image verification, bundle, and installation commands are in
[`docs/INSERTION_POLICY_DOCKER_GROUND_TRUTH.md`](docs/INSERTION_POLICY_DOCKER_GROUND_TRUTH.md).

## Build the Flowstate skills

Flowstate builds expect this workspace layout:

```text
ws_aic_phase1/
  src/aic/
  src/sdk-ros/
```

From the workspace root:

```bash
bash src/aic/flowstate/scripts/build_check_board_visibility_skill.sh
bash src/aic/flowstate/scripts/build_move_to_board_skill.sh
bash src/aic/flowstate/scripts/build_pose_kv_store_skill.sh
bash src/aic/flowstate/scripts/build_test_skill.sh
```

See [`flowstate/README.md`](flowstate/README.md) for process wiring, required
asset labels, safety limits, and installation instructions.

## Configuration

Controller configuration is read from environment variables. The historical
`RL_INSERT_*` prefix is retained for deployment compatibility even though the
active insertion path is deterministic. Defaults and safety limits live beside
the controller implementations; deployment-specific overrides are set in the
container or Flowstate solution configuration.

Do not weaken motion, force, deadline, or perception gates without rerunning the
relevant validation suite.

## Documentation

- [`docs/CURRENT_SYSTEM.md`](docs/CURRENT_SYSTEM.md) — source of truth for the
  active architecture and validation workflow
- [`aic_model/README.md`](aic_model/README.md) — participant model layout
- [`flowstate/README.md`](flowstate/README.md) — Flowstate skills and deployment
- [`docs/INSERTION_EVENT_POLICY.md`](docs/INSERTION_EVENT_POLICY.md) — physical
  insertion success-event interpretation
- [`docs/SC_PLUG_POSE_RESULTS.md`](docs/SC_PLUG_POSE_RESULTS.md) — SC perception
  measurements and evidence
- [`tools/README.md`](tools/README.md) — developer utilities
- [`docs/challenge_rules.md`](docs/challenge_rules.md) — official behavior and
  interface requirements

## Current status

`main` contains the canonical SFP and SC insertion implementation, board-search
skills, and pose KV store. Source-level controller and geometry tests cover the
active logic. A release should additionally build the Linux/AMD64 model image,
build all Flowstate bundles, and run end-to-end trials in the evaluation
environment.

## License

This project and the retained AIC toolkit are licensed under the Apache License
2.0. See [`LICENSE`](LICENSE) and individual package metadata for details.
