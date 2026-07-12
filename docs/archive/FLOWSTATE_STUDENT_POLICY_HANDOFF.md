> **ARCHIVED / SUPERSEDED (2026-07-12).** Pre-rewrite policy handoff (lines 7-8 were already noted as inaccurate). RLInsert is now self-contained + rl mode; see `docs/FLOWSTATE_STATUS.md`.
> Kept for history only — do not follow as current instructions.

# Flowstate student policy handoff

Updated: 2026-07-11

## Current state

The `flowstate_v1` SFP student is installed in the running Flowstate solution.
The service started and loaded `aic_model.RLInsert` successfully. Its exact
captured handoff fixture now produces a moderate inward action instead of the
old saturated lateral action.

The latest deployed build changes nearest-target selection to consider both
SFP keypoint groups (`sfp_port_0` and `sfp_port_1`) on every detected NIC. It
also enables a deterministic perception-guided handoff: retract, align to the
selected socket, then approach to 26 mm before starting the student.

The new model has not yet been run through the Flowstate behavior tree. The CLI
can install and inspect services but cannot execute that UI-owned process, so
there is no Flowstate insertion score yet. MuJoCo held-out results are below.

## Repository artifacts

| Artifact | SHA-256 |
| --- | --- |
| `models/final_insert_sfp_flowstate_v1.pt` | `2da20284ffc713ee4cb4048a0cebee6531509ae3487c24ab217cc510269b9908` |
| `models/final_insert_sfp_flowstate_v1.ts` | `64d4726eb7b6d4b25df2d4a3b4d7d1af5a4bc2522081cd5b4fda451bc1282cf6` |
| `models/final_insert_sfp_flowstate_v1.ts.contract.json` | `6d5105f72a2051c08dfe86ef32da898295c3edf650f4548dec8f8bf597b4d894` |
| `models/final_insert_sfp_gazebo_v1.ts` | `79c511587dfa6289a801a51d9c5283e11dc6c047bb5c8f77dd93581c122edbf0` |
| `models/final_insert_sfp_gazebo_v1.ts.contract.json` | `3cd416feb260b07b98d586711f5b51b051eed308c7acede44e00b0897ea23eb3` |

Important code:

- `aic_model/aic_model/RLInsert.py`: policy loading, 69-value observation,
  perception handoff, Cartesian action loop, and safety exits.
- `aic_model/aic_model/rl_insert_contract.py`: shared 69x6 numerical contract,
  TCP-to-SFP-tip calibration, port frame, and action scaling.
- `aic_example_policies/aic_example_policies/ros/PerceptionInsert.py`: camera
  detection, multiview triangulation, and closest-socket selection.
- `docker/aic_model/Dockerfile.student_flowstate`: exact thin image used for
  the latest Flowstate deployment.
- `RL/student_teacher/TACC_NEXT_AGENT_HANDOFF.md`: TACC paths, training job,
  held-out results, and authentication procedure.
- `RL/student_teacher/tacc/`: preflight, Slurm, and held-out evaluation tools.
- `RL/student_teacher/parity/`: exact Flowstate fixture, matched MuJoCo
  handoffs, field substitution audit, candidate gate, and evaluation reports.
- `docs/FLOWSTATE_MUJOCO_PARITY_20260711.md`: failure diagnosis and permanent
  adapter rationale.

## Training provenance

The selected epoch-25 checkpoint is:

```text
/scratch/11590/satya_a/aic/student_flowstate_v1_seed0_5900b41/student_a_ep025.pt
```

Its fixed-seed MuJoCo held-out evaluation was:

```text
210/300 success (70.00%)
88 timeout
2 bad_collision
```

This did not pass the strict zero-collision gate. Epoch 40 reached 215/300
success but had four bad collisions; epoch 10 reached 187/300 with two. Epoch
25 was selected as the best measured success/safety tradeoff. These results are
MuJoCo results, not Flowstate or Gazebo scores.

## Flowstate deployment snapshot

Snapshot-sensitive values from 2026-07-11:

```text
organization: tar-2@xfa-prod-aic-us
VM/cluster: vmp-f5ed-053nou72
solution: 582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH
service instance: aic_model
asset: ai.intrinsic.aic_model.0.0.1+c84d8e248aa372bfa959e0e0b790f6150d96ffd1900226879d6da3798741d393
```

On 2026-07-11 this asset was re-uploaded from the local
`my-solution:student-flowstate-guided-v5` image. The displayed asset version did
not change because the service manifest identity stayed the same, but the image
contents now include the Flowstate router entrypoint in
`docker/aic_model/Dockerfile.student_flowstate`. That entrypoint sets
`ZENOH_CONFIG_OVERRIDE` from `AIC_MODEL_ROUTER_ADDR` and `AIC_MODEL_PASSWD`
instead of relying on rmw_zenoh peer scouting.

After upload, the `aic_model` service instance was deleted, added back, and the
solution was restarted in sim mode on `vmp-f5ed-053nou72`. A direct cluster add
then reported `instance already exists with id "aic_model"`, which is expected.
`inctl logs --service aic_model` may return `resource not found` until the
service emits logs; use a Flowstate lifecycle/configure attempt as the runtime
verification.

The lifecycle node must remain named `aic_model`. Do not rename it to the asset
or wrapper name.

An obsolete service named `aic_insertion_policy` previously ran another
`aic_model` ROS node and action server on `/insert_cable`. Both servers accepted
the same goal. The old server returned first, causing Flowstate to deactivate
the real student after about 0.7 seconds of motion. The obsolete service was
deleted. Before debugging policy math, verify it has not been recreated:

```bash
inctl logs --org 'tar-2@xfa-prod-aic-us' \
  --solution '582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH' \
  --service aic_insertion_policy --since 2m --tail 20
```

`resource not found` is the desired result. Also treat ROS action-client logs
such as `unknown goal response` or `unknown result response` as evidence of a
duplicate `/insert_cable` action server.

## Current runtime settings

The thin Flowstate image sets:

```text
RL_INSERT_MODEL=/models/final_insert_sfp_flowstate_v1.ts
RL_INSERT_WRENCH_MODE=baseline
AIC_SFP_TARGET_MODE=nearest_tip
RL_INSERT_PREPOSITION=1
RL_INSERT_HANDOFF_GAP_M=0.026
RL_INSERT_HANDOFF_LATERAL_SIGMA_M=0
RL_INSERT_HANDOFF_AXIAL_SIGMA_M=0
RL_INSERT_HANDOFF_ROT_SIGMA_RAD=0
```

The runtime and training wrench contracts now match. Baseline subtraction is
initialized once per insertion and preserves contact-force changes.

The thin Dockerfile was built from the preserved prior AMD64 service image.
The latest local bundle is intentionally not checked into Git:

```text
/private/tmp/aic-flowstate-guided-v5/images/aic_model/aic_model.bundle.tar
```

## Closest-port behavior

Earlier nearest mode still obeyed `task.port_name` indirectly: it sliced the
eight NIC keypoints down to the requested four before comparing distances.
The current implementation builds candidates for both keypoint groups and
then chooses the minimum 3D distance from the calibrated SFP tip.

Expected log evidence in the latest build:

```text
candidate_ports=[0, 1]
SFP target: nearest-tip mode selected candidate ...
SFP selected nearest candidate (sfp_port_N; requested ...)
[rl] randomized handoff target | gap_mm=26.0 lateral_mm=[0.0, 0.0] ...
```

The preposition sequence first retracts to 120 mm, then approaches to 65 mm
and 26 mm. Moving away from the board at the start is expected.

## Last confirmed bad run

Before prepositioning and both-port candidate generation were enabled, the
student started at:

```text
depth=-10.4 mm
lateral=4.0 mm
```

Its first raw action was approximately:

```text
[-0.944, 0.963, -0.445, -0.182, 0.394, -0.042]
```

The lateral error grew to 12.5 mm and triggered the 12 mm contract safety
abort after about three seconds. The parity audit identified `obs[0:6]` joint
offsets as the dominant cause. `flowstate_v1` masks those simulator-specific IK
values inside the model; it does not weaken the safety limit.

Priority checks:

1. Confirm only one `/insert_cable` action server responds.
2. Confirm the latest logs include candidates for both SFP port slots.
3. Confirm prepositioning reaches near-zero lateral/rotation error at 26 mm.
4. Confirm the log reports `RL_INSERT_WRENCH_MODE=baseline` behavior and the
   first action is close to the fixture direction, not the old saturated one.
5. Record success, timeout, safety abort, and insertion depth for the Flowstate
   run; do not report the MuJoCo score as a Flowstate score.

## Flowstate process requirements

For each insertion group, keep this sequence:

1. Tare force/torque sensor.
2. Switch to AIC controller.
3. Invoke insertion policy.
4. Switch to default controller.

Configure and activate `aic_model` before the insertion group, and deactivate
it only after the policy returns. Cable End 2 was disabled during single-policy
debugging. The insertion block used a 180 second time limit.

Bridge failure signature:

```text
MotionUpdate stream writer not initialized.
ICON Server operational status: Disabled!
Failed to start ICON client and session.
```

When that appears, policy outputs are being published but cannot move the
robot. Recover `robot_controller`, then let `switch_to_aic_controller_skill`
restart the bridge before invoking the policy.

## Verification commands

From the repository root:

```bash
python3 -m py_compile \
  aic_model/aic_model/RLInsert.py \
  aic_model/aic_model/rl_insert_contract.py \
  aic_example_policies/aic_example_policies/ros/PerceptionInsert.py

shasum -a 256 \
  models/final_insert_sfp_flowstate_v1.pt \
  models/final_insert_sfp_flowstate_v1.ts \
  models/final_insert_sfp_flowstate_v1.ts.contract.json

docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE=flowstate:aic_model-student-79c51158 \
  -f docker/aic_model/Dockerfile.student_flowstate \
  -t my-solution:student-flowstate-v1-64d4726e .
```

Do not commit Flowstate login state, access tokens, `/private/tmp` bundles, TACC
passwords, MFA codes, WandB keys, or generated 7 GB image archives.
