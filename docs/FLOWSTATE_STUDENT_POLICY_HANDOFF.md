# Flowstate student policy handoff

Updated: 2026-07-10

## Current state

The SFP student policy loads and runs in Flowstate. It perceives SFP sockets,
publishes Cartesian commands, and has produced visible robot motion. It is not
yet a reliable insertion policy.

The latest deployed build changes nearest-target selection to consider both
SFP keypoint groups (`sfp_port_0` and `sfp_port_1`) on every detected NIC. It
also enables a deterministic perception-guided handoff: retract, align to the
selected socket, then approach to 26 mm before starting the student.

This latest target-selection/preposition build was deployed but had not been
run and scored when this handoff was written.

## Repository artifacts

| Artifact | SHA-256 |
| --- | --- |
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

## Training provenance

The artifact contract points to:

```text
/scratch/11590/satya_a/aic/student_gazebo_v1_seed0_5465a5c_baseline/student_a.pt
```

The MuJoCo held-out evaluation was 252/300 successes (0.84):

```text
seed 10001: 0.86
seed 20002: 0.90
seed 30003: 0.76
```

`gazebo_v1` is the observation contract name. This result was produced in
MuJoCo, not in Gazebo or Flowstate. The 0.95 export gate failed, so the original
Slurm script stopped before its normal export step. Preserve the checked-in
TorchScript hash above; the exact later manual export command was not captured.

## Flowstate deployment snapshot

Snapshot-sensitive values from 2026-07-10:

```text
organization: tar-2@xfa-prod-aic-us
VM/cluster: vmp-f5ed-iea4i2cn
solution: 582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH
service instance: aic_model
asset: ai.intrinsic.aic_model.0.0.1+62df6b7111bdcddabc5848af6b2156ef6d70c2c70c5ab0ba9c087f87c2ea7e62
```

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
RL_INSERT_WRENCH_MODE=zero
AIC_SFP_TARGET_MODE=nearest_tip
RL_INSERT_PREPOSITION=1
RL_INSERT_HANDOFF_GAP_M=0.026
RL_INSERT_HANDOFF_LATERAL_SIGMA_M=0
RL_INSERT_HANDOFF_AXIAL_SIGMA_M=0
RL_INSERT_HANDOFF_ROT_SIGMA_RAD=0
```

There is an unresolved contract mismatch: the artifact contract records
`wrench_mode=baseline`, while the deployed image forces `zero`. Baseline and
zero are identical at the first sample but differ after contact. This is a
high-priority experiment, not a settled choice.

The thin Dockerfile depends on the machine-local AMD64 image
`my-solution:v8`. A clean clone cannot reproduce the Flowstate image until that
base is exported/shared or the full `docker/aic_model/Dockerfile` build is
fixed. The full build previously stopped because `pixi.lock` was out of sync.
The 7.1 GB Flowstate bundle is intentionally not checked into Git.

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
abort after about three seconds. Do not raise that safety limit merely to keep
the run alive. Determine whether deterministic prepositioning fixes the
distribution shift. If not, compare Flowstate observations and actions with a
MuJoCo rollout at the same `delta_port` and orientation.

Priority checks:

1. Confirm only one `/insert_cable` action server responds.
2. Confirm the latest logs include candidates for both SFP port slots.
3. Confirm prepositioning reaches near-zero lateral/rotation error at 26 mm.
4. Compare `obs[32:38]`, `obs[38:50]`, and action direction against MuJoCo.
5. Test `RL_INSERT_WRENCH_MODE=baseline` with force-frame verification.
6. If the student still drives laterally outward, retrain or add a justified
   residual safety projection; do not hide it by weakening collision guards.

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
  models/final_insert_sfp_gazebo_v1.ts \
  models/final_insert_sfp_gazebo_v1.ts.contract.json

docker build --platform linux/amd64 \
  -f docker/aic_model/Dockerfile.student_flowstate \
  -t my-solution:student-gazebo-v1-debug .
```

Do not commit Flowstate login state, access tokens, `/private/tmp` bundles, TACC
passwords, MFA codes, WandB keys, or generated 7 GB image archives.
