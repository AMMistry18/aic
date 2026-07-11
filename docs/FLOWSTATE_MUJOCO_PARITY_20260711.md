# Flowstate/MuJoCo policy parity

Updated: 2026-07-11

## Failure cause

The Flowstate perception and preposition stages were working. At the captured
handoff, the SFP tip was centered laterally and rotationally and was 21.9 mm
outside the socket. The old `gazebo_v1` student nevertheless produced a nearly
saturated lateral action:

```text
old Flowstate action: [-0.9746, 0.9008, -0.3420, -0.2510, 0.3284, -0.0821]
```

`RL/student_teacher/parity/analyze_observation_parity.py` compared that exact
69-value observation with matched MuJoCo handoffs. Replacing only the first six
joint-offset values changed the old model action to:

```text
[0.0587, -0.0246, 0.1485, -0.0308, -0.0109, -0.0465]
```

The action-distance improvement was `0.9836`; every other single field group
improved it by at most `0.0040`. The two systems were at equivalent Cartesian
plug poses but on different valid robot IK branches. Therefore the joint
offsets were not a portable task feature, and no fixed numeric translation was
justified.

## Permanent adapter

The new `flowstate_v1` feature mode masks the following inputs inside the
student network:

```text
obs[0:6]    robot joint offsets
obs[12:19]  absolute TCP pose
obs[25:32]  absolute port pose
```

All relative plug/port geometry, motion, controller hints, wrench values, last
action, and tip axes remain active. Masking inside the model makes the same
TorchScript artifact invariant to Flowstate's IK branch and avoids fabricating
MuJoCo joint values in deployment code.

The deployment wrench mode is `baseline`, matching the training contract. The
old image incorrectly forced `zero`, which agreed only at the first sample and
discarded contact-force changes afterward.

## Training and evaluation

TACC job `3296109` retrained on the preserved 310,000-transition dataset:

```text
source: /work2/11590/satya_a/stampede3/aic-flowstate-v1-5900b41
run:    /scratch/11590/satya_a/aic/student_flowstate_v1_seed0_5900b41
mode:   flowstate_v1, baseline wrench, deploy action convention
epochs: 40
```

The epoch-40 periodic evaluation was 44/50 success with no bad collisions, but
the fixed three-seed held-out run was weaker:

```text
epoch 40: 215/300 success, 81 timeout, 4 bad_collision (71.67%)
epoch 25: 210/300 success, 88 timeout, 2 bad_collision (70.00%)
epoch 10: 187/300 success, 111 timeout, 2 bad_collision (62.33%)
```

All candidates had positive median inward axial action and median lateral
action opposing the lateral error. No evaluated checkpoint passed the strict
zero-collision gate. Epoch 25 was deployed as the measured success/safety
tradeoff. The JSON reports are committed under `RL/student_teacher/parity/`.

## Reproduction

Capture matched MuJoCo handoffs on TACC:

```bash
pixi run --as-is python -m RL.student_teacher.parity.capture_mujoco_handoff \
  --checkpoint /path/to/student_a.pt \
  --torchscript /path/to/student.ts \
  --output /path/to/mujoco_handoffs.json
```

Run the field substitution audit:

```bash
pixi run --as-is python -m RL.student_teacher.parity.analyze_observation_parity \
  --torchscript models/final_insert_sfp_gazebo_v1.ts \
  --flowstate RL/student_teacher/parity/flowstate_handoff_20260711.json \
  --mujoco RL/student_teacher/parity/mujoco_handoffs.json \
  --output RL/student_teacher/parity/parity_analysis.json
```

The TACC scripts used for refitting and candidate selection are:

```text
RL/student_teacher/tacc/refit_flowstate_v1.slurm
RL/student_teacher/tacc/evaluate_flowstate_checkpoints.slurm
```
