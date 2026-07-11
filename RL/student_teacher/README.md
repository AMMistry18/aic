# `RL/student_teacher/` — teacher→student distillation (third RL algorithm)

This is the **third** RL training approach in `RL/`, alongside:

| folder | approach |
|---|---|
| `RL/sb3_sac/` | plain SB3 SAC |
| `RL/residual_sac/` | residual SAC on the scripted controller |
| **`RL/student_teacher/`** | **privileged teacher → distilled deployable student** |

The goal is a **deployable** last-inch insertion policy that runs on the *real*
observation (wrist images + F/T + poses) with **no privileged state**, trained by
imitating a privileged teacher that *does* see ground truth.

## Pipeline (3 stages)

```
1. scripted privileged teacher      scripted_teacher.py / scripted_teacher_funnel.py
      │   hand-coded last-inch controller on 21-D privileged state
      ▼
2. residual RL teacher              train_residual_teacher.py  (+ residual_env.py)
      │   SAC learns a small residual on top of the scripted funnel teacher
      │   → produces weights/teacher_level1.zip
      ▼
3. distill to deployable student    distill_dataset.py → train_student.py
      │   roll out the FROZEN teacher, record (deployable_obs, effective_action)
      │   pairs, then behavior-clone a CNN(image)+MLP(vector) student
      │   → produces weights/student.pt
```

`*_a` variants (`student_env_a.py`, `train_student_a.py`) are an alternate student
observation/action formulation.

## Layout

```
student_teacher/
  scripted_teacher.py          stage 1: scripted privileged last-inch controller
  scripted_teacher_funnel.py   stage 1: "funnel" variant used as the residual base
  residual_env.py              stage 2: ResidualTeacherWrapper env builder
  train_residual_teacher.py    stage 2: SAC residual-teacher trainer
  distill_dataset.py           stage 3: teacher-rollout dataset generator (BC targets)
  student_env.py               stage 3: deployable-obs student env wrapper
  train_student.py             stage 3: BC trainer for the deployable student
  student_env_a.py             stage 3: alternate student obs/action formulation
  train_student_a.py           stage 3: BC trainer for the "_a" student
  teacher_contract.py          recovered frozen-teacher 21-D observation adapter
  export_student_a.py          checkpoint -> verified 69x6 TorchScript exporter
  parity/                      Flowstate/MuJoCo handoff capture and field audit
  tacc/                        TACC preflight, training, and held-out evaluation jobs
  REDISTILL_GAZEBO.md          Gazebo-v1 contract, validation, and remote commands
  TEACHER_OBS_INTERFACE.md     privileged/deployable observation contract
  weights/                     committed trained weights (see below)
  dataset/                     distillation shards — GITIGNORED (1.6 GB, regenerable)
```

## Weights (`weights/`, committed)

| file | ~size | what it is |
|---|---|---|
| `teacher_level1.zip` | 3.3 MB | frozen residual RL **teacher** (stage 2 output); the default BC source |
| `residual_funnel_lvl1.zip` | 3.3 MB | residual-teacher checkpoint from the `residual_funnel_lvl1` run |
| `student.pt` | 2.2 MB | distilled **student** — from the `student_distill_smoke` run (short run; retrain for a full policy) |
| `student_a.pt` | 0.6 MB | distilled student, `_a` formulation |

`distill_dataset.py` / `train_student_a.py` default `TEACHER_ZIP_DEFAULT` to
`RL/student_teacher/weights/teacher_level1.zip`.

## Dataset (`dataset/`, gitignored)

`dataset/` holds the stage-3 distillation shards (`shard_w*_*.npz`, ~1.6 GB from
the `student_distill_r2` run). It is **gitignored** (`RL/student_teacher/dataset/`)
because it exceeds GitHub's file limits and is fully regenerable:

```bash
python -m RL.student_teacher.distill_dataset --out RL/output/student_teacher/student_distill
```

## Dependencies

Imports resolve against the current `main` RL layout plus two leaf modules that
were restored for this pipeline:

- `RL/scene_env.py` — from `main` (exports `SceneInsertEnv`, `SceneEnvConfig`)
- `RL/observation.py`, `RL/env.py` — **restored** (leaf modules, no further deps);
  `student_env.py` needs `RL.observation` at import time.

## Running

```bash
# stage 2 — train the residual teacher (writes teacher_level1-style model.zip)
WANDB_MODE=offline python -m RL.student_teacher.train_residual_teacher \
    --steps 150000 --num-envs 4 --out RL/output/student_teacher/residual_funnel_lvl1 --seed 0

# stage 3 — distill the deployable student (auto-generates dataset if missing)
python -m RL.student_teacher.train_student \
    --transitions 150000 --epochs 40 --num-envs 12 \
    --out RL/output/student_teacher/student_distill --seed 0
```

Runtime outputs go under `RL/output/student_teacher/` (scratch; not the committed
`weights/`).

## Flowstate parity mode

`flowstate_v1` keeps the 69-value deploy contract but masks three fields that
are simulator-specific rather than task-specific:

- joint offsets `obs[0:6]`
- absolute TCP pose `obs[12:19]`
- absolute port pose `obs[25:32]`

The evidence and reproduction tools are in `parity/`. The captured Flowstate
handoff and matched MuJoCo handoffs showed that equivalent Cartesian plug poses
can use different valid IK branches. Substituting only `obs[0:6]` removed most
of the old model's action mismatch, so `flowstate_v1` masks those values in the
model instead of synthesizing fake MuJoCo joint angles at deployment.
