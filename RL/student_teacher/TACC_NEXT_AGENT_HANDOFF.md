# TACC continuation handoff

## 2026-07-11 Flowstate parity continuation

The Flowstate failure was reproduced from an exact 69-value handoff fixture and
matched MuJoCo states. The dominant mismatch is `obs[0:6]`: equivalent
Cartesian handoffs use different valid IK branches in Flowstate and MuJoCo.
See `docs/FLOWSTATE_MUJOCO_PARITY_20260711.md` and
`RL/student_teacher/parity/` for the evidence and reproduction scripts.

New TACC state:

| Purpose | Path/job |
| --- | --- |
| Source snapshot | `/work2/11590/satya_a/stampede3/aic-flowstate-v1-5900b41` |
| Run output | `/scratch/11590/satya_a/aic/student_flowstate_v1_seed0_5900b41` |
| Parity fixtures | `/scratch/11590/satya_a/aic/flowstate_parity_20260711` |
| Training and epoch-40 evaluation | Slurm `3296109` |
| Epoch-25 held-out evaluation | Slurm `3296154` |
| Epoch-10 held-out evaluation | Slurm `3296172` |

Job `3296109` trained `feature_mode=flowstate_v1` for 40 epochs on the
preserved 310,000-transition baseline-wrench dataset. It intentionally did not
regenerate or modify the dataset or frozen teacher. The strict held-out reports
so far are:

```text
epoch 40: 215/300 success, 81 timeout, 4 bad_collision
epoch 25: 210/300 success, 88 timeout, 2 bad_collision
epoch 10: 187/300 success, 111 timeout, 2 bad_collision
```

All failed the zero-collision gate. Epoch 25 is the deployed checkpoint because
it halves epoch 40's collision count for only a 1.67-point success reduction.
Do not describe the epoch-40 periodic 44/50 result as the final score; the fixed
300-episode evaluation is the selection evidence. Runtime deployment for any
`flowstate_v1` artifact must use `RL_INSERT_WRENCH_MODE=baseline`.

Flowstate deployment snapshot:

```text
solution: 582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH
cluster: vmp-f5ed-08hc5dz6
service: aic_model
asset: ai.intrinsic.aic_model.0.0.1+732d52e2a62e9aaffe07abc65e256a7ec03ddd82154cd1b574eb4a176bf190c2
```

## Authentication

- TACC user: `satya_a`
- Host: `stampede3.tacc.utexas.edu`
- Login: `ssh satya_a@stampede3.tacc.utexas.edu`
- TACC requires the account password and then the six-digit TACC MFA token.
- Do **not** store or request credentials in files, shell history, or source
  control.

This run used an SSH ControlMaster socket at
`/tmp/codex-tacc-%r@%h:%p`. Assume a future session must authenticate again.
After login, confirm access with:

```bash
printf 'WORK=%s\nSCRATCH=%s\n' "$WORK" "$SCRATCH"
squeue -u satya_a
```

Expected filesystem roots:

```text
WORK=/work2/11590/satya_a/stampede3
SCRATCH=/scratch/11590/satya_a
```

## TACC projects and run state

| Purpose | Path |
| --- | --- |
| Original source snapshot | `/work2/11590/satya_a/stampede3/aic-gazebo-v1-9ae671e` |
| Baseline-wrench source snapshot | `/work2/11590/satya_a/stampede3/aic-gazebo-v1-5465a5c` |
| Baseline run output | `/scratch/11590/satya_a/aic/student_gazebo_v1_seed0_5465a5c_baseline` |
| Slurm logs | `/scratch/11590/satya_a/aic/slurm` |

The baseline source snapshot is source-only and reuses the already-existing
Pixi environment via its `.pixi` symlink.  Do not delete or modify the frozen
teacher:

```text
RL/student_teacher/weights/teacher_level1.zip
sha256=fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4
```

## Completed experiment

Slurm job `3292858` completed on 2026-07-10 using Pixi directly on TACC (no
Distrobox, Docker, or Apptainer):

```text
--transitions 150000 --epochs 40 --num-envs 12
--action-convention deploy
--wrench-mode baseline
--feature-mode gazebo_v1
--dagger-iters 4 --dagger-transitions 40000 --dagger-epochs 12
--eval-episodes 100 --seed 0
```

Standalone and in-job preflight passed.  The student runs MuJoCo headlessly
with `MUJOCO_GL=egl`; `gazebo_v1` is an observation contract, not a Gazebo
simulation run.

Held-out result in `held_out_evaluation.json`:

```text
overall success: 0.84 (252 / 300)
seed 10001: 0.86
seed 20002: 0.90
seed 30003: 0.76
```

The job correctly stopped before TorchScript export because the script required
`--min-success 0.95`.  It produced `student_a.pt`, checkpoints, dataset shards,
metrics, preflight reports, and the held-out report.  No TorchScript artifact
exists for this run.

The old zero-wrench scratch run and its offline WandB record were removed at
the user's request.  Do not remove the baseline run above, the source
snapshots, the Pixi environment, teacher, assets, or datasets without explicit
authorization.

## WandB

The preserved offline WandB record is:

```text
/work2/11590/satya_a/stampede3/aic-gazebo-v1-5465a5c/wandb/offline-run-20260710_030828-pdz5fbtb
```

`wandb sync` was attempted but TACC has no configured API key.  Once the user
has logged in, sync it with:

```bash
cd /work2/11590/satya_a/stampede3/aic-gazebo-v1-5465a5c
export PATH="$HOME/.pixi/bin:$PATH"
pixi run --as-is wandb sync wandb/offline-run-20260710_030828-pdz5fbtb
```

## Important deployment distinction

Pixi on TACC is the correct environment for MuJoCo distillation.  Actual AIC
Engine + Gazebo evaluation is performed separately in the local `aic_eval`
Distrobox workflow.  The existing provenance defers Gazebo deployment until
deploy-side per-episode wrench-baseline subtraction and force/torque frame
parity are certified.  Waiving the 0.95 MuJoCo statistical gate does not waive
that separate contract-parity requirement.

Read `RL/student_teacher/TACC_HANDOFF.md` before any additional training or
deployment.  Do not train or modify the teacher before the requested preflight
checks pass.
