# TACC handoff: Gazebo-v1 SFP student redistillation

## Objective

Train and export a new SFP last-inch student on TACC using the frozen privileged
teacher, then return the artifacts for guarded AIC Engine/Gazebo validation.
Do **not** treat the legacy `student_a.pt` or
`models/final_insert_sfp_contractA_v1.ts` as deployable.

The desired outcome is a versioned 69-input/6-output TorchScript student that:

- consumes the same numerical observation contract in MuJoCo and ROS;
- uses perception-derived port pose at deployment, never object GT TF;
- moves inward under positive axial action without lateral drift;
- reaches at least 95% held-out closed-loop success before Flowstate packaging.

## Checkout and authoritative assets

Start from repository commit `9ae671e` (`main`, also on `origin/main` at handoff
time). Verify these files exist before allocating training compute:

```bash
git rev-parse HEAD
git status --short
sha256sum \
  RL/student_teacher/weights/teacher_level1.zip \
  RL/student_teacher/weights/student_a.pt \
  models/final_insert_sfp_contractA_v1.ts
```

Expected hashes:

```text
fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4  teacher_level1.zip
6dbf7fe533c16f40d1dc12023fe5347c58d6387095b1be873901860296502122  student_a.pt
1a354730b689ecb0f4ba52873918ea3be1291230c6dc6858a4ba150c6bf571a1  final_insert_sfp_contractA_v1.ts
```

`teacher_level1.zip` is the crown jewel. It was revalidated through the repaired
adapter at 10/10 level-1 MuJoCo success. Do not retrain or modify the teacher.

At handoff, the local worktree also contains unrelated Isaac/USD generated-file
changes. Do not include or clean those as part of this task. The handoff file and
Dockerfile shared-contract copy may be newer than `9ae671e`; inspect the incoming
patch/worktree before training.

## What was fixed

- `teacher_contract.py` reconstructs the frozen teacher's exact 21-D input after
  `SceneEnvConfig.privileged_obs` and `_privileged_obs()` were removed.
- `rl_insert_contract.py` is the single NumPy source of truth for:
  quaternion sign, port basis `[lat_x, lat_y, inward]`, calibrated TCP-to-tip
  transform, 69-D observation layout, and deploy action scaling.
- `student_env_a.py` and `RLInsert.py` both use that shared contract.
- Student wrench fields default to zero because Gazebo/MuJoCo FT sign and frame
  equivalence are not yet certified.
- Student tip observation is reconstructed from measurable TCP, with configurable
  per-episode hidden grasp/calibration noise. Teacher labels continue to use GT.
- `gazebo_v1` masks absolute TCP and port world-pose fields inside the model so
  the network cannot memorize one MuJoCo board placement.
- DAgger shards use unique round prefixes and are actually aggregated.
- The exporter embeds the feature mask in TorchScript and checks numerical parity.
- Bundled `best.pt` SFP perception is active in `RLInsert`; poses over 25 px
  reprojection error are rejected.
- The submission Dockerfile must copy both `RLInsert.py` and
  `rl_insert_contract.py` after its Pixi install layer.

Read these before changing anything:

- `RL/student_teacher/REDISTILL_GAZEBO.md`
- `RL/student_teacher/student_env_a.py`
- `RL/student_teacher/train_student_a.py`
- `RL/student_teacher/export_student_a.py`
- `aic_model/aic_model/rl_insert_contract.py`
- `aic_model/aic_model/RLInsert.py`

## Established no-training results

- Pure numerical contract tests: 3/3 passed.
- Teacher adapter: 10/10 level-1 success; final smoke 3/3.
- TCP-derived SFP tip versus MuJoCo GT: 0.031 mm error.
- `+1` deploy axial action: +3.5 mm inward, effectively zero lateral component.
- TorchScript export parity: max absolute error `0.0`.
- Legacy `student_a.pt` under the corrected producer contract: 0/10
  (9 timeouts, 1 bad collision).
- Legacy TorchScript in real Gazebo: no completed insertions. Axial sign flipping
  improved depth but lateral error reached the 12 mm safety abort, even with
  learned rotations disabled.
- First-scene SFP perception was good (about 1.1 mm worst-axis port error), but a
  later requested-slot match produced 157 px reprojection despite high detector
  confidence. The 25 px gate now rejects that case.

## TACC environment and preflight

Use a compute node, not the login node. Request at least 12 CPU cores and one GPU
for the combined rollout/fit job; put the output/dataset under TACC scratch.
Exact SLURM account, partition, and module names are site/user specific—discover
them rather than guessing.

From the repository root:

```bash
pixi install --locked
export PYTHONPATH="$PWD"
export WANDB_MODE=offline
export MUJOCO_GL=egl

pixi run --as-is pytest -q aic_model/test/test_rl_insert_contract.py
pixi run --as-is python -m RL.student_teacher.train_student_a --help
```

Before the full job, construct one `make_student_env_a(...)`, assert student obs
shape `(69,)`, teacher obs shape `(21,)`, load `teacher_level1.zip`, obtain one
teacher target, and execute one `step_sim`. This is inference/physics only.
Do not proceed if any shape, finite-value, inward-axis, or teacher-load check fails.

## Full remote training job

Choose a fresh scratch directory. `--regen` deletes old `shard_*.npz` in that
output's dataset directory, so never point it at artifacts that must be retained.

```bash
export RUN_DIR="$SCRATCH/aic/student_gazebo_v1_seed0"

pixi run --as-is python -m RL.student_teacher.train_student_a \
  --teacher-zip RL/student_teacher/weights/teacher_level1.zip \
  --transitions 150000 --epochs 40 --num-envs 12 \
  --action-convention deploy \
  --wrench-mode zero \
  --feature-mode gazebo_v1 \
  --perception-noise 1.0 \
  --grasp-noise 1.0 \
  --level 1.0 \
  --dagger-iters 2 \
  --dagger-transitions 30000 \
  --dagger-epochs 10 \
  --eval-episodes 100 \
  --regen \
  --out "$RUN_DIR" \
  --seed 0
```

Do not judge the run from BC loss alone. Monitor `metrics.jsonl`, termination
counts, NaNs, worker exit codes, and whether DAgger shard counts increase after
each round. A successful DAgger run should contain base `shard_w...` plus
`shard_dagger_r01_...` and `shard_dagger_r02_...` files.

## Acceptance and export

Before export:

1. Evaluate the final checkpoint for at least 100 level-1 episodes with
   perception noise `1.0`, grasp noise `1.0`, zero wrench, and `gazebo_v1`.
2. Repeat held-out evaluation with multiple reset seeds. Target >=95% overall
   success and inspect all failures; a high mean hiding bad-collision failures is
   not acceptable.
3. Confirm the learned axial action is positive/inward in aligned outside-mouth
   states and lateral action opposes lateral error.
4. Preserve all metrics and configuration metadata.

Export:

```bash
pixi run --as-is python -m RL.student_teacher.export_student_a \
  --checkpoint "$RUN_DIR/student_a.pt" \
  --out "$RUN_DIR/final_insert_sfp_gazebo_v1.ts"
```

The exporter must report TorchScript parity at or below `1e-6` (normally `0.0`).
Bring back together:

```text
student_a.pt
final_insert_sfp_gazebo_v1.ts
final_insert_sfp_gazebo_v1.ts.contract.json
metrics.jsonl
train_meta.txt
```

Also retain the exact git SHA, SLURM script/job ID, Pixi lockfile, seed, and run
directory. Do not overwrite the legacy checkpoint/model; use versioned names.

## Back on the Gazebo/Flowstate workstation

1. Install the new `.ts` under `models/` and point `RL_INSERT_MODEL`/Dockerfile at
   it. Keep runtime `RL_INSERT_WRENCH_MODE=zero`.
2. Rebuild the `aic_model` package or image so both `RLInsert.py` and
   `rl_insert_contract.py` are installed.
3. Run AIC Engine/Gazebo with `ground_truth:=false`, bundled `best.pt`
   perception, attached cables, ACL enabled, and the existing contract safety
   limits.
4. Test in stages: perfect handoffs, near-perfect level-1 variance, then repeated
   randomized trials across every SFP module/slot. Stop on wrong axial direction,
   lateral growth, high force, or perception quality rejection.
5. Require repeated Gazebo insertion success near the target reliability (at
   least 19/20 as an initial gate), not only MuJoCo success.
6. Only then replace the submission model path, build the OCI image, run the full
   local compose evaluator/lifecycle/ACL checks, assign a new immutable tag, and
   upload/register it in Flowstate/the challenge portal.

## Non-negotiable prohibitions

- No object/scoring GT TF at runtime.
- Do not deploy the legacy Contract-A model as the final student.
- Do not re-enable raw FT fields until frame/sign/baseline parity is measured.
- Do not remove perception reprojection rejection to make trials continue.
- Do not accept a model based only on training loss or MuJoCo teacher-forced BC.
- Do not overwrite or lose `teacher_level1.zip`.
