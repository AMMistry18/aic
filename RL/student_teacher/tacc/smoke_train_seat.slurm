#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00

# Portable 8-vs-16 worker feasibility smoke for the seat SAC trainer.
# Supply the TACC allocation, partition, and log paths on the sbatch command
# line; do not add --gres/--gpus-per-task on Stampede3.

set -euo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the checked-out repository}"
: "${SCRATCH:?SCRATCH is not set}"

NUM_ENVS="${NUM_ENVS:-16}"
SMOKE_STEPS="${SMOKE_STEPS:-20000}"
SMOKE_EVAL_EPISODES="${SMOKE_EVAL_EPISODES:-9}"
SMOKE_TAG="${SMOKE_TAG:-seat_smoke_${NUM_ENVS}env_${SLURM_JOB_ID}}"
SMOKE_OUT="${SMOKE_OUT:-$SCRATCH/aic/$SMOKE_TAG}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-tar2}"
WANDB_ORGANIZATION="${WANDB_ORGANIZATION:-anshulmistry1-the-university-of-texas-at-austin-org}"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE
export WANDB_ENTITY
export WANDB_ORGANIZATION
export WANDB_REQUIRE_ONLINE=1

mkdir -p "$SMOKE_OUT"

{
  echo "project_dir=$PROJECT_DIR"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "host=$(hostname)"
  echo "num_envs=$NUM_ENVS"
  echo "smoke_steps=$SMOKE_STEPS"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
  echo "wandb_mode=$WANDB_MODE"
  echo "wandb_entity=$WANDB_ENTITY"
  echo "wandb_organization=$WANDB_ORGANIZATION"
} | tee "$SMOKE_OUT/smoke_meta.txt"

if [[ "${SKIP_PIXI_INSTALL:-0}" == "1" ]]; then
  test -x .pixi/envs/default/bin/python
  echo "[pixi] reusing prevalidated lock-identical environment"
else
  pixi install --locked
fi
pixi run --as-is python -m py_compile \
  RL/student_teacher/seat_env.py \
  RL/student_teacher/student_v3_sac.py \
  RL/student_teacher/train_seat.py \
  RL/student_teacher/evaluate_seat_checkpoints.py

nvidia-smi -L
pixi run --as-is python - <<'PY'
import torch
import wandb

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= 1, "no CUDA GPU is visible"
print(f"[gpu] {torch.cuda.get_device_name(0)}")
api = wandb.Api(timeout=30)
list(api.projects(entity="tar2", per_page=1))
assert api.default_entity == "tar2", api.default_entity
print("[wandb] authenticated team entity tar2")
PY

start_seconds=$SECONDS
pixi run --as-is python -u -m RL.student_teacher.train_seat \
  --out "$SMOKE_OUT" \
  --stage deployment \
  --steps "$SMOKE_STEPS" \
  --num-envs "$NUM_ENVS" \
  --seed 1600 \
  --gradient-steps -1 \
  --checkpoint-freq "$SMOKE_STEPS" \
  --eval-freq "$SMOKE_STEPS" \
  --eval-episodes "$SMOKE_EVAL_EPISODES" \
  --final-eval-episodes "$SMOKE_EVAL_EPISODES" \
  --wandb-log-freq 1000 \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-organization "$WANDB_ORGANIZATION" \
  --require-online-wandb \
  --no-video \
  2>&1 | tee "$SMOKE_OUT/train.log"
elapsed_seconds=$((SECONDS - start_seconds))

test -f "$SMOKE_OUT/model.zip"
test -f "$SMOKE_OUT/replay_buffer.pkl"
test -f "$SMOKE_OUT/evaluation.json"
test -f "$SMOKE_OUT/seat_actor.ts"
test -f "$SMOKE_OUT/best_model.zip"
test -f "$SMOKE_OUT/best_seat_actor.ts"
test -f "$SMOKE_OUT/final_selection_evaluation.json"

{
  echo "elapsed_seconds=$elapsed_seconds"
  echo "effective_steps_per_second=$(awk -v steps="$SMOKE_STEPS" -v seconds="$elapsed_seconds" 'BEGIN { if (seconds > 0) printf "%.6f", steps / seconds; else print "nan" }')"
  echo "result=PASS"
} | tee -a "$SMOKE_OUT/smoke_meta.txt"
