#!/bin/bash
#SBATCH --job-name=aic-seat-rl
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

# One exclusive-node job runs seed 0's full nominal-to-full curriculum.  rtx-small
# has two GPUs per node but no Slurm GPU GRES, so CUDA_VISIBLE_DEVICES=0 exposes
# exactly one GPU to the job.  W&B credentials are inherited from the environment.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/aic-seat-rl-20260713}"
STEPS_NOMINAL="${STEPS_NOMINAL:-300000}"
STEPS_FULL="${STEPS_FULL:-300000}"
NUM_ENVS="${NUM_ENVS:-8}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-50000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-1000}"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl
export WANDB_MODE=online
export OMP_NUM_THREADS=1
# rtx-small does not expose GPUs as Slurm GRES.  The exclusive-node allocation
# keeps tasks on distinct nodes; expose exactly one physical device per task.
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$SCRATCH/aic/slurm"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[wandb] WANDB_API_KEY is present; requesting online mode"
else
  echo "[wandb] WANDB_API_KEY is absent; trainer will attempt existing auth, then fall back offline"
fi

pixi install --locked
pixi run --as-is python -m py_compile \
  RL/student_teacher/seat_env.py \
  RL/student_teacher/student_v3_sac.py \
  RL/student_teacher/train_seat.py

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L
pixi run --as-is python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, (
    f"expected exactly one visible GPU, found {torch.cuda.device_count()}"
)
print(f"[gpu] torch sees exactly one GPU: {torch.cuda.get_device_name(0)}")
PY

SEED=0
GIT_COMMIT="$(git rev-parse --short=12 HEAD)"
TAG="${TAG:-phase3_seed${SEED}_${GIT_COMMIT}_${SLURM_JOB_ID}}"
nominal_dir="$SCRATCH/aic/seat_nominal_seed${SEED}_${TAG}"
full_dir="$SCRATCH/aic/seat_full_seed${SEED}_${TAG}"

run_stage() {
  local stage="$1"
  local out_dir="$2"
  local init_from="${3:-}"
  local -a resume_args=()

  if [[ -f "$out_dir/RUN_COMPLETE" ]]; then
    echo "[resume] ${stage} already complete: $out_dir"
    return
  fi
  mkdir -p "$out_dir"
  cp RL/student_teacher/tacc/train_seat.slurm "$out_dir/submitted.slurm"
  if [[ -f "$out_dir/model.zip" ]]; then
    resume_args=(--resume)
    echo "[resume] resuming ${stage} from $out_dir/model.zip"
  elif [[ -n "$init_from" ]]; then
    test -f "$init_from"
    resume_args=(--init-from "$init_from")
    echo "[train] warm-starting ${stage} from $init_from"
  fi

  pixi run --as-is python -u \
    -m RL.student_teacher.train_seat \
    --out "$out_dir" \
    --stage "$stage" \
    --steps "${stage_steps[$stage]}" \
    --num-envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --eval-freq "$EVAL_FREQ" \
    --eval-episodes "$EVAL_EPISODES" \
    --wandb-log-freq "$WANDB_LOG_FREQ" \
    --video \
    "${resume_args[@]}" \
    2>&1 | tee "$out_dir/train.log"
  test -f "$out_dir/model.zip"
  test -f "$out_dir/replay_buffer.pkl"
  test -f "$out_dir/progress.json"
  test -f "$out_dir/seat_actor.ts"
}

declare -A stage_steps=(
  [nominal]="$STEPS_NOMINAL"
  [full]="$STEPS_FULL"
)
echo "[train] seed=$SEED tag=$TAG commit=$GIT_COMMIT time_limit=2-00:00:00"
echo "[train] checkpoint_freq=$CHECKPOINT_FREQ eval_freq=$EVAL_FREQ wandb_log_freq=$WANDB_LOG_FREQ"
run_stage nominal "$nominal_dir"
run_stage full "$full_dir" "$nominal_dir/model.zip"
touch "$SCRATCH/aic/seat_rl_seed${SEED}_${TAG}_COMPLETE"
