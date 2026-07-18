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
STEPS_BOOTSTRAP="${STEPS_BOOTSTRAP:-200000}"
STEPS_DEPLOYMENT="${STEPS_DEPLOYMENT:-400000}"
NUM_ENVS="${NUM_ENVS:-8}"
SEED="${SEED:-0}"
LR_BOOTSTRAP="${LR_BOOTSTRAP:-3e-4}"
LR_DEPLOYMENT="${LR_DEPLOYMENT:-1e-4}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-40000}"
EVAL_FREQ="${EVAL_FREQ:-40000}"
EVAL_EPISODES="${EVAL_EPISODES:-60}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-180}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-1000}"
VIDEO_FREQ="${VIDEO_FREQ:-50000}"
GRADIENT_STEPS="${GRADIENT_STEPS:--1}"
# The post-cache local 8-env smoke sustained 6 fps while doing one SAC update
# per transition on CPU.  600k / 6 = 27.8 h, leaving >20 h inside the 48 h
# allocation for startup, evaluation, video, export, and TACC variance.
BUDGET_FPS_FLOOR="${BUDGET_FPS_FLOOR:-6}"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-tar2}"
export WANDB_ORGANIZATION="${WANDB_ORGANIZATION:-anshulmistry1-the-university-of-texas-at-austin-org}"
export WANDB_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=1
# rtx-small does not expose GPUs as Slurm GRES.  The exclusive-node allocation
# keeps tasks on distinct nodes; expose exactly one physical device per task.
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$SCRATCH/aic/slurm"

echo "[wandb] online required entity=$WANDB_ENTITY organization=$WANDB_ORGANIZATION"

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

GIT_COMMIT="$(git rev-parse --short=12 HEAD)"
TAG="${TAG:-phase3_seed${SEED}_${GIT_COMMIT}_${SLURM_JOB_ID}}"
bootstrap_dir="$SCRATCH/aic/seat_bootstrap_seed${SEED}_${TAG}"
deployment_dir="$SCRATCH/aic/seat_deployment_seed${SEED}_${TAG}"
total_steps=$((STEPS_BOOTSTRAP + STEPS_DEPLOYMENT))
estimated_seconds=$(((total_steps + BUDGET_FPS_FLOOR - 1) / BUDGET_FPS_FLOOR))
checkpoint_seconds=$(((CHECKPOINT_FREQ + BUDGET_FPS_FLOOR - 1) / BUDGET_FPS_FLOOR))
if (( estimated_seconds > 40 * 60 * 60 )); then
  echo "[budget] ERROR: ${total_steps} steps at ${BUDGET_FPS_FLOOR} fps " \
       "needs ${estimated_seconds}s, exceeding the 40h training budget"
  exit 2
fi

run_stage() {
  local stage="$1"
  local out_dir="$2"
  local init_from="${3:-}"
  local learning_rate="${stage_learning_rates[$stage]}"
  local tb_dir="$PROJECT_DIR/.tensorboard/$TAG/$stage"
  local wandb_name="seat_${stage}_seed${SEED}_${GIT_COMMIT}_job${SLURM_JOB_ID}"
  local -a resume_args=()

  if [[ -f "$out_dir/RUN_COMPLETE" ]]; then
    echo "[resume] ${stage} already complete: $out_dir"
    return
  fi
  mkdir -p "$out_dir"
  mkdir -p "$tb_dir"
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
    --gradient-steps "$GRADIENT_STEPS" \
    --learning-rate "$learning_rate" \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --video-freq "$VIDEO_FREQ" \
    --eval-freq "$EVAL_FREQ" \
    --eval-episodes "$EVAL_EPISODES" \
    --final-eval-episodes "$FINAL_EVAL_EPISODES" \
    --wandb-log-freq "$WANDB_LOG_FREQ" \
    --tensorboard-log "$tb_dir" \
    --wandb-run-name "$wandb_name" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-organization "$WANDB_ORGANIZATION" \
    --require-online-wandb \
    --video \
    "${resume_args[@]}" \
    2>&1 | tee "$out_dir/train.log"
  test -f "$out_dir/model.zip"
  test -f "$out_dir/replay_buffer.pkl"
  test -f "$out_dir/progress.json"
  test -f "$out_dir/seat_actor.ts"
  test -f "$out_dir/best_model.zip"
  test -f "$out_dir/best_seat_actor.ts"
  test -f "$out_dir/final_selection_evaluation.json"
}

declare -A stage_steps=(
  [bootstrap]="$STEPS_BOOTSTRAP"
  [deployment]="$STEPS_DEPLOYMENT"
)
declare -A stage_learning_rates=(
  [bootstrap]="$LR_BOOTSTRAP"
  [deployment]="$LR_DEPLOYMENT"
)
echo "[train] seed=$SEED tag=$TAG commit=$GIT_COMMIT time_limit=2-00:00:00"
echo "[budget] total_steps=$total_steps fps_floor=$BUDGET_FPS_FLOOR estimated_seconds=$estimated_seconds checkpoint_seconds=$checkpoint_seconds"
echo "[train] checkpoint_freq=$CHECKPOINT_FREQ eval_freq=$EVAL_FREQ video_freq=$VIDEO_FREQ wandb_log_freq=$WANDB_LOG_FREQ gradient_steps=$GRADIENT_STEPS"
run_stage bootstrap "$bootstrap_dir"
run_stage deployment "$deployment_dir" "$bootstrap_dir/best_model.zip"
touch "$SCRATCH/aic/seat_rl_seed${SEED}_${TAG}_COMPLETE"
