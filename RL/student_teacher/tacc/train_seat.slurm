#!/bin/bash
#SBATCH --job-name=aic-seat-rl
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

# One exclusive-node job runs seed 0's full nominal-to-full curriculum.  rtx-small
# has two GPUs per node but no Slurm GPU GRES, so CUDA_VISIBLE_DEVICES=0 exposes
# exactly one GPU to the job.  W&B credentials are inherited from the environment.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/aic-seat-rl-20260713}"
TAG="${TAG:-20260713}"
STEPS_NOMINAL="${STEPS_NOMINAL:-300000}"
STEPS_FULL="${STEPS_FULL:-300000}"
NUM_ENVS="${NUM_ENVS:-8}"

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
nominal_dir="$SCRATCH/aic/seat_nominal_seed${SEED}_${TAG}"
full_dir="$SCRATCH/aic/seat_full_seed${SEED}_${TAG}"
test ! -e "$nominal_dir"
test ! -e "$full_dir"
mkdir -p "$nominal_dir" "$full_dir"
cp RL/student_teacher/tacc/train_seat.slurm "$nominal_dir/submitted.slurm"
cp RL/student_teacher/tacc/train_seat.slurm "$full_dir/submitted.slurm"

pixi run --as-is python -u \
  -m RL.student_teacher.train_seat \
  --out "$nominal_dir" \
  --stage nominal \
  --steps "$STEPS_NOMINAL" \
  --num-envs "$NUM_ENVS" \
  --seed "$SEED" \
  2>&1 | tee "$nominal_dir/train.log"
test -f "$nominal_dir/model.zip"
test -f "$nominal_dir/seat_actor.ts"

pixi run --as-is python -u \
  -m RL.student_teacher.train_seat \
  --out "$full_dir" \
  --stage full \
  --init-from "$nominal_dir/model.zip" \
  --steps "$STEPS_FULL" \
  --num-envs "$NUM_ENVS" \
  --seed "$SEED" \
  2>&1 | tee "$full_dir/train.log"
test -f "$full_dir/model.zip"
test -f "$full_dir/seat_actor.ts"
touch "$SCRATCH/aic/seat_rl_seed${SEED}_${TAG}_COMPLETE"
