#!/bin/bash
#SBATCH --job-name=aic-seat-rl
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

# Three parallel seeds, one per GPU.  Within each seed, train nominal contact
# first, then warm-start full grasp/perception randomization from that seed's
# nominal checkpoint.  W&B credentials are read from the inherited environment;
# never put WANDB_API_KEY in this file.

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
mkdir -p "$SCRATCH/aic/slurm"

pixi install --locked
pixi run --as-is python -m py_compile \
  RL/student_teacher/seat_env.py \
  RL/student_teacher/student_v3_sac.py \
  RL/student_teacher/train_seat.py

run_seed() {
  local seed="$1"
  local gpu="$2"
  local nominal_dir="$SCRATCH/aic/seat_nominal_seed${seed}_${TAG}"
  local full_dir="$SCRATCH/aic/seat_full_seed${seed}_${TAG}"
  test ! -e "$nominal_dir"
  test ! -e "$full_dir"
  mkdir -p "$nominal_dir" "$full_dir"
  cp RL/student_teacher/tacc/train_seat.slurm "$nominal_dir/submitted.slurm"
  cp RL/student_teacher/tacc/train_seat.slurm "$full_dir/submitted.slurm"

  CUDA_VISIBLE_DEVICES="$gpu" pixi run --as-is python -u \
    -m RL.student_teacher.train_seat \
    --out "$nominal_dir" \
    --stage nominal \
    --steps "$STEPS_NOMINAL" \
    --num-envs "$NUM_ENVS" \
    --seed "$seed" \
    2>&1 | tee "$nominal_dir/train.log"
  test -f "$nominal_dir/model.zip"
  test -f "$nominal_dir/seat_actor.ts"

  CUDA_VISIBLE_DEVICES="$gpu" pixi run --as-is python -u \
    -m RL.student_teacher.train_seat \
    --out "$full_dir" \
    --stage full \
    --init-from "$nominal_dir/model.zip" \
    --steps "$STEPS_FULL" \
    --num-envs "$NUM_ENVS" \
    --seed "$seed" \
    2>&1 | tee "$full_dir/train.log"
  test -f "$full_dir/model.zip"
  test -f "$full_dir/seat_actor.ts"
  touch "$SCRATCH/aic/seat_rl_seed${seed}_${TAG}_COMPLETE"
}

run_seed 0 0 &
PID0=$!
run_seed 1 1 &
PID1=$!
run_seed 2 2 &
PID2=$!

STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
wait "$PID2" || STATUS=1
test "$STATUS" -eq 0

touch "$SCRATCH/aic/seat_rl_3seed_${TAG}_COMPLETE"
