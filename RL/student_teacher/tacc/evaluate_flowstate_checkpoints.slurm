#!/bin/bash
#SBATCH --job-name=aic-flow-eval
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="$WORK/aic-flowstate-v1-5900b41"
RUN_DIR="$SCRATCH/aic/student_flowstate_v1_seed0_5900b41"
EPOCHS=(10 25 30)
EPOCH="${EPOCHS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="$RUN_DIR/student_a_ep$(printf '%03d' "$EPOCH").pt"
REPORT="$RUN_DIR/held_out_evaluation_ep$(printf '%03d' "$EPOCH").json"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl

test -f "$CHECKPOINT"
set +e
pixi run --as-is python -m RL.student_teacher.tacc.evaluate_student_a \
  --checkpoint "$CHECKPOINT" \
  --out "$REPORT" \
  --episodes-per-seed 100 \
  --seeds 10001 20002 30003 \
  --min-success 0.0
STATUS=$?
set -e

test -s "$REPORT"
echo "checkpoint=$CHECKPOINT evaluator_status=$STATUS"
