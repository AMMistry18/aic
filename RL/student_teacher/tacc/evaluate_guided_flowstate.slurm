#!/bin/bash
#SBATCH --job-name=aic-guided-eval
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="$WORK/aic-flowstate-v1-5900b41"
RUN_DIR="$SCRATCH/aic/guided_flowstate_v2_20260711"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl

mkdir -p "$RUN_DIR"

pixi run --as-is python -m RL.student_teacher.parity.evaluate_guided_controller \
  --episodes-per-seed 100 \
  --seeds 10001 20002 30003 \
  --perception-noise 0 \
  --grasp-noise 0 \
  --output "$RUN_DIR/exact_pose_300.json"

set +e
pixi run --as-is python -m RL.student_teacher.parity.evaluate_guided_controller \
  --episodes-per-seed 100 \
  --seeds 10001 20002 30003 \
  --perception-noise 1 \
  --grasp-noise 1 \
  --output "$RUN_DIR/noisy_pose_300.json"
NOISY_STATUS=$?
set -e

sha256sum "$RUN_DIR"/*.json > "$RUN_DIR/artifact_sha256.txt"
echo "noisy_evaluator_status=$NOISY_STATUS"
touch "$RUN_DIR/RUN_COMPLETE"
