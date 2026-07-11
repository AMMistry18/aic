#!/bin/bash
#SBATCH --job-name=aic-gzv1-preflight
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="$WORK/aic-gazebo-v1-5465a5c"
cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export WANDB_MODE=offline
export MUJOCO_GL=egl

echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
echo "project_dir=$PROJECT_DIR"
nvidia-smi
pixi --version
pixi install --locked
pixi run --as-is pytest -q aic_model/test/test_rl_insert_contract.py
pixi run --as-is python -m RL.student_teacher.train_student_a --help
pixi run --as-is python -m RL.student_teacher.tacc.preflight_student_a \
  | tee "$PROJECT_DIR/PREFLIGHT_REPORT.json"
sha256sum pixi.lock RL/student_teacher/weights/teacher_level1.zip \
  > "$PROJECT_DIR/PREFLIGHT_HASHES.txt"
touch "$PROJECT_DIR/PREFLIGHT_PASSED"
