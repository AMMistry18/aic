#!/bin/bash
#SBATCH --job-name=aic-v3-gate0
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="$WORK/aic-residual-v3-gate0-20260712"
RUN_DIR="$SCRATCH/aic/residual_v3_gate0_20260712"
EXPECTED_TEACHER_SHA="fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl
export WANDB_MODE=offline

test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR" "$SCRATCH/aic/slurm"
cp RL/student_teacher/tacc/gate0_contact_jam.slurm "$RUN_DIR/submitted.slurm"

ACTUAL_TEACHER_SHA="$(sha256sum RL/student_teacher/weights/teacher_level1.zip | awk '{print $1}')"
test "$ACTUAL_TEACHER_SHA" = "$EXPECTED_TEACHER_SHA"

{
  echo "host=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "project_dir=$PROJECT_DIR"
  echo "run_dir=$RUN_DIR"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "teacher_sha256=$ACTUAL_TEACHER_SHA"
  pixi --version
} | tee "$RUN_DIR/run_details.txt"

pixi run --as-is python -m py_compile \
  RL/scene_env.py \
  RL/student_teacher/student_env_a.py \
  RL/student_teacher/gate0_contact_jam.py
pixi run --as-is python -m pytest -q \
  RL/student_teacher/test_gate0_contact_jam.py

set +e
pixi run --as-is python -u -m RL.student_teacher.gate0_contact_jam \
  --controller guided \
  --nominal-episodes 10 \
  --randomized-episodes 40 \
  --seed 20260712 \
  --level 0.65 \
  --output "$RUN_DIR/gate0_report.json" \
  2>&1 | tee "$RUN_DIR/gate0_console.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e

{
  echo "gate_exit_status=$GATE_STATUS"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$RUN_DIR/run_details.txt"
sha256sum "$RUN_DIR/gate0_report.json" "$RUN_DIR/gate0_console.log" \
  > "$RUN_DIR/artifact_sha256.txt"
touch "$RUN_DIR/RUN_FINISHED"
if [[ "$GATE_STATUS" -eq 0 ]]; then
  touch "$RUN_DIR/GATE0_PASSED"
fi
exit "$GATE_STATUS"
