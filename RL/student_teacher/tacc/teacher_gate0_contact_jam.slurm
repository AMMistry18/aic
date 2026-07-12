#!/bin/bash
#SBATCH --job-name=aic-teacher-gate0
#SBATCH --account=IRI26004
#SBATCH --partition=skx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="$WORK/aic-teacher-gate0-20260712"
RUN_DIR="$SCRATCH/aic/teacher_gate0_20260712"
EXPECTED_TEACHER_SHA="fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=disable
export WANDB_MODE=offline

test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR" "$SCRATCH/aic/slurm"
cp RL/student_teacher/tacc/teacher_gate0_contact_jam.slurm "$RUN_DIR/submitted.slurm"

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

pixi run --as-is python -m py_compile RL/student_teacher/gate0_contact_jam.py
pixi run --as-is python -m pytest -q RL/student_teacher/test_gate0_contact_jam.py

set +e
pixi run --as-is python -u -m RL.student_teacher.gate0_contact_jam \
  --controller teacher \
  --teacher-zip RL/student_teacher/weights/teacher_level1.zip \
  --nominal-episodes 10 \
  --randomized-episodes 40 \
  --seed 20260712 \
  --level 0.65 \
  --jam-min-force-n 20 \
  --output "$RUN_DIR/teacher_gate0_report.json" \
  2>&1 | tee "$RUN_DIR/teacher_gate0_console.log"
SEMANTIC_STATUS=${PIPESTATUS[0]}
set -e

python - "$RUN_DIR/teacher_gate0_report.json" "$SEMANTIC_STATUS" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
randomized = report["randomized"]
success = float(randomized["success_rate"])
classification = (
    "hard_contact_valid" if success >= 0.80 else "nominal_regime_only")
summary = {
    "controller": report["controller"],
    "teacher_sha256": report["teacher_sha256"],
    "nominal_success_rate": report["nominal"]["success_rate"],
    "randomized_success_rate": success,
    "randomized_jam_rate": randomized["jam_count"] / randomized["episodes"],
    "randomized_counts": randomized["counts"],
    "classification": classification,
    "semantic_gate_exit_status": int(sys.argv[2]),
}
pathlib.Path(sys.argv[1]).with_name("teacher_regime_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

{
  echo "semantic_gate_exit_status=$SEMANTIC_STATUS"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$RUN_DIR/run_details.txt"
sha256sum "$RUN_DIR"/teacher_gate0_report.json \
  "$RUN_DIR"/teacher_gate0_console.log \
  "$RUN_DIR"/teacher_regime_summary.json \
  > "$RUN_DIR/artifact_sha256.txt"
touch "$RUN_DIR/RUN_FINISHED"

# A weak teacher result is scientific evidence, not an infrastructure failure.
exit 0
