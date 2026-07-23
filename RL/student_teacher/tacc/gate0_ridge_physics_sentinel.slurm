#!/bin/bash
# Physics-only guided-controller sentinel for ridge/contact tuning.
# This uses a fresh disposable source copy; it must never alter prior snapshots.
#SBATCH --job-name=aic-ridge-physics
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/aic-ridge-physics-20260712}"
RUN_DIR="${RUN_DIR:-$SCRATCH/aic/gate0_ridge_fix_20260712}"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl
export WANDB_MODE=offline

test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR" "$SCRATCH/aic/slurm"
cp RL/student_teacher/tacc/gate0_ridge_physics_sentinel.slurm "$RUN_DIR/submitted.slurm"
grep -Fq 'self.model.geom_solref[self._ridge_geom_ids] = (0.006, 1.0)' \
  RL/scene_env.py

{
  echo "host=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "project_dir=$PROJECT_DIR"
  echo "run_dir=$RUN_DIR"
  echo "scene_ridge_solref=(0.006, 1.0)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  pixi --version
} | tee "$RUN_DIR/run_details.txt"

pixi run --as-is python -m py_compile RL/scene_env.py RL/student_teacher/gate0_contact_jam.py

set +e
pixi run --as-is python -u -m RL.student_teacher.gate0_contact_jam \
  --controller guided \
  --nominal-episodes 10 \
  --randomized-episodes 40 \
  --seed 20260712 \
  --level 0.65 \
  --output "$RUN_DIR/gate0_ridge_fix.json" \
  2>&1 | tee "$RUN_DIR/gate0_ridge_fix.log"
STATUS=${PIPESTATUS[0]}
set -e

pixi run --as-is python - "$RUN_DIR/gate0_ridge_fix.json" "$RUN_DIR/gate0_ridge_fix.log" <<'PY'
import json
import re
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
log = Path(sys.argv[2]).read_text()
rows = report['randomized']['results'] + report['nominal']['results']
traces = [point for row in rows for point in row['trace']]
peak_lateral_mm = max(point['lateral_mm'] for point in traces)
peak_force_n = max(point['force_n'] for point in traces)
jumps = [
    abs(trace[i]['lateral_mm'] - trace[i - 1]['lateral_mm'])
    for row in rows for trace in [row['trace']] for i in range(1, len(trace))
]
summary = {
    'nominal_success_rate': report['nominal']['success_rate'],
    'randomized_success_rate': report['randomized']['success_rate'],
    'randomized_counts': report['randomized']['counts'],
    'qacc_warning_count': len(re.findall(r'QACC', log)),
    'peak_lateral_mm': peak_lateral_mm,
    'max_single_step_lateral_jump_mm': max(jumps, default=0.0),
    'peak_force_n': peak_force_n,
}
Path(sys.argv[1]).with_name('physics_summary.json').write_text(
    json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY

echo "gate_exit_status=$STATUS" | tee -a "$RUN_DIR/run_details.txt"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_DIR/run_details.txt"
touch "$RUN_DIR/RUN_FINISHED"
exit "$STATUS"
