#!/bin/bash
#SBATCH --job-name=aic-v3-pilots
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="$WORK/aic-residual-v3-pilot-20260712"
GATE_REPORT="$SCRATCH/aic/residual_v3_gate0_20260712/gate0_report.json"
PRIOR_DIR="$SCRATCH/aic/residual_v3_prior_20260712"
EXPECTED_TEACHER_SHA="fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl
export WANDB_MODE=offline
export OMP_NUM_THREADS=1
mkdir -p "$SCRATCH/aic/slurm"

python - "$GATE_REPORT" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Gate-0 report missing: {path}")
report = json.loads(path.read_text())
if report.get("passed") is not True:
    raise SystemExit("Gate-0 did not pass; pilot launch forbidden")
PY

ACTUAL_TEACHER_SHA="$(sha256sum RL/student_teacher/weights/teacher_level1.zip | awk '{print $1}')"
test "$ACTUAL_TEACHER_SHA" = "$EXPECTED_TEACHER_SHA"
test -f models/final_insert_sfp_flowstate_v1.ts
pixi run --as-is python -m pytest -q \
  RL/student_teacher/test_gate0_contact_jam.py \
  RL/student_teacher/test_student_v3_sac.py
pixi run --as-is python -m py_compile \
  RL/student_teacher/student_v3_env.py \
  RL/student_teacher/student_v3_sac.py \
  RL/student_teacher/student_v3_prior.py \
  RL/student_teacher/train_student_v3.py

if [[ ! -f "$PRIOR_DIR/manifest.json" ]]; then
  test ! -e "$PRIOR_DIR"
  mkdir -p "$PRIOR_DIR"
  pixi run --as-is python -u -m RL.student_teacher.student_v3_prior \
    --out "$PRIOR_DIR" \
    --teacher-transitions 30000 \
    --student-transitions 15000 \
    --failure-transitions 15000 \
    --seed 20260712 \
    2>&1 | tee "$PRIOR_DIR/collector.log"
fi

run_seed() {
  local seed="$1"
  local gpu="$2"
  local run_dir="$SCRATCH/aic/residual_v3_pilot_seed${seed}_20260712"
  test ! -e "$run_dir"
  mkdir -p "$run_dir"
  cp RL/student_teacher/tacc/train_student_v3_pilots.slurm "$run_dir/submitted.slurm"
  CUDA_VISIBLE_DEVICES="$gpu" pixi run --as-is python -u \
    -m RL.student_teacher.train_student_v3 \
    --prior-manifest "$PRIOR_DIR/manifest.json" \
    --out "$run_dir" \
    --stage contact \
    --steps 300000 \
    --num-envs 12 \
    --seed "$seed" \
    2>&1 | tee "$run_dir/train.log"
}

run_seed 0 0 &
PID0=$!
run_seed 1 1 &
PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
test "$STATUS" -eq 0
run_seed 2 0

python - <<'PY'
import json, os, pathlib
scratch = pathlib.Path(os.environ["SCRATCH"]) / "aic"
rows = []
for seed in range(3):
    path = scratch / f"residual_v3_pilot_seed{seed}_20260712/evaluation.json"
    rows.append(json.loads(path.read_text()))
rows.sort(key=lambda x: (
    x["collision_rate"], -x["success_rate"], x["force_abort_rate"],
    x["peak_force_p95_n"], x["stall_steps_mean"]))
out = scratch / "residual_v3_pilot_selection_20260712.json"
out.write_text(json.dumps({
    "ranking": rows,
    "continue_seeds": [row["seed"] for row in rows[:2]],
}, indent=2) + "\n")
print(out.read_text())
PY

touch "$SCRATCH/aic/residual_v3_pilots_20260712_COMPLETE"
