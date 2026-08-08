#!/bin/bash
#SBATCH --job-name=aic-align-rl
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

# Align-first alignment RL (single-RL, align-only). No RLPD prior, no Gate-0,
# no teacher weights: this policy trains from scratch and rewards ALIGNMENT
# (lateral + rotation), holding z at a standoff. See:
#   RL/student_teacher/align_env.py
#   RL/student_teacher/train_align.py
#   docs/CURRENT_SYSTEM.md

set -euo pipefail

PROJECT_DIR="$WORK/aic-align-rl-20260712"
TAG="20260712"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export MUJOCO_GL=egl
export WANDB_MODE=offline
export OMP_NUM_THREADS=1
mkdir -p "$SCRATCH/aic/slurm"

# --- materialize the pixi env in this fresh checkout --------------------------
# A fresh `git checkout` has pixi.toml/pixi.lock but NO installed .pixi/envs/, so
# `pixi run --as-is` would fall through to system Python. Install first (as the
# from-checkout slurms train.slurm / refit_flowstate_v1.slurm / preflight.slurm do).
pixi install --locked

# --- compile gate --------------------------------------------------------------
pixi run --as-is python -m py_compile \
  RL/student_teacher/align_env.py \
  RL/student_teacher/train_align.py \
  RL/student_teacher/student_v3_sac.py

# --- SMOKE run: tiny budget, confirm it steps in MuJoCo + emits evaluation.json -
SMOKE_DIR="$SCRATCH/aic/align_smoke_${TAG}"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"
CUDA_VISIBLE_DEVICES=0 pixi run --as-is python -u \
  -m RL.student_teacher.train_align \
  --out "$SMOKE_DIR" \
  --stage align \
  --steps 2000 \
  --num-envs 2 \
  --learning-starts 200 \
  --eval-episodes 3 \
  --seed 999 \
  2>&1 | tee "$SMOKE_DIR/train.log"
test -f "$SMOKE_DIR/evaluation.json"   # smoke must produce an eval or we stop
echo "[smoke] OK -> $SMOKE_DIR/evaluation.json"
cat "$SMOKE_DIR/evaluation.json"

# --- FULL run: 3 seeds, align stage, 300k steps, one per GPU -------------------
run_seed() {
  local seed="$1"
  local gpu="$2"
  local run_dir="$SCRATCH/aic/align_rl_seed${seed}_${TAG}"
  test ! -e "$run_dir"
  mkdir -p "$run_dir"
  cp RL/student_teacher/tacc/train_align.slurm "$run_dir/submitted.slurm"
  CUDA_VISIBLE_DEVICES="$gpu" pixi run --as-is python -u \
    -m RL.student_teacher.train_align \
    --out "$run_dir" \
    --stage align \
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

# --- rank seeds by align success, then final errors ----------------------------
python - "$TAG" <<'PY'
import json, os, pathlib, sys
tag = sys.argv[1]
scratch = pathlib.Path(os.environ["SCRATCH"]) / "aic"
rows = []
for seed in range(3):
    path = scratch / f"align_rl_seed{seed}_{tag}/evaluation.json"
    rows.append(json.loads(path.read_text()))
rows.sort(key=lambda x: (
    -x["align_success_rate"], x["fail_rate"],
    x["final_lat_err_p50_mm"], x["final_rot_err_p50_deg"],
    x["steps_to_align_mean"]))
out = scratch / f"align_rl_selection_{tag}.json"
out.write_text(json.dumps({
    "ranking": rows,
    "continue_seeds": [row["seed"] for row in rows[:2]],
}, indent=2) + "\n")
print(out.read_text())
PY

touch "$SCRATCH/aic/align_rl_${TAG}_COMPLETE"
