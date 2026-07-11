#!/bin/bash
#SBATCH --job-name=aic-gzv1-s0
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="$WORK/aic-gazebo-v1-5465a5c"
RUN_DIR="$SCRATCH/aic/student_gazebo_v1_seed0_5465a5c_baseline"
ARTIFACT_BASENAME="final_insert_sfp_gazebo_v1_5465a5c_baseline_seed0"
cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export WANDB_MODE=offline
export WANDB_RUN_NAME="student_gazebo_v1_5465a5c_baseline_seed0"
export MUJOCO_GL=egl

test -f "$PROJECT_DIR/PREFLIGHT_PASSED"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp SOURCE_PROVENANCE.txt pixi.lock PREFLIGHT_REPORT.json PREFLIGHT_HASHES.txt "$RUN_DIR/"
cp RL/student_teacher/tacc/train.slurm "$RUN_DIR/submitted_train.slurm"

{
  echo "host=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "project_dir=$PROJECT_DIR"
  echo "run_dir=$RUN_DIR"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi
  pixi --version
} | tee "$RUN_DIR/run_details.txt"

# Repeat the no-training safety gate in the allocated training environment.
pixi install --locked
pixi run --as-is pytest -q aic_model/test/test_rl_insert_contract.py
pixi run --as-is python -m RL.student_teacher.tacc.preflight_student_a \
  | tee "$RUN_DIR/preflight_repeat.json"

pixi run --as-is python -m RL.student_teacher.train_student_a \
  --teacher-zip RL/student_teacher/weights/teacher_level1.zip \
  --transitions 150000 --epochs 40 --num-envs 12 \
  --action-convention deploy \
  --wrench-mode baseline \
  --feature-mode gazebo_v1 \
  --perception-noise 1.0 \
  --grasp-noise 1.0 \
  --level 1.0 \
  --dagger-iters 4 \
  --dagger-transitions 40000 \
  --dagger-epochs 12 \
  --eval-episodes 100 \
  --regen \
  --out "$RUN_DIR" \
  --seed 0

find "$RUN_DIR/dataset" -maxdepth 1 -type f -name 'shard_*.npz' \
  -printf '%f\n' | sort > "$RUN_DIR/dataset_shards.txt"
grep -q '^shard_w' "$RUN_DIR/dataset_shards.txt"
grep -q '^shard_dagger_r01_' "$RUN_DIR/dataset_shards.txt"
grep -q '^shard_dagger_r02_' "$RUN_DIR/dataset_shards.txt"
grep -q '^shard_dagger_r03_' "$RUN_DIR/dataset_shards.txt"
grep -q '^shard_dagger_r04_' "$RUN_DIR/dataset_shards.txt"

pixi run --as-is python -m RL.student_teacher.tacc.evaluate_student_a \
  --checkpoint "$RUN_DIR/student_a.pt" \
  --out "$RUN_DIR/held_out_evaluation.json" \
  --episodes-per-seed 100 \
  --seeds 10001 20002 30003 \
  --min-success 0.95

pixi run --as-is python -m RL.student_teacher.export_student_a \
  --checkpoint "$RUN_DIR/student_a.pt" \
  --out "$RUN_DIR/${ARTIFACT_BASENAME}.ts" \
  | tee "$RUN_DIR/export_report.json"

sha256sum \
  "$RUN_DIR/student_a.pt" \
  "$RUN_DIR/${ARTIFACT_BASENAME}.ts" \
  "$RUN_DIR/${ARTIFACT_BASENAME}.ts.contract.json" \
  "$RUN_DIR/metrics.jsonl" \
  "$RUN_DIR/train_meta.txt" \
  "$RUN_DIR/held_out_evaluation.json" \
  > "$RUN_DIR/artifact_sha256.txt"

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "status=complete"
} | tee -a "$RUN_DIR/run_details.txt"
touch "$RUN_DIR/RUN_COMPLETE"
