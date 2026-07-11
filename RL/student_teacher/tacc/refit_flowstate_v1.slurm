#!/bin/bash
#SBATCH --job-name=aic-flow-v1
#SBATCH --account=IRI26004
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/11590/satya_a/aic/slurm/%x-%j.out
#SBATCH --error=/scratch/11590/satya_a/aic/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="$WORK/aic-flowstate-v1-5900b41"
BASELINE_RUN="$SCRATCH/aic/student_gazebo_v1_seed0_5465a5c_baseline"
RUN_DIR="$SCRATCH/aic/student_flowstate_v1_seed0_5900b41"
PARITY_DIR="$SCRATCH/aic/flowstate_parity_20260711"

cd "$PROJECT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR"
export WANDB_MODE=offline
export WANDB_RUN_NAME="student_flowstate_v1_seed0_5900b41"
export MUJOCO_GL=egl

test -d "$BASELINE_RUN/dataset"
test -f "$PARITY_DIR/flowstate_handoff_20260711.json"
test -f "$PARITY_DIR/mujoco_handoffs.json"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"
ln -s "$BASELINE_RUN/dataset" "$RUN_DIR/dataset"

{
  echo "host=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "project_dir=$PROJECT_DIR"
  echo "baseline_dataset=$BASELINE_RUN/dataset"
  echo "run_dir=$RUN_DIR"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi
  pixi --version
} | tee "$RUN_DIR/run_details.txt"

pixi install --locked
pixi run --as-is pytest -q aic_model/test/test_rl_insert_contract.py

pixi run --as-is python -m RL.student_teacher.train_student_a \
  --teacher-zip RL/student_teacher/weights/teacher_level1.zip \
  --epochs 40 --num-envs 1 \
  --action-convention deploy \
  --wrench-mode baseline \
  --feature-mode flowstate_v1 \
  --perception-noise 1.0 \
  --grasp-noise 1.0 \
  --level 1.0 \
  --eval-every 5 --eval-episodes 50 \
  --out "$RUN_DIR" \
  --seed 0

pixi run --as-is python -m RL.student_teacher.tacc.evaluate_student_a \
  --checkpoint "$RUN_DIR/student_a.pt" \
  --out "$RUN_DIR/held_out_evaluation.json" \
  --episodes-per-seed 100 \
  --seeds 10001 20002 30003 \
  --min-success 0.0

pixi run --as-is python -m RL.student_teacher.export_student_a \
  --checkpoint "$RUN_DIR/student_a.pt" \
  --out "$RUN_DIR/final_insert_sfp_flowstate_v1.ts" \
  | tee "$RUN_DIR/export_report.json"

pixi run --as-is python "$PARITY_DIR/check_candidate_on_flowstate.py" \
  --checkpoint "$RUN_DIR/student_a.pt" \
  --torchscript "$RUN_DIR/final_insert_sfp_flowstate_v1.ts" \
  --flowstate "$PARITY_DIR/flowstate_handoff_20260711.json" \
  --mujoco "$PARITY_DIR/mujoco_handoffs.json" \
  --output "$RUN_DIR/flowstate_fixture_check.json"

sha256sum \
  "$RUN_DIR/student_a.pt" \
  "$RUN_DIR/final_insert_sfp_flowstate_v1.ts" \
  "$RUN_DIR/final_insert_sfp_flowstate_v1.ts.contract.json" \
  "$RUN_DIR/held_out_evaluation.json" \
  "$RUN_DIR/flowstate_fixture_check.json" \
  > "$RUN_DIR/artifact_sha256.txt"

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "status=complete"
} | tee -a "$RUN_DIR/run_details.txt"
touch "$RUN_DIR/RUN_COMPLETE"
