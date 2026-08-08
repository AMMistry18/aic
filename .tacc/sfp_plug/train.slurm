#!/usr/bin/env bash
#SBATCH --job-name=sfp-pose-train
#SBATCH --partition=rtx-small
#SBATCH --account=IRI26004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work2/11590/satya_a/stampede3/aic-sfp-pose-datagen-20260719/training/logs/sfp-pose-train-%j.out

set -eo pipefail

readonly job_root=/work2/11590/satya_a/stampede3/aic-sfp-pose-datagen-20260719
readonly repo_root=/work2/11590/satya_a/stampede3/aic-gazebo-v1-9ae671e
readonly dataset_yaml="${job_root}/dataset/aic_sfp_plug_pose.yaml"
readonly training_root="${job_root}/training"

mkdir -p \
  "${training_root}/logs" \
  "${training_root}/runs" \
  "${training_root}/reports" \
  "${training_root}/cache/ultralytics" \
  "${training_root}/cache/torch" \
  "${training_root}/tmp"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_GROUP
export YOLO_CONFIG_DIR="${training_root}/cache/ultralytics"
export TORCH_HOME="${training_root}/cache/torch"
export TMPDIR="${training_root}/tmp"

test -f "${dataset_yaml}"
cd -- "${repo_root}"
source .pixi/envs/default/setup.bash

.pixi/envs/default/bin/python tools/perception/sfp_plug/train.py \
  --data "${dataset_yaml}" \
  --model yolo11s-pose.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 8 \
  --workers 8 \
  --device 0 \
  --project "${training_root}/runs" \
  --name sfp_plug_pose_yolo11s_960 \
  --final-weights "${training_root}/best_sfp_plug_pose.pt" \
  --metrics-out "${training_root}/reports/train_sfp_plug_pose.json"
