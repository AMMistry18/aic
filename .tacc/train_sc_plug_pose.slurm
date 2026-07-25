#!/usr/bin/env bash
# Train only after the SC collection gate has verified nonempty train/val/test
# splits and matching 29-token pose labels.
#SBATCH --job-name=sc-plug-pose-train
#SBATCH --partition=rtx-small
#SBATCH --account=IRI26004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260719/training/logs/sc-plug-pose-train-%j.out

set -euo pipefail

readonly job_root=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260719
readonly repo_root=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-src-20260719
readonly dataset_root="${job_root}/dataset"
readonly dataset_yaml="${dataset_root}/aic_sc_plug_pose.yaml"
readonly training_root="${job_root}/training"
readonly min_total_images="${SC_MIN_TOTAL_IMAGES:-3600}"
readonly min_split_images="${SC_MIN_SPLIT_IMAGES:-300}"
readonly dataloader_workers="${SC_TRAIN_WORKERS:-0}"

mkdir -p \
  "${training_root}/logs" \
  "${training_root}/runs" \
  "${training_root}/reports" \
  "${training_root}/cache/ultralytics" \
  "${training_root}/cache/torch" \
  "${training_root}/tmp"

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_GROUP
export YOLO_CONFIG_DIR="${training_root}/cache/ultralytics"
export TORCH_HOME="${training_root}/cache/torch"
# PyTorch worker IPC uses AF_UNIX sockets below TMPDIR.  The work2 path is
# longer than the kernel's socket-path limit, so use a short node-local path.
export TMPDIR="/tmp/aic-sc-train-${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

test -f "${dataset_yaml}"
cd -- "${repo_root}"
# Pixi's generated ROS setup scripts read several optional variables directly.
# Source them with the same nounset behavior as an interactive shell, then
# restore strict handling for the dataset gate and training command.
set +u
source .pixi/envs/default/setup.bash
set -u

.pixi/envs/default/bin/python - "${dataset_root}" \
  "${min_total_images}" "${min_split_images}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
minimum_total = int(sys.argv[2])
minimum_split = int(sys.argv[3])
counts = {}
for split in ("train", "val", "test"):
    images = sorted((root / "images" / split).glob("*.png"))
    labels = root / "labels" / split
    if len(images) < minimum_split:
        raise SystemExit(f"{split} has only {len(images)} images")
    for image in images:
        tokens = (labels / f"{image.stem}.txt").read_text().split()
        if len(tokens) != 29:
            raise SystemExit(f"invalid pose label for {image.name}: {len(tokens)} tokens")
    counts[split] = len(images)
if sum(counts.values()) < minimum_total:
    raise SystemExit(f"dataset too small: {counts}, required total {minimum_total}")
print(f"SC dataset gate passed: {counts}")
PY

# The SC trainer re-checks nonempty train/val splits and 29-token pose labels
# before it opens a GPU, then copies the best checkpoint to the SC path.
.pixi/envs/default/bin/python train_sc_plug_pose.py \
  --data "${dataset_yaml}" \
  --model yolo11s-pose.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 8 \
  --workers "${dataloader_workers}" \
  --device 0 \
  --project "${training_root}/runs" \
  --name sc_plug_pose_yolo11s_960 \
  --final-weights "${training_root}/best_sc_plug_pose.pt" \
  --metrics-out "${training_root}/reports/train_sc_plug_pose.json" \
  --seed 20260720

# Geometric validation against simulator TF ground truth.  This, not the YOLO
# keypoint metrics, is the number that decides whether the estimator can be
# trusted to seat the plug: read position_error_mm.median against the 0.4 mm
# working target and lateral_error_mm.p95 against the 0.725 mm binding
# clearance.  Deliberately not --enforce: a first run that misses the gate is
# information, not a reason to discard the checkpoint.
.pixi/envs/default/bin/python validate_sc_plug_pose.py \
  --mode dataset \
  --weights "${training_root}/best_sc_plug_pose.pt" \
  --data "${dataset_yaml}" \
  --split test \
  --device 0 \
  --imgsz 960 \
  --report "${training_root}/reports/validate_sc_plug_pose_test.json"
