#!/usr/bin/env bash
# Train only after all three physical-mouth collection batches have completed.
#SBATCH --job-name=sc-mouth-pose-train
#SBATCH --partition=rtx-small
#SBATCH --account=IRI26004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work2/11590/satya_a/stampede3/aic-sc-mouth-pose-datagen-20260727/training/logs/sc-mouth-pose-train-%j.out

set -eo pipefail

readonly job_root="${SC_MOUTH_JOB_ROOT:-/work2/11590/satya_a/stampede3/aic-sc-mouth-pose-datagen-20260727}"
readonly repo_root="${SC_MOUTH_REPO_ROOT:-/work2/11590/satya_a/stampede3/aic-sc-mouth-pose-src-20260727}"
readonly dataset_root="${job_root}/dataset"
readonly dataset_yaml="${dataset_root}/aic_sc_mouth_pose.yaml"
readonly training_root="${job_root}/training"
readonly min_total_images="${SC_MOUTH_MIN_TOTAL_IMAGES:-3000}"
readonly min_split_images="${SC_MOUTH_MIN_SPLIT_IMAGES:-300}"
readonly dataloader_workers="${SC_MOUTH_TRAIN_WORKERS:-0}"

mkdir -p \
  "${training_root}/logs" \
  "${training_root}/runs" \
  "${training_root}/reports" \
  "${training_root}/cache/ultralytics" \
  "${training_root}/cache/torch"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_GROUP
export YOLO_CONFIG_DIR="${training_root}/cache/ultralytics"
export TORCH_HOME="${training_root}/cache/torch"
# Dataloader AF_UNIX paths under work2 exceed Linux's socket-path limit.
export TMPDIR="/tmp/aic-sc-mouth-train-${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
trap 'rm -rf -- "${TMPDIR}"' EXIT INT TERM

test -f "${dataset_yaml}"
cd -- "${repo_root}"
set +u
source .pixi/envs/default/setup.bash
set -u

.pixi/envs/default/bin/python - "${dataset_root}" "${min_total_images}" "${min_split_images}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
minimum_total = int(sys.argv[2])
minimum_split = int(sys.argv[3])
yaml_path = root / "aic_sc_mouth_pose.yaml"
if "kpt_shape: [5, 3]" not in yaml_path.read_text(encoding="utf-8"):
    raise SystemExit(f"{yaml_path} is not the five-keypoint physical-mouth dataset")
counts = {}
for split in ("train", "val", "test"):
    images = sorted((root / "images" / split).glob("*.png"))
    labels = root / "labels" / split
    if len(images) < minimum_split:
        raise SystemExit(f"{split} has {len(images)} images; need {minimum_split}")
    for image in images:
        label = labels / f"{image.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"missing label for {image}")
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if len(line.split()) != 20:
                raise SystemExit(f"{label}:{line_number} is not a 20-token five-keypoint label")
    counts[split] = len(images)
if sum(counts.values()) < minimum_total:
    raise SystemExit(f"dataset too small: {counts}; need {minimum_total} images")
print(f"SC physical-mouth dataset gate passed: {counts}")
PY

.pixi/envs/default/bin/python tools/perception/sc_mouth/train.py \
  --data "${dataset_yaml}" \
  --model yolo11s-pose.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 8 \
  --workers "${dataloader_workers}" \
  --device 0 \
  --project "${training_root}/runs" \
  --name sc_mouth_pose_yolo11s_960 \
  --final-weights "${training_root}/best_sc_mouth_pose.pt" \
  --metrics-out "${training_root}/reports/train_sc_mouth_pose.json" \
  --seed 20260727

# Record geometric held-out performance.  This is deliberately informational,
# not a deployment switch: the live controller and LOCAL_SC_PORT_KPS remain on
# the legacy model until a separate atomic runtime migration is reviewed.
.pixi/envs/default/bin/python tools/perception/sc_mouth/validate.py \
  --weights "${training_root}/best_sc_mouth_pose.pt" \
  --data "${dataset_yaml}" \
  --split test \
  --device 0 \
  --imgsz 960 \
  --report "${training_root}/reports/validate_sc_mouth_pose_test.json"
