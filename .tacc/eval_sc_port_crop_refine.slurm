#!/usr/bin/env bash
#SBATCH --job-name=sc-port-crop-eval
#SBATCH --partition=rtx-small
#SBATCH --account=IRI26004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work2/11590/satya_a/stampede3/aic-sc-port-pose-datagen-20260726/logs/sc-port-crop-eval-%j.out

set -euo pipefail

readonly job_root=/work2/11590/satya_a/stampede3/aic-sc-port-pose-datagen-20260726
readonly repo_root=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-src-20260725
readonly eval_root="${job_root}/eval_src"
readonly dataset_root="${job_root}/smoke_dataset"
readonly weights="${eval_root}/aic_example_policies/aic_example_policies/ros/weights/best_sc_pose.pt"
readonly report="${job_root}/reports/sc_port_crop_sweep_${SLURM_JOB_ID}.json"
readonly crop_pad_scales="${SC_CROP_PAD_SCALES:-1.5,2,3,4,6,8,12,16,24,32}"

mkdir -p "${job_root}/reports" "${job_root}/cache/ultralytics" "${job_root}/cache/torch"
export YOLO_CONFIG_DIR="${job_root}/cache/ultralytics"
export TORCH_HOME="${job_root}/cache/torch"
export PYTHONPATH="${eval_root}/aic_example_policies"

test -s "${weights}"
test -s "${dataset_root}/aic_sc_pose_raw.yaml"
image_count=$(find "${dataset_root}/images/val" -type f -name '*.png' | wc -l)
label_count=$(find "${dataset_root}/labels/val" -type f -name '*.txt' | wc -l)
bad_labels=$(find "${dataset_root}/labels/val" -type f -name '*.txt' -exec awk \
  'NF != 17 {bad++} END {print bad+0}' {} + | awk '{sum+=$1} END {print sum+0}')
echo "preflight images=${image_count} labels=${label_count} bad_labels=${bad_labels}"
[[ "${image_count}" -ge 30 ]]
[[ "${image_count}" -eq "${label_count}" ]]
[[ "${bad_labels}" -eq 0 ]]

cd "${eval_root}"
set +u
source "${repo_root}/.pixi/envs/default/setup.bash"
set -u
"${repo_root}/.pixi/envs/default/bin/python" eval_sc_pose_model.py \
  --weights "${weights}" \
  --data "${dataset_root}/aic_sc_pose_raw.yaml" \
  --split val \
  --conf 0.15 \
  --crop-pad-scales "${crop_pad_scales}" \
  --warmup 10 \
  --report "${report}"

echo "report=${report}"
