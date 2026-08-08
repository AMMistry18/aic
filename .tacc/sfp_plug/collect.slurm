#!/usr/bin/env bash
#SBATCH --job-name=sfp-pose-data
#SBATCH --partition=rtx-small
#SBATCH --account=IRI26004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work2/11590/satya_a/stampede3/aic-sfp-pose-datagen-20260719/logs/sfp-pose-data-%j.out

# TACC's Apptainer and Pixi setup scripts read optional unset variables.
# Retain fail-fast and pipefail, but do not enable nounset around sourced setup.
set -eo pipefail
module load tacc-apptainer/1.4.1

readonly job_root=/work2/11590/satya_a/stampede3/aic-sfp-pose-datagen-20260719
readonly repo_root=/work2/11590/satya_a/stampede3/aic-gazebo-v1-9ae671e
readonly dataset_root="${job_root}/dataset"
readonly engine_home="$(mktemp -d "/tmp/aic-sfp-pose-${SLURM_JOB_ID}.XXXXXX")"
readonly trial_config="${job_root}/config/sfp_plug_pose_trials_resume_0325.yaml"
readonly eval_image="${job_root}/containers/aic_eval_latest.sif"

mkdir -p "${job_root}/logs" "${job_root}/containers" "${dataset_root}" "${engine_home}"

# Hard-disable all W&B initialization and syncing for this collection job.
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_GROUP
export APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_WANDB_DISABLED=true
export APPTAINERENV_WANDB_SILENT=true
export ROS_LOG_DIR="${engine_home}/.ros/log"
export APPTAINERENV_ROS_LOG_DIR="${ROS_LOG_DIR}"

export AIC_SFP_PLUG_POSE_OUTPUT_DIR="${dataset_root}"
export AIC_SFP_PLUG_POSE_VIEWPOINTS=1
export AIC_SFP_PLUG_POSE_FRAMES_PER_VIEW=3
export AIC_SFP_PLUG_POSE_SAVE_DEBUG=0
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/127.0.0.1:7447"];transport/shared_memory/enabled=false'

if [[ ! -s "${eval_image}" ]]; then
  tmp_image="${eval_image}.partial.${SLURM_JOB_ID}"
  rm -f -- "${tmp_image}"
  apptainer pull --disable-cache "${tmp_image}" docker://ghcr.io/intrinsic-dev/aic/aic_eval:latest
  mv -- "${tmp_image}" "${eval_image}"
fi

sim_pid=''
janitor_pid=''
cleanup() {
  if [[ -n "${janitor_pid}" ]] && kill -0 "${janitor_pid}" 2>/dev/null; then
    kill "${janitor_pid}" 2>/dev/null || true
    wait "${janitor_pid}" 2>/dev/null || true
  fi
  if [[ -n "${sim_pid}" ]] && kill -0 "${sim_pid}" 2>/dev/null; then
    kill "${sim_pid}" 2>/dev/null || true
    wait "${sim_pid}" 2>/dev/null || true
  fi
  rm -rf -- "${engine_home}"
}
trap cleanup EXIT INT TERM

bag_janitor() {
  while true; do
    if [[ -d "${engine_home}/aic_results" ]]; then
      find "${engine_home}/aic_results" \
        -mindepth 1 -maxdepth 1 -type d -mmin +2 \
        -exec rm -rf -- {} +
    fi
    sleep 30
  done
}

apptainer exec --nv \
  --home "${engine_home}:/home1/11590/satya_a" \
  --bind "${job_root}:${job_root}" \
  --env WANDB_MODE=disabled,WANDB_DISABLED=true,WANDB_SILENT=true \
  "${eval_image}" \
  /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    aic_engine_config_file:="${trial_config}" \
    gazebo_gui:=false \
    launch_rviz:=false \
  >"${job_root}/logs/eval-${SLURM_JOB_ID}.log" 2>&1 &
sim_pid=$!
bag_janitor &
janitor_pid=$!

# Start the policy well inside the engine's 30-second discovery window.
sleep 8
if ! kill -0 "${sim_pid}" 2>/dev/null; then
  wait "${sim_pid}"
fi

cd -- "${repo_root}"
source .pixi/envs/default/setup.bash
export PYTHONPATH="${repo_root}/aic_example_policies:${repo_root}/aic_model${PYTHONPATH:+:${PYTHONPATH}}"

.pixi/envs/default/bin/ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.DataCollectorSfpPlugPoseGT \
  >"${job_root}/logs/collector-${SLURM_JOB_ID}.log" 2>&1

if kill -0 "${sim_pid}" 2>/dev/null; then
  kill "${sim_pid}" 2>/dev/null || true
  wait "${sim_pid}" 2>/dev/null || true
fi
sim_pid=''

image_count=$(find "${dataset_root}/images" -type f -name '*.png' 2>/dev/null | wc -l)
label_count=$(find "${dataset_root}/labels" -type f -name '*.txt' 2>/dev/null | wc -l)
printf 'collection_complete job=%s images=%s labels=%s\n' \
  "${SLURM_JOB_ID}" "${image_count}" "${label_count}" \
  | tee "${job_root}/COLLECTION_COMPLETE.txt"
