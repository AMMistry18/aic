#!/usr/bin/env bash
#SBATCH --job-name=sc-plug-pose
#SBATCH --partition=rtx-small
#SBATCH --account=IRI26004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260719/logs/sc-plug-pose-%j.out

set -eo pipefail
module load tacc-apptainer/1.4.1

readonly job_root=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260719
readonly repo_root=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-src-20260719
readonly mode="${SC_MODE:-smoke}"
readonly trial_count="${SC_TRIAL_COUNT:-10}"
readonly start_index="${SC_TRIAL_START:-1}"
readonly seed="${SC_SEED:-20260719}"
readonly eval_image="${job_root}/containers/aic_eval_pinned.sif"
readonly engine_home="$(mktemp -d "/tmp/aic-sc-plug-${SLURM_JOB_ID}.XXXXXX")"

case "${mode}" in
  smoke)
    readonly dataset_root="${job_root}/smoke_dataset"
    readonly debug_overlays=1
    readonly storage_cap_kib=$((2 * 1024 * 1024))
    ;;
  full)
    readonly dataset_root="${job_root}/dataset"
    readonly debug_overlays=0
    # Expected output is about 5-7 GiB for 4,050 PNGs.  Stop well before a
    # runaway logger or cache can consume hundreds of GiB.
    readonly storage_cap_kib=$((15 * 1024 * 1024))
    ;;
  *)
    echo "SC_MODE must be smoke or full, got: ${mode}" >&2
    exit 2
    ;;
esac

readonly trial_config="${job_root}/config/sc_plug_pose_${mode}_${start_index}_${trial_count}.yaml"
mkdir -p "${job_root}/logs" "${job_root}/config" "${dataset_root}" "${engine_home}"

if [[ ! -s "${eval_image}" ]]; then
  echo "Missing pinned evaluation SIF: ${eval_image}" >&2
  echo "Refusing to pull a mutable latest image." >&2
  exit 3
fi
if [[ ! -x "${repo_root}/.pixi/envs/default/bin/python" ]]; then
  echo "Missing staged Pixi environment under ${repo_root}" >&2
  exit 4
fi

cd -- "${repo_root}"
.pixi/envs/default/bin/python generate_sc_plug_pose_trials.py \
  --template aic_engine/config/sc_data_collect.yaml \
  --out "${trial_config}" \
  --trials "${trial_count}" \
  --start-index "${start_index}" \
  --seed "${seed}" \
  --time-limit 20

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_GROUP
export APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_WANDB_DISABLED=true
export APPTAINERENV_WANDB_SILENT=true
export ROS_LOG_DIR="${engine_home}/.ros/log"
export APPTAINERENV_ROS_LOG_DIR="${ROS_LOG_DIR}"
export AIC_SC_PLUG_POSE_OUTPUT_DIR="${dataset_root}"
export AIC_SC_PLUG_POSE_VIEWPOINTS=1
export AIC_SC_PLUG_POSE_FRAMES_PER_VIEW=3
export AIC_SC_PLUG_POSE_SAVE_DEBUG="${debug_overlays}"
export AIC_SC_PLUG_POSE_TRIAL_START="${start_index}"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/127.0.0.1:7447"];transport/shared_memory/enabled=false'

sim_pid=''
collector_pid=''
janitor_pid=''
watchdog_pid=''
collector_watchdog_pid=''
sim_health_watchdog_pid=''
readonly sim_log="${job_root}/logs/eval-${mode}-${SLURM_JOB_ID}.log"
readonly sim_crash_marker="${job_root}/SIMULATOR_CRASH_${SLURM_JOB_ID}.txt"
terminate_simulator() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  # The simulator is launched under setsid below, so this targets only its
  # process group, not this Slurm shell or the collector.
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "${pid}" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}
wait_for_clean_simulator_shutdown() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  # The engine normally performs its own shutdown shortly after the collector
  # receives the final action result.  Interrupting it during that window can
  # leave the Apptainer parent stuck.  Reap a clean exit first; force only a
  # genuinely lingering simulator.
  for _ in $(seq 1 45); do
    local state
    state=$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')
    if [[ -z "${state}" || "${state}" == Z* ]]; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  terminate_simulator "${pid}"
}
cleanup() {
  for pid in "${collector_watchdog_pid}" "${sim_health_watchdog_pid}" \
    "${watchdog_pid}" "${janitor_pid}" "${collector_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  terminate_simulator "${sim_pid}"
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

storage_watchdog() {
  while true; do
    used_kib=$(du -sk "${dataset_root}" 2>/dev/null | awk '{print $1}')
    if (( used_kib > storage_cap_kib )); then
      printf 'storage_cap_exceeded used_kib=%s cap_kib=%s\n' \
        "${used_kib}" "${storage_cap_kib}" \
        >"${job_root}/STORAGE_CAP_EXCEEDED_${SLURM_JOB_ID}.txt"
      [[ -n "${collector_pid}" ]] && kill "${collector_pid}" 2>/dev/null || true
      [[ -n "${sim_pid}" ]] && kill "${sim_pid}" 2>/dev/null || true
      return 1
    fi
    sleep 30
  done
}

simulator_health_watchdog() {
  # Gazebo can leave its outer Apptainer process alive after a component
  # container dies.  Detect that fatal inner-process condition so the collector
  # does not wait forever for a goal that will never arrive.
  while kill -0 "${sim_pid}" 2>/dev/null; do
    if [[ -s "${sim_log}" ]] && grep -qE \
      'OGRE EXCEPTION|terminate called after throwing|component_container.*process has died' \
      "${sim_log}"; then
      printf 'simulator_crash job=%s detected_at=%s\n' \
        "${SLURM_JOB_ID}" "$(date -Is)" >"${sim_crash_marker}"
      terminate_simulator "${sim_pid}"
      return 0
    fi
    sleep 5
  done
}

collector_health_watchdog() {
  local watched_collector_pid="$1"
  while kill -0 "${watched_collector_pid}" 2>/dev/null; do
    if [[ -e "${sim_crash_marker}" ]]; then
      kill "${watched_collector_pid}" 2>/dev/null || true
      return 0
    fi
    sleep 2
  done
}

rm -f -- "${sim_crash_marker}"
setsid apptainer exec --nv \
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
  >"${sim_log}" 2>&1 &
sim_pid=$!
bag_janitor &
janitor_pid=$!
storage_watchdog &
watchdog_pid=$!
simulator_health_watchdog &
sim_health_watchdog_pid=$!

sleep 8
if ! kill -0 "${sim_pid}" 2>/dev/null; then
  wait "${sim_pid}"
fi

source .pixi/envs/default/setup.bash
export PYTHONPATH="${repo_root}/aic_example_policies:${repo_root}/aic_model${PYTHONPATH:+:${PYTHONPATH}}"
.pixi/envs/default/bin/ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.DataCollectorScPlugPoseGT \
  >"${job_root}/logs/collector-${mode}-${SLURM_JOB_ID}.log" 2>&1 &
collector_pid=$!
collector_health_watchdog "${collector_pid}" &
collector_watchdog_pid=$!
collector_status=0
wait "${collector_pid}" || collector_status=$?
collector_pid=''
if [[ -n "${collector_watchdog_pid}" ]]; then
  kill "${collector_watchdog_pid}" 2>/dev/null || true
  wait "${collector_watchdog_pid}" 2>/dev/null || true
  collector_watchdog_pid=''
fi

wait_for_clean_simulator_shutdown "${sim_pid}"
sim_pid=''

image_count=$(find "${dataset_root}/images" -type f -name '*.png' 2>/dev/null | wc -l)
label_count=$(find "${dataset_root}/labels" -type f -name '*.txt' 2>/dev/null | wc -l)
metadata_count=$(find "${dataset_root}/metadata" -type f -name '*.json' 2>/dev/null | wc -l)
bad_labels=$(find "${dataset_root}/labels" -type f -name '*.txt' -exec awk \
  'NF != 29 {bad++} END {print bad+0}' {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
used_kib=$(du -sk "${dataset_root}" | awk '{print $1}')

printf 'collection_complete job=%s mode=%s trials=%s images=%s labels=%s metadata=%s bad_labels=%s used_kib=%s collector_status=%s\n' \
  "${SLURM_JOB_ID}" "${mode}" "${trial_count}" "${image_count}" \
  "${label_count}" "${metadata_count}" "${bad_labels}" "${used_kib}" \
  "${collector_status}" \
  | tee "${job_root}/COLLECTION_${mode}_${SLURM_JOB_ID}.txt"

[[ "${collector_status}" -eq 0 ]]
[[ ! -e "${sim_crash_marker}" ]]
[[ "${image_count}" -gt 0 ]]
[[ "${image_count}" -eq "${label_count}" ]]
[[ "${bad_labels}" -eq 0 ]]

# TACC can cap a user at two queued jobs.  When requested for the final
# continuation batch, keep the already allocated GPU and invoke the strict
# training launcher only after collection has passed every integrity check.
if [[ "${SC_TRAIN_AFTER_COLLECTION:-0}" == "1" ]]; then
  [[ "${mode}" == "full" ]]
  mkdir -p "${job_root}/training/logs"
  exec "${repo_root}/train_sc_plug_pose.slurm" \
    >"${job_root}/training/logs/sc-plug-pose-train-inline-${SLURM_JOB_ID}.out" \
    2>&1
fi
