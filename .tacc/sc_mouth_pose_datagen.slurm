#!/usr/bin/env bash
# Collect simulator-TF labels for the physical SC front-mouth model.
#
# The legacy sc_port_pose_datagen job remains untouched: it writes the smaller
# virtual 8.8 x 6.0 mm target used by best_sc_pose.pt.  This job has a separate
# root and uses DataCollectorScMouthPoseGT's 22.407 x 8.10 mm + centre contract.
#SBATCH --job-name=sc-mouth-pose-data
#SBATCH --partition=rtx-small
#SBATCH --account=IRI26004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work2/11590/satya_a/stampede3/aic-sc-mouth-pose-datagen-20260727/logs/sc-mouth-pose-%j.out

# Do not add `set -u`: TACC's module and Pixi setup scripts read optional unset
# shell variables.  Preserve normal command and pipeline failure handling.
set -eo pipefail
module load tacc-apptainer/1.4.1

readonly job_root="${SC_MOUTH_JOB_ROOT:-/work2/11590/satya_a/stampede3/aic-sc-mouth-pose-datagen-20260727}"
readonly repo_root="${SC_MOUTH_REPO_ROOT:-/work2/11590/satya_a/stampede3/aic-sc-mouth-pose-src-20260727}"
readonly eval_image="${SC_MOUTH_EVAL_IMAGE:-/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260725/containers/aic_eval_pinned.sif}"
readonly mode="${SC_MODE:-smoke}"
readonly start_index="${SC_MOUTH_TRIAL_START:-1}"
readonly seed="${SC_MOUTH_SEED:-20260727}"
case "${mode}" in
  smoke)
    readonly trial_count="${SC_MOUTH_TRIAL_COUNT:-10}"
    # A fresh job-id directory prevents a failed smoke run from passing its
    # integrity checks against labels left by an earlier run.
    readonly dataset_root="${job_root}/smoke_dataset/${SLURM_JOB_ID}"
    readonly save_debug=1
    readonly minimum_images=30
    readonly viewpoints_per_trial="${SC_MOUTH_VIEWPOINTS:-18}"
    readonly far_view_probability="${SC_MOUTH_FAR_VIEW_PROB:-0.30}"
    ;;
  full)
    # Run four 25-trial batches (start 1, 26, 51, 76) into this same directory.
    # Restarts are materially faster than letting Gazebo degrade for hundreds
    # of trials, and global trial indices keep splits and file names stable.
    readonly trial_count="${SC_MOUTH_TRIAL_COUNT:-25}"
    readonly dataset_root="${job_root}/dataset"
    readonly save_debug=0
    readonly minimum_images=90
    # 14 genuinely different wrist viewpoints x 100 trials gives 4,200 native
    # camera images before visibility drops.  This is faster and more useful
    # than padding a smaller pose set with near-duplicate frames.
    readonly viewpoints_per_trial="${SC_MOUTH_VIEWPOINTS:-14}"
    # Batches 1-2 already provide broad coverage.  Bias the remaining samples
    # toward the deployment handoff range to reduce no-label camera frames.
    readonly far_view_probability="${SC_MOUTH_FAR_VIEW_PROB:-0.10}"
    ;;
  *)
    echo "SC_MODE must be smoke or full, got ${mode}" >&2
    exit 2
    ;;
esac
readonly trial_config="${job_root}/config/sc_mouth_pose_${mode}_${start_index}_${trial_count}.yaml"
readonly engine_home="$(mktemp -d "/tmp/aic-sc-mouth-${SLURM_JOB_ID}.XXXXXX")"
readonly sim_log="${job_root}/logs/eval-${mode}-${SLURM_JOB_ID}.log"
readonly engine_log="${job_root}/logs/engine-${mode}-${SLURM_JOB_ID}.log"
readonly sim_log_cap_bytes="${SC_MOUTH_SIM_LOG_CAP_BYTES:-$((512 * 1024 * 1024))}"
readonly collector_log="${job_root}/logs/collector-${mode}-${SLURM_JOB_ID}.log"
readonly stop_reason_file="${job_root}/STOP_REASON_${SLURM_JOB_ID}.txt"
readonly sim_crash_marker="${job_root}/SIMULATOR_CRASH_${SLURM_JOB_ID}.txt"
readonly collector_reaped_marker="${job_root}/COLLECTOR_REAPED_${SLURM_JOB_ID}.txt"
readonly usage_log="${job_root}/logs/usage-${mode}-${SLURM_JOB_ID}.csv"
readonly dataset_cap_kib="${SC_MOUTH_DATASET_CAP_KIB:-$((30 * 1024 * 1024))}"
readonly engine_home_cap_kib="${SC_MOUTH_ENGINE_HOME_CAP_KIB:-$((120 * 1024 * 1024))}"
readonly job_root_cap_kib="${SC_MOUTH_JOB_ROOT_CAP_KIB:-$((20 * 1024 * 1024))}"

mkdir -p "${job_root}/logs" "${job_root}/config" "${dataset_root}" "${engine_home}"

if [[ ! -s "${eval_image}" ]]; then
  echo "Missing pinned evaluation SIF: ${eval_image}" >&2
  exit 3
fi
if [[ ! -x "${repo_root}/.pixi/envs/default/bin/python" ]]; then
  echo "Missing staged Pixi environment: ${repo_root}/.pixi/envs/default/bin/python" >&2
  exit 4
fi

cd -- "${repo_root}"
.pixi/envs/default/bin/python generate_sc_plug_pose_trials.py \
  --template aic_engine/config/sc_data_collect.yaml \
  --out "${trial_config}" \
  --trials "${trial_count}" \
  --start-index "${start_index}" \
  --seed "${seed}" \
  --time-limit 180

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_GROUP
export APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_WANDB_DISABLED=true
export APPTAINERENV_WANDB_SILENT=true
export ROS_LOG_DIR="${engine_home}/.ros/log"
export APPTAINERENV_ROS_LOG_DIR="${ROS_LOG_DIR}"
export AIC_SC_MOUTH_POSE_OUTPUT_DIR="${dataset_root}"
export AIC_SC_MOUTH_POSE_SAVE_DEBUG="${save_debug}"
export AIC_SC_MOUTH_POSE_VIEWPOINTS="${viewpoints_per_trial}"
export AIC_SC_MOUTH_FAR_VIEW_PROB="${far_view_probability}"
export AIC_SC_MOUTH_TRIAL_START="${start_index}"
export AIC_SC_MOUTH_SEED="${seed}"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/127.0.0.1:7447"];transport/shared_memory/enabled=false'

sim_pid=''
collector_pid=''
engine_pid=''
watchdog_pid=''
sim_health_pid=''
reaper_pid=''
progress_pid=''
janitor_pid=''

terminate_process_group() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
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

cleanup() {
  for pid in "${watchdog_pid}" "${sim_health_pid}" "${reaper_pid}" "${progress_pid}" "${janitor_pid}" "${collector_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  terminate_process_group "${engine_pid}"
  terminate_process_group "${sim_pid}"
  rm -rf -- "${engine_home}"
}
trap cleanup EXIT INT TERM

stop_everything() {
  local reason="$1"
  printf '%s job=%s at=%s\n' "${reason}" "${SLURM_JOB_ID}" "$(date -Is)" >"${stop_reason_file}"
  echo "STOPPING: ${reason}"
  [[ -n "${collector_pid}" ]] && kill "${collector_pid}" 2>/dev/null || true
  [[ -n "${engine_pid}" ]] && kill -- "-${engine_pid}" 2>/dev/null || true
  [[ -n "${sim_pid}" ]] && kill -- "-${sim_pid}" 2>/dev/null || true
}

bag_janitor() {
  # Rosbags are diagnostic byproducts, not dataset inputs.  Leaving them to
  # grow has previously exhausted job storage during otherwise healthy runs.
  while true; do
    if [[ -d "${engine_home}/aic_results" ]]; then
      find "${engine_home}/aic_results" -mindepth 1 -maxdepth 1 -type d -mmin +1 -exec rm -rf -- {} +
    fi
    sleep 15
  done
}

storage_watchdog() {
  echo 'epoch,dataset_kib,engine_home_kib,job_root_kib' >"${usage_log}"
  while true; do
    local dataset_kib engine_kib root_kib
    dataset_kib=$(du -sk "${dataset_root}" 2>/dev/null | awk '{print $1}')
    engine_kib=$(du -sk "${engine_home}" 2>/dev/null | awk '{print $1}')
    root_kib=$(du -sk "${job_root}" 2>/dev/null | awk '{print $1}')
    : "${dataset_kib:=0}"
    : "${engine_kib:=0}"
    : "${root_kib:=0}"
    printf '%s,%s,%s,%s\n' "$(date +%s)" "${dataset_kib}" "${engine_kib}" "${root_kib}" >>"${usage_log}"
    if (( dataset_kib > dataset_cap_kib )); then
      stop_everything "dataset_cap_exceeded dataset_kib=${dataset_kib} cap=${dataset_cap_kib}"
      return 1
    fi
    if (( engine_kib > engine_home_cap_kib )); then
      stop_everything "engine_home_cap_exceeded engine_kib=${engine_kib} cap=${engine_home_cap_kib}"
      return 1
    fi
    if (( root_kib > job_root_cap_kib )); then
      stop_everything "job_root_cap_exceeded root_kib=${root_kib} cap=${job_root_cap_kib}"
      return 1
    fi
    sleep 30
  done
}

simulator_health_watchdog() {
  while kill -0 "${sim_pid}" 2>/dev/null; do
    if [[ -s "${sim_log}" ]] && grep -qE \
      'OGRE EXCEPTION|terminate called after throwing|component_container.*process has died|aic_engine.*process has died|Engine failed to initialize|Failed to find a valid clock' \
      "${sim_log}"; then
      printf 'simulator_crash job=%s detected_at=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" >"${sim_crash_marker}"
      stop_everything "simulator_crash"
      return 0
    fi
    sleep 5
  done
}

progress_watchdog() {
  local waited=0 completed=0 last_completed=0
  local startup_deadline="${SC_MOUTH_STARTUP_DEADLINE_S:-420}"
  local stall_deadline="${SC_MOUTH_STALL_DEADLINE_S:-420}"
  while true; do
    sleep 30
    completed=$(grep -c 'SC MOUTH GT collection done' "${collector_log}" 2>/dev/null || true)
    : "${completed:=0}"
    if (( completed > last_completed )); then
      last_completed="${completed}"
      waited=0
      continue
    fi
    waited=$((waited + 30))
    if (( last_completed == 0 && waited >= startup_deadline )); then
      stop_everything "no_trial_started_within_${startup_deadline}s"
      return 1
    fi
    if (( last_completed > 0 && waited >= stall_deadline )); then
      stop_everything "collection_stalled_after_${last_completed}_trials_for_${stall_deadline}s"
      return 1
    fi
  done
}

collection_reaper() {
  local grace="${SC_MOUTH_TEARDOWN_GRACE_S:-45}"
  while kill -0 "${collector_pid}" 2>/dev/null; do
    if grep -q 'on_shutdown' "${collector_log}" 2>/dev/null; then
      echo "reaper: collector reached on_shutdown; allowing ${grace}s to exit"
      sleep "${grace}"
      if kill -0 "${collector_pid}" 2>/dev/null; then
        printf 'collector_reaped job=%s at=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" >"${collector_reaped_marker}"
        kill -TERM "${collector_pid}" 2>/dev/null || true
        sleep 10
        kill -KILL "${collector_pid}" 2>/dev/null || true
      fi
      return 0
    fi
    sleep 5
  done
}

rm -f -- "${stop_reason_file}" "${sim_crash_marker}" "${collector_reaped_marker}"
# Importing ROS policy code can take longer than aic_engine's initial lifecycle
# probe on an uncached compute node.  Warm it before the simulator starts, then
# launch the participant as soon as the transport is available below.
set +u
source .pixi/envs/default/setup.bash
set -u
export PYTHONPATH="${repo_root}/aic_example_policies:${repo_root}/aic_model${PYTHONPATH:+:${PYTHONPATH}}"
.pixi/envs/default/bin/python -c 'import aic_example_policies.ros.DataCollectorScMouthPoseGT'
# Gazebo can flood the simulator stderr with repeated physics diagnostics.
# Keep draining it after the cap so it can never block the simulator, but retain
# enough context for debugging without consuming the work filesystem.
cap_stream() {
  awk -v max="$1" '
    { if (total < max) { print; total += length($0) + 1; fflush() } }
    END { if (total >= max) print "[simulator log capped at " max " bytes]" }
  '
}
# The stock in-container aic_engine has a fixed ten-second clock timeout and
# a fixed lifecycle-model timeout.  Cold Gazebo can exceed the former, while a
# cold Python/ROS policy can exceed the latter.  Start Gazebo alone, wait for
# its clock, make the policy ready, then start the engine as a separate process.
# This preserves both readiness conditions instead of racing them.
setsid apptainer exec --nv \
  --home "${engine_home}:/home1/11590/satya_a" \
  --bind "${job_root}:${job_root}" \
  --env WANDB_MODE=disabled,WANDB_DISABLED=true,WANDB_SILENT=true \
  "${eval_image}" /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=false \
    gazebo_gui:=false \
    launch_rviz:=false > >(cap_stream "${sim_log_cap_bytes}" >"${sim_log}") 2>&1 &
sim_pid=$!
simulator_health_watchdog &
sim_health_pid=$!
bag_janitor &
janitor_pid=$!
storage_watchdog &
watchdog_pid=$!

clock_ready_deadline="${SC_MOUTH_CLOCK_READY_DEADLINE_S:-120}"
clock_log="${job_root}/logs/clock-${mode}-${SLURM_JOB_ID}.log"
clock_waited=0
clock_ready=0
while (( clock_waited < clock_ready_deadline )); do
  if timeout 5 .pixi/envs/default/bin/ros2 topic echo /clock --once >"${clock_log}" 2>&1; then
    clock_ready=1
    break
  fi
  if ! kill -0 "${sim_pid}" 2>/dev/null; then
    stop_everything "simulator_exited_before_clock"
    exit 1
  fi
  sleep 1
  clock_waited=$((clock_waited + 1))
done
if (( clock_ready == 0 )); then
  stop_everything "clock_not_ready_within_${clock_ready_deadline}s"
  exit 1
fi

.pixi/envs/default/bin/ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.DataCollectorScMouthPoseGT >"${collector_log}" 2>&1 &
collector_pid=$!

policy_ready_deadline="${SC_MOUTH_POLICY_READY_DEADLINE_S:-360}"
policy_waited=0
while ! grep -q 'Using policy: DataCollectorScMouthPoseGT' "${collector_log}" 2>/dev/null; do
  if ! kill -0 "${collector_pid}" 2>/dev/null; then
    wait "${collector_pid}" || true
    echo "policy exited before readiness" >&2
    exit 1
  fi
  if (( policy_waited >= policy_ready_deadline )); then
    echo "policy did not become ready within ${policy_ready_deadline}s" >&2
    exit 1
  fi
  sleep 1
  policy_waited=$((policy_waited + 1))
done

setsid apptainer exec --nv \
  --home "${engine_home}:/home1/11590/satya_a" \
  --bind "${job_root}:${job_root}" \
  --env WANDB_MODE=disabled,WANDB_DISABLED=true,WANDB_SILENT=true \
  "${eval_image}" bash -lc '
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/127.0.0.1:7447\"];transport/shared_memory/enabled=false"
    exec /ws_aic/install/lib/aic_engine/aic_engine --ros-args \
      -p use_sim_time:=true \
      -p config_file_path:="$1" \
      -p model_discovery_timeout_seconds:=60 \
      -p model_configure_timeout_seconds:=60
  ' bash "${trial_config}" >"${engine_log}" 2>&1 &
engine_pid=$!

progress_watchdog &
progress_pid=$!
collection_reaper &
reaper_pid=$!
collector_status=0
wait "${collector_pid}" || collector_status=$?
collector_pid=''

for pid in "${progress_pid}" "${reaper_pid}" "${sim_health_pid}" "${janitor_pid}" "${watchdog_pid}"; do
  if [[ -n "${pid}" ]]; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
done
progress_pid='' reaper_pid='' sim_health_pid='' janitor_pid='' watchdog_pid=''
if [[ -e "${collector_reaped_marker}" ]] && grep -q 'SC MOUTH GT collection done' "${collector_log}"; then
  echo "collector was reaped after completed shutdown; treating collection as successful"
  collector_status=0
fi
terminate_process_group "${engine_pid}"
engine_pid=''
terminate_process_group "${sim_pid}"
sim_pid=''

image_count=$(find "${dataset_root}/images" -type f -name '*.png' 2>/dev/null | wc -l)
label_count=$(find "${dataset_root}/labels" -type f -name '*.txt' 2>/dev/null | wc -l)
metadata_count=$(find "${dataset_root}/metadata" -type f -name '*.json' 2>/dev/null | wc -l)
bad_labels=$(find "${dataset_root}/labels" -type f -name '*.txt' -exec awk 'NF != 20 {bad++} END {print bad+0}' {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
printf 'collection_complete job=%s mode=%s start=%s trials=%s images=%s labels=%s metadata=%s bad_labels=%s collector_status=%s\n' \
  "${SLURM_JOB_ID}" "${mode}" "${start_index}" "${trial_count}" "${image_count}" "${label_count}" "${metadata_count}" "${bad_labels}" "${collector_status}" \
  | tee "${job_root}/COLLECTION_${mode}_${SLURM_JOB_ID}.txt"

[[ ! -e "${stop_reason_file}" ]]
[[ ! -e "${sim_crash_marker}" ]]
[[ "${collector_status}" -eq 0 ]]
[[ "${image_count}" -ge "${minimum_images}" ]]
[[ "${image_count}" -eq "${label_count}" ]]
[[ "${bad_labels}" -eq 0 ]]
test -f "${dataset_root}/aic_sc_mouth_pose.yaml"
