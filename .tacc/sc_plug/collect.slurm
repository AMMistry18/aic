#!/usr/bin/env bash
#SBATCH --job-name=sc-plug-pose
#SBATCH --partition=rtx-small
#SBATCH --account=IRI26004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260725/logs/sc-plug-pose-%j.out

set -eo pipefail
module load tacc-apptainer/1.4.1

readonly job_root=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260725
readonly repo_root=/work2/11590/satya_a/stampede3/aic-sc-plug-pose-src-20260725
readonly mode="${SC_MODE:-smoke}"
readonly trial_count="${SC_TRIAL_COUNT:-10}"
readonly start_index="${SC_TRIAL_START:-1}"
readonly seed="${SC_SEED:-20260725}"
readonly eval_image="${job_root}/containers/aic_eval_pinned.sif"
readonly engine_home="$(mktemp -d "/tmp/aic-sc-plug-${SLURM_JOB_ID}.XXXXXX")"

case "${mode}" in
  smoke)
    readonly dataset_root="${job_root}/smoke_dataset"
    readonly debug_overlays=1
    readonly storage_cap_kib=$((4 * 1024 * 1024))
    ;;
  full)
    readonly dataset_root="${job_root}/dataset"
    readonly debug_overlays=0
    # Expected output is about 5-7 GiB for 4,050 PNGs.  The cap is a runaway
    # guard, not a budget: /work2 has ~680 GiB of quota headroom, so a tight cap
    # buys nothing and only risks killing a healthy run near the end.
    readonly storage_cap_kib=$((40 * 1024 * 1024))
    ;;
  *)
    echo "SC_MODE must be smoke or full, got: ${mode}" >&2
    exit 2
    ;;
esac

# --- space hardening -------------------------------------------------------
# Three separate budgets.  The dataset is the thing we want; the other two are
# byproducts that have no business growing without bound.  engine_home is on
# node-local xfs (3.4 TiB), the rest is Lustre under a 1000 GiB user quota.
readonly job_root_cap_kib="${SC_JOB_ROOT_CAP_KIB:-$((80 * 1024 * 1024))}"
readonly engine_home_cap_kib="${SC_ENGINE_HOME_CAP_KIB:-$((120 * 1024 * 1024))}"
readonly sim_log_cap_bytes="${SC_SIM_LOG_CAP_BYTES:-$((2 * 1024 * 1024 * 1024))}"
# Stop cleanly while there is still room to flush and finalise, rather than
# being killed mid-write with a half-written label.
readonly work_free_floor_kib="${SC_WORK_FREE_FLOOR_KIB:-$((20 * 1024 * 1024))}"

readonly trial_config="${job_root}/config/sc_plug_pose_${mode}_${start_index}_${trial_count}.yaml"
mkdir -p "${job_root}/logs" "${job_root}/config" "${dataset_root}" "${engine_home}"

readonly usage_log="${job_root}/logs/usage-${mode}-${SLURM_JOB_ID}.csv"
readonly stop_reason_file="${job_root}/STOP_REASON_${SLURM_JOB_ID}.txt"

work_free_kib() {
  # Soft quota minus usage.  Falls back to "effectively unlimited" if the
  # filesystem does not report a quota, so this can never spuriously abort.
  lfs quota -u "${USER}" /work2 2>/dev/null | awk '
    $1 ~ /work2/ {
      used=$2; soft=$3
      gsub(/[^0-9]/,"",used); gsub(/[^0-9]/,"",soft)
      if (soft+0 > 0) print soft-used; else print 999999999
      exit
    }'
}

dir_kib() { du -sk "$1" 2>/dev/null | awk '{print $1}'; }

# --- preflight -------------------------------------------------------------
if [[ ! -s "${eval_image}" ]]; then
  echo "Missing pinned evaluation SIF: ${eval_image}" >&2
  echo "Refusing to pull a mutable latest image." >&2
  exit 3
fi
if [[ ! -x "${repo_root}/.pixi/envs/default/bin/python" ]]; then
  echo "Missing staged Pixi environment under ${repo_root}" >&2
  exit 4
fi

preflight_free_kib="$(work_free_kib)"
: "${preflight_free_kib:=999999999}"
printf 'preflight work2_free_kib=%s (%.1f GiB) floor_kib=%s\n' \
  "${preflight_free_kib}" "$((preflight_free_kib / 1024))e-3" "${work_free_floor_kib}" 2>/dev/null || true
echo "preflight: /work2 free ${preflight_free_kib} KiB, need > ${work_free_floor_kib} KiB"
if (( preflight_free_kib < work_free_floor_kib )); then
  echo "Refusing to start: only ${preflight_free_kib} KiB of /work2 quota free." >&2
  echo "Free space or raise SC_WORK_FREE_FLOOR_KIB before resubmitting." >&2
  exit 5
fi
echo "preflight: node-local /tmp"
df -h /tmp | tail -1

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
readonly collector_reaped_marker="${job_root}/COLLECTOR_REAPED_${SLURM_JOB_ID}.txt"
progress_pid=''
readonly collector_log="${job_root}/logs/collector-${mode}-${SLURM_JOB_ID}.log"
reaper_pid=''

# Bound the simulator log without ever blocking the writer.  Gazebo is capable
# of emitting tens of GiB over a 12 hour run; an unbounded redirect onto Lustre
# is the most likely way a healthy collection dies of "no space".  awk keeps
# draining stdin after the cap so the simulator never stalls on a full pipe.
cap_stream() {
  awk -v max="$1" '
    { if (tot < max) { print; tot += length($0) + 1; fflush() } }
    END { if (tot >= max) print "[log capped at " max " bytes]" }
  '
}

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
  for pid in "${collector_watchdog_pid}" "${reaper_pid}" "${progress_pid}" "${sim_health_watchdog_pid}" \
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
  # Rosbags are pure byproduct here; nothing downstream reads them.  Sweep more
  # often and with a shorter grace than the original 30 s / 2 min so a burst
  # cannot outrun the janitor.
  while true; do
    if [[ -d "${engine_home}/aic_results" ]]; then
      find "${engine_home}/aic_results" \
        -mindepth 1 -maxdepth 1 -type d -mmin +1 \
        -exec rm -rf -- {} +
    fi
    sleep 15
  done
}

stop_everything() {
  local reason="$1"
  printf '%s job=%s at=%s\n' "${reason}" "${SLURM_JOB_ID}" "$(date -Is)" \
    >"${stop_reason_file}"
  echo "STOPPING: ${reason}"
  [[ -n "${collector_pid}" ]] && kill "${collector_pid}" 2>/dev/null || true
  [[ -n "${sim_pid}" ]] && kill "${sim_pid}" 2>/dev/null || true
}

storage_watchdog() {
  # Watches every place that can grow, not just the dataset, and records a
  # trend so a stop is diagnosable after the fact instead of a guess.
  echo "epoch,dataset_kib,job_root_kib,engine_home_kib,work_free_kib" >"${usage_log}"
  while true; do
    local ds jr eh free
    ds=$(dir_kib "${dataset_root}"); : "${ds:=0}"
    jr=$(dir_kib "${job_root}");     : "${jr:=0}"
    eh=$(dir_kib "${engine_home}");  : "${eh:=0}"
    free=$(work_free_kib);           : "${free:=999999999}"
    printf '%s,%s,%s,%s,%s\n' "$(date +%s)" "${ds}" "${jr}" "${eh}" "${free}" \
      >>"${usage_log}"

    if (( ds > storage_cap_kib )); then
      printf 'storage_cap_exceeded used_kib=%s cap_kib=%s\n' \
        "${ds}" "${storage_cap_kib}" \
        >"${job_root}/STORAGE_CAP_EXCEEDED_${SLURM_JOB_ID}.txt"
      stop_everything "dataset_cap_exceeded dataset_kib=${ds} cap=${storage_cap_kib}"
      return 1
    fi
    if (( jr > job_root_cap_kib )); then
      stop_everything "job_root_cap_exceeded job_root_kib=${jr} cap=${job_root_cap_kib}"
      return 1
    fi
    if (( eh > engine_home_cap_kib )); then
      stop_everything "engine_home_cap_exceeded engine_home_kib=${eh} cap=${engine_home_cap_kib}"
      return 1
    fi
    if (( free < work_free_floor_kib )); then
      stop_everything "work2_quota_floor_hit free_kib=${free} floor=${work_free_floor_kib}"
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
    # aic_engine gives up if /clock is not published within 10 s.  When Gazebo's
    # render fallback is slow (libEGL dri2 failure -> headless) it loses that
    # race, dies, and then nothing drives any trial: the collector waits forever
    # and the job burns its full walltime.  Those patterns must be fatal too.
    if [[ -s "${sim_log}" ]] && grep -qE \
      'OGRE EXCEPTION|terminate called after throwing|component_container.*process has died|aic_engine.*process has died|Engine failed to initialize|Failed to find a valid clock' \
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

progress_watchdog() {
  # Generic liveness guard.  Whatever the cause -- engine init race, a wedged
  # Gazebo, a stuck goal -- if no trial completes for this long, the run is not
  # going to recover, and waiting out a 12 hour walltime learns nothing.  Fail
  # in minutes so the job can simply be resubmitted.
  local log="$1"
  local startup_deadline="${SC_STARTUP_DEADLINE_S:-420}"
  local stall_deadline="${SC_STALL_DEADLINE_S:-420}"
  local waited=0 last_count=0 count
  while true; do
    sleep 30
    count=$(grep -c 'collection complete: trial=' "${log}" 2>/dev/null || echo 0)
    if (( count > last_count )); then
      last_count="${count}"
      waited=0
      continue
    fi
    waited=$((waited + 30))
    if (( last_count == 0 )) && (( waited >= startup_deadline )); then
      stop_everything "no_trial_started_within_${startup_deadline}s (check ${sim_log} for engine init failure)"
      return 1
    fi
    if (( last_count > 0 )) && (( waited >= stall_deadline )); then
      stop_everything "collection_stalled_after_${last_count}_trials_for_${stall_deadline}s"
      return 1
    fi
  done
}

collection_reaper() {
  # The collector finishes every trial, writes the dataset yaml and runs its
  # full lifecycle down to on_shutdown -- and then does not exit.  Left alone
  # the script blocks in `wait` on a process that has nothing left to do, and
  # the job burns walltime until Slurm times it out.  That is what happened to
  # jobs 3329045 (cancelled at 8h22) and 3329874 (TIMEOUT at 12h00): the data
  # was already on disk.  Detect the completed shutdown and force the exit.
  local watched_collector_pid="$1"
  local log="$2"
  local grace="${SC_TEARDOWN_GRACE_S:-45}"
  while kill -0 "${watched_collector_pid}" 2>/dev/null; do
    if grep -q 'on_shutdown' "${log}" 2>/dev/null; then
      echo "reaper: collector reached on_shutdown; allowing ${grace}s to exit cleanly"
      local waited=0
      while (( waited < grace )); do
        kill -0 "${watched_collector_pid}" 2>/dev/null || return 0
        sleep 5
        waited=$((waited + 5))
      done
      if kill -0 "${watched_collector_pid}" 2>/dev/null; then
        echo "reaper: collector still alive after ${grace}s, forcing termination"
        printf 'collector_reaped job=%s at=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" \
          >"${collector_reaped_marker}"
        kill -TERM "${watched_collector_pid}" 2>/dev/null || true
        sleep 10
        kill -KILL "${watched_collector_pid}" 2>/dev/null || true
      fi
      return 0
    fi
    sleep 5
  done
}

rm -f -- "${sim_crash_marker}" "${stop_reason_file}"
# Process substitution, not a pipe: $! must remain the setsid leader so that
# terminate_simulator can signal the whole process group.
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
  > >(cap_stream "${sim_log_cap_bytes}" >"${sim_log}") 2>&1 &
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
  >"${collector_log}" 2>&1 &
collector_pid=$!
collector_health_watchdog "${collector_pid}" &
collector_watchdog_pid=$!
collection_reaper "${collector_pid}" "${collector_log}" &
reaper_pid=$!
progress_watchdog "${collector_log}" &
progress_pid=$!
collector_status=0
wait "${collector_pid}" || collector_status=$?
collector_pid=''
for _pid_var in collector_watchdog_pid reaper_pid progress_pid; do
  _pid="${!_pid_var}"
  if [[ -n "${_pid}" ]]; then
    kill "${_pid}" 2>/dev/null || true
    wait "${_pid}" 2>/dev/null || true
    printf -v "${_pid_var}" '%s' ''
  fi
done
# A reaped collector exits non-zero by construction.  That is a successful run
# whose teardown had to be forced, not a collection failure -- the integrity
# checks below are what actually decide whether the data is good.
if [[ -e "${collector_reaped_marker}" ]] \
  && grep -q 'SC plug GT collection complete' "${collector_log}" 2>/dev/null; then
  echo "collector was reaped after a completed shutdown; treating as success"
  collector_status=0
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

echo "--- storage trend (last 5 samples) ---"
tail -5 "${usage_log}" 2>/dev/null || true
if [[ -e "${stop_reason_file}" ]]; then
  echo "--- stopped early ---"
  cat "${stop_reason_file}"
fi

[[ ! -e "${stop_reason_file}" ]]
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
  exec "${repo_root}/.tacc/sc_plug/train.slurm" \
    >"${job_root}/training/logs/sc-plug-pose-train-inline-${SLURM_JOB_ID}.out" \
    2>&1
fi
