#!/usr/bin/env bash
# Collect SC plug-pose ground-truth data locally, on one workstation GPU.
#
# This is the local counterpart to .tacc/sc_plug/collect.slurm.  It runs
# the evaluation simulator in a plain docker container and the
# DataCollectorScPlugPoseGT policy in the repo pixi env on the host.
#
#   ./scripts/sc_plug_pose_collect_local.sh smoke        # 5 trials, overlays on
#   ./scripts/sc_plug_pose_collect_local.sh full         # 450 trials
#   SC_TRIAL_START=451 ./scripts/sc_plug_pose_collect_local.sh full  # resume
#
# WHY PLAIN DOCKER AND NOT THE aic_eval DISTROBOX
#
#   1. Rendering.  Ogre2 enumerates EGL devices and takes the first that
#      initializes.  In distrobox that is the DRM node /dev/dri/card1, which
#      this host's user cannot open (its ACL grants only gdm-greeter), so the
#      render window never comes up.  With --gpus all, docker injects
#      /dev/nvidia* and the NVIDIA EGL device without /dev/dri, so Ogre2 gets
#      the GPU directly.
#
#   2. Networking.  gz-transport discovers peers over UDP multicast and this
#      host's loopback has no MULTICAST flag, so --network host breaks
#      discovery between gz_server and ros_gz_sim ("Could not initialize
#      Gazebo connection").  The docker bridge does carry multicast, so the
#      sim runs on bridge with only the Zenoh router port published.
#
#   3. User.  Running the container with --user <hostuid> leaves a uid absent
#      from /etc/passwd, which also silently breaks gz-transport discovery.
#      The container runs as root and writes nothing outside itself except the
#      read-only trial-config bind.
#
# PREREQUISITE: a working NVIDIA driver.  If the loaded kernel module and the
# installed userspace disagree, nvidia EGL enumerates zero devices, Ogre2 falls
# back to llvmpipe and the sim runs at a real-time factor near 0.0003, which is
# useless for collection.  The preflight below refuses to start in that state.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODE="${1:-smoke}"
readonly RUN_ROOT="${SC_RUN_ROOT:-${HOME}/aic_perception_data}"
readonly SIM_IMAGE="${SC_SIM_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}"
readonly SIM_CPUS="${SC_SIM_CPUS:-12}"
readonly SIM_MEM="${SC_SIM_MEM:-12g}"
readonly ZENOH_PORT="${SC_ZENOH_PORT:-7447}"
readonly CONTAINER_NAME="${SC_CONTAINER_NAME:-aic_sim_sc_plug_pose}"
readonly TRIAL_START="${SC_TRIAL_START:-1}"
readonly SEED="${SC_SEED:-20260725}"

case "${MODE}" in
  smoke)
    DATASET_ROOT="${RUN_ROOT}/sc_plug_pose_smoke"
    TRIALS="${SC_TRIAL_COUNT:-5}"
    SAVE_DEBUG=1
    # A smoke run should never grow past a few hundred megabytes.
    STORAGE_CAP_KIB=$((1024 * 1024))
    ;;
  full)
    DATASET_ROOT="${RUN_ROOT}/sc_plug_pose"
    TRIALS="${SC_TRIAL_COUNT:-450}"
    SAVE_DEBUG=0
    # 450 trials x 3 cameras x 3 frames = 4050 PNGs, about 6 GiB.  Stop well
    # before a runaway can fill a disk that is already near capacity.
    STORAGE_CAP_KIB=$((12 * 1024 * 1024))
    ;;
  *)
    echo "usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac
readonly DATASET_ROOT TRIALS SAVE_DEBUG STORAGE_CAP_KIB

readonly WORK_DIR="${DATASET_ROOT}/.run"
readonly LOG_DIR="${WORK_DIR}/logs"
readonly CONFIG="${WORK_DIR}/sc_plug_pose_${MODE}_${TRIAL_START}_${TRIALS}.yaml"
readonly SIM_LOG="${LOG_DIR}/sim-${MODE}-${TRIAL_START}.log"
readonly COLLECTOR_LOG="${LOG_DIR}/collector-${MODE}-${TRIAL_START}.log"
mkdir -p "${DATASET_ROOT}" "${LOG_DIR}"

# ---------------------------------------------------------------- preflight
echo "== preflight =="
free_kib=$(df -Pk "${DATASET_ROOT}" | awk 'NR==2{print $4}')
required_kib=$((STORAGE_CAP_KIB + 2 * 1024 * 1024))
if (( free_kib < required_kib )); then
  echo "FAIL: only $((free_kib / 1024 / 1024)) GiB free at ${DATASET_ROOT};" \
       "need about $((required_kib / 1024 / 1024)) GiB for mode=${MODE}" >&2
  exit 3
fi
echo "  disk free: $((free_kib / 1024 / 1024)) GiB"

if ! docker run --rm --gpus all "${SIM_IMAGE}" \
     --entrypoint true >/dev/null 2>&1; then
  : # --entrypoint after the image is ignored; the real check is below.
fi
gpu_probe=$(docker run --rm --gpus all --entrypoint bash "${SIM_IMAGE}" -c '
  python3 - <<PY 2>/dev/null
import ctypes
egl = ctypes.CDLL("libEGL.so.1")
egl.eglGetProcAddress.restype = ctypes.c_void_p
fn = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
if not fn:
    print("0"); raise SystemExit
q = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
                     ctypes.POINTER(ctypes.c_void_p),
                     ctypes.POINTER(ctypes.c_int))(fn)
n = ctypes.c_int(0); devs = (ctypes.c_void_p * 16)()
q(16, devs, ctypes.byref(n))
print(n.value)
PY
' 2>/dev/null | tr -d '[:space:]')
if [[ "${gpu_probe}" != "1" && "${gpu_probe}" != "2" ]]; then
  echo "FAIL: EGL enumerated '${gpu_probe:-none}' devices in the sim container." >&2
  echo "      The NVIDIA driver is probably still mismatched (check that the" >&2
  echo "      loaded kernel module version equals the installed userspace)." >&2
  echo "      Collecting under llvmpipe is not viable; fix the driver first." >&2
  exit 4
fi
echo "  EGL devices visible to the sim container: ${gpu_probe}"

# ------------------------------------------------------------ trial config
echo "== generating ${TRIALS} trials from index ${TRIAL_START} =="
"${REPO_ROOT}/.pixi/envs/default/bin/python" \
  "${REPO_ROOT}/generate_sc_plug_pose_trials.py" \
  --template "${REPO_ROOT}/aic_engine/config/sc_data_collect.yaml" \
  --out "${CONFIG}" \
  --trials "${TRIALS}" \
  --start-index "${TRIAL_START}" \
  --seed "${SEED}" \
  --time-limit 20

# ------------------------------------------------------------------ run it
sim_started=0
collector_pid=''
watchdog_pid=''
janitor_pid=''

cleanup() {
  for pid in "${watchdog_pid}" "${janitor_pid}" "${collector_pid}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  if (( sim_started )); then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
echo "== starting simulator =="
docker run --rm --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${ZENOH_PORT}:7447" \
  --cpus "${SIM_CPUS}" --memory "${SIM_MEM}" \
  -v "${WORK_DIR}:${WORK_DIR}:ro" \
  -e WANDB_MODE=disabled -e WANDB_DISABLED=true -e WANDB_SILENT=true \
  "${SIM_IMAGE}" \
    ground_truth:=true \
    start_aic_engine:=true \
    aic_engine_config_file:="${CONFIG}" \
    gazebo_gui:=false \
    launch_rviz:=false \
  >"${SIM_LOG}" 2>&1 &
sim_started=1

# The engine waits about 30 s for aic_model, so the collector must be up well
# before that; wait for the controllers to activate first.
for _ in $(seq 1 90); do
  grep -q "Successfully switched controllers" "${SIM_LOG}" 2>/dev/null && break
  grep -qE "Unable to create the rendering window|process has died" "${SIM_LOG}" 2>/dev/null && {
    echo "FAIL: simulator died during startup; see ${SIM_LOG}" >&2; exit 5; }
  sleep 2
done

storage_watchdog() {
  while true; do
    used_kib=$(du -sk "${DATASET_ROOT}" 2>/dev/null | awk '{print $1}')
    if (( ${used_kib:-0} > STORAGE_CAP_KIB )); then
      echo "storage_cap_exceeded used_kib=${used_kib} cap_kib=${STORAGE_CAP_KIB}" \
        | tee "${DATASET_ROOT}/STORAGE_CAP_EXCEEDED.txt" >&2
      [[ -n "${collector_pid}" ]] && kill "${collector_pid}" 2>/dev/null || true
      docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
      return 1
    fi
    sleep 30
  done
}
bag_janitor() {
  # The engine writes a rosbag per trial; none of it is needed for the dataset
  # and on a 96%-full disk it is the fastest way to run out of space.
  while true; do
    for root in "${HOME}/aic_results" "${AIC_RESULTS_DIR:-}"; do
      [[ -n "${root}" && -d "${root}" ]] || continue
      find "${root}" -mindepth 1 -maxdepth 1 -type d -mmin +2 -exec rm -rf -- {} + 2>/dev/null || true
    done
    sleep 30
  done
}
storage_watchdog & watchdog_pid=$!
bag_janitor & janitor_pid=$!

echo "== starting collector =="
(
  cd "${REPO_ROOT}"
  export AIC_SC_PLUG_POSE_OUTPUT_DIR="${DATASET_ROOT}"
  export AIC_SC_PLUG_POSE_VIEWPOINTS="${SC_VIEWPOINTS:-1}"
  export AIC_SC_PLUG_POSE_FRAMES_PER_VIEW="${SC_FRAMES_PER_VIEW:-3}"
  export AIC_SC_PLUG_POSE_SAVE_DEBUG="${SAVE_DEBUG}"
  export AIC_SC_PLUG_POSE_TRIAL_START="${TRIAL_START}"
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
  export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/127.0.0.1:${ZENOH_PORT}\"];transport/shared_memory/enabled=false"
  export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
  # Pixi's generated ROS setup scripts read several optional variables
  # directly, so relax nounset across the source as a login shell would.
  set +u
  source .pixi/envs/default/setup.bash
  set -u
  export PYTHONPATH="${REPO_ROOT}/aic_example_policies:${REPO_ROOT}/aic_model${PYTHONPATH:+:${PYTHONPATH}}"
  exec .pixi/envs/default/bin/ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.DataCollectorScPlugPoseGT
) >"${COLLECTOR_LOG}" 2>&1 &
collector_pid=$!

collector_status=0
wait "${collector_pid}" || collector_status=$?
collector_pid=''

# ------------------------------------------------------------- integrity
images=$(find "${DATASET_ROOT}/images" -name '*.png' 2>/dev/null | wc -l)
labels=$(find "${DATASET_ROOT}/labels" -name '*.txt' 2>/dev/null | wc -l)
metadata=$(find "${DATASET_ROOT}/metadata" -name '*.json' 2>/dev/null | wc -l)
bad=$(find "${DATASET_ROOT}/labels" -name '*.txt' -exec awk 'NF!=29{c++} END{print c+0}' {} + 2>/dev/null \
      | awk '{s+=$1} END{print s+0}')
used=$(du -sh "${DATASET_ROOT}" 2>/dev/null | cut -f1)

printf 'collection_complete mode=%s trials=%s images=%s labels=%s metadata=%s bad_labels=%s used=%s status=%s\n' \
  "${MODE}" "${TRIALS}" "${images}" "${labels}" "${metadata}" "${bad}" "${used}" "${collector_status}" \
  | tee "${DATASET_ROOT}/COLLECTION_${MODE}_${TRIAL_START}.txt"

[[ "${images}" -gt 0 ]] || { echo "FAIL: no images collected" >&2; exit 6; }
[[ "${images}" -eq "${labels}" ]] || { echo "FAIL: image/label count mismatch" >&2; exit 7; }
[[ "${bad}" -eq 0 ]] || { echo "FAIL: ${bad} labels are not 29-token poses" >&2; exit 8; }
echo "OK"
