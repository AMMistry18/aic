#!/usr/bin/env bash
# ============================================================================
# AIC MuJoCo pipeline + MJX setup.
#
# Runs end-to-end from a clean machine. Each step is idempotent: if it's
# already been done, the step exits 0 without re-doing the work.
#
# Usage:
#   ./aic_utils/aic_mujoco/setup_pipeline.sh          # full pipeline
#   ./aic_utils/aic_mujoco/setup_pipeline.sh deps     # only host deps (apt + CUDA)
#   ./aic_utils/aic_mujoco/setup_pipeline.sh ros      # only ROS 2 install
#   ./aic_utils/aic_mujoco/setup_pipeline.sh workspace # only colcon build of gz-mujoco, mujoco_vendor, mujoco_ros2_control
#   ./aic_utils/aic_mujoco/setup_pipeline.sh sdf      # only export /tmp/aic.sdf
#   ./aic_utils/aic_mujoco/setup_pipeline.sh mjcf     # only sdformat_mjcf + add_cable_plugin
#   ./aic_utils/aic_mujoco/setup_pipeline.sh mjx      # only the python mjx env
#   ./aic_utils/aic_mujoco/setup_pipeline.sh smoke    # only the smoke tests
# ============================================================================

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# ---- colored logging ----
B="\033[1m" G="\033[32m" Y="\033[33m" R="\033[31m" N="\033[0m"
log()   { printf "${B}[setup]${N} %s\n" "$*"; }
ok()    { printf "${G}[ok]${N} %s\n" "$*"; }
warn()  { printf "${Y}[warn]${N} %s\n" "$*" >&2; }
fail()  { printf "${R}[fail]${N} %s\n" "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || fail "missing: $1"; }

PHASE="${1:-all}"

# ============================================================================
# STEP 1: host dependencies (apt + CUDA-aware pip)
# ============================================================================
step_deps() {
  log "STEP 1: host dependencies"

  if [[ "$PHASE" != "deps" && "$PHASE" != "all" ]]; then return; fi

  # 1.1 base apt
  SUDO=""
  if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi
  export DEBIAN_FRONTEND=noninteractive

  $SUDO apt update
  $SUDO apt install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release wget git \
      build-essential cmake pkg-config \
      python3 python3-dev python3-pip python3-venv \
      libegl1 libgl1 libglib2.0-0 libxext6 libx11-6 \
      libgles2 libglu1-mesa mesa-utils mesa-common-dev \
      libusb-1.0-0-dev libudev-dev \
      swig libfuse2

  # 1.2 ROS 2 apt key + repo (Kilted)
  if [[ ! -f /etc/apt/sources.list.d/ros2.list ]]; then
    log "  adding ROS 2 Kilted apt repo"
    $SUDO curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
        | $SUDO tee /etc/apt/sources.list.d/ros2.list
  fi

  # 1.3 Gazebo apt repo (Harmonic — matches aic_eval container)
  if [[ ! -f /etc/apt/sources.list.d/gazebo-stable.list ]]; then
    log "  adding Gazebo stable apt repo"
    $SUDO curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        | $SUDO tee /etc/apt/sources.list.d/gazebo-stable.list
  fi

  $SUDO apt update

  # 1.4 ROS 2 base + Gazebo Harmonic + colcon + vcstool
  $SUDO apt install -y --no-install-recommends \
      ros-kilted-ros-base ros-kilted-ros-core \
      ros-kilted-gz-sim-vendor ros-kilted-gz-ros2-control \
      python3-colcon-common-extensions python3-vcstool \
      python3-rosdep python3-rosinstall-generator

  if ! $SUDO rosdep init 2>/dev/null; then ok "rosdep already initialized"; fi
  $SUDO rosdep update || warn "rosdep update failed; we'll retry inside the workspace"

  ok "host dependencies installed"
}

# ============================================================================
# STEP 2: ROS 2 install (done by step_deps on Ubuntu 24.04 + Noble)
# ============================================================================
step_ros() {
  log "STEP 2: ROS 2 install (handled by step 1 on Ubuntu 24.04)"
  if ! dpkg -l ros-kilted-ros-base >/dev/null 2>&1; then
    warn "ROS Kilted not detected — re-run with './setup_pipeline.sh deps'"
  else
    ok "ros-kilted-ros-base installed ($(dpkg -s ros-kilted-ros-base | awk '/^Version:/{print $2}'))"
  fi
}

# ============================================================================
# STEP 3: build aic_ws (colcon) with gz-mujoco, mujoco_vendor, mujoco_ros2_control
# ============================================================================
step_workspace() {
  log "STEP 3: AIC workspace build (gz-mujoco + mujoco_vendor + mujoco_ros2_control)"

  WS="$HOME/aic_ws"
  if [[ ! -d "$WS/src" ]]; then
    log "  creating $WS"
    mkdir -p "$WS/src"
  fi

  # 3.1 if ROS is installed natively, source it; otherwise skip — step 5
  # (the docker image) is the alternative for users who don't want a host
  # ROS install.
  if [[ -f /opt/ros/kilted/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/kilted/setup.bash
  else
    warn "ROS Kilted not installed natively — skipping native workspace build."
    warn "Either run ./setup_pipeline.sh deps first, or use the docker path in step 5."
    return 0
  fi

  # 3.2 clone gz-mujoco + mujoco_vendor + mujoco_ros2_control
  cd "$WS/src"
  if [[ ! -d "gz-mujoco" ]]; then
    git clone https://github.com/gazebosim/gz-mujoco.git -b aic
  fi
  if [[ ! -d "mujoco/mujoco_vendor" ]]; then
    git clone https://github.com/yadunund/mujoco_vendor.git -b aic mujoco/mujoco_vendor
  fi
  if [[ ! -d "mujoco/mujoco_ros2_control" ]]; then
    git clone https://github.com/yadunund/mujoco_ros2_control.git -b aic mujoco/mujoco_ros2_control
  fi

  # 3.3 symlink the AIC repo's aic_description, aic_assets, aic_bringup,
  # aic_controller, aic_interfaces, aic_engine, aic_gazebo, aic_mujoco
  # into the workspace so colcon can build them as one set.
  if [[ ! -L "$WS/src/aic" ]]; then
    ln -sf "$REPO_ROOT" "$WS/src/aic"
    ok "  symlinked $WS/src/aic -> $REPO_ROOT"
  fi

  # 3.4 rosdep
  if [[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo -E rosdep install --from-paths "$WS/src" --ignore-src \
        --rosdistro kilted -y --skip-keys "gz-cmake3 DART libogre-dev libogre-next-2.3-dev" || warn "rosdep had errors, continuing"
  else
    warn "rosdep sources.list missing — skipping rosdep install. Try: sudo rosdep init && sudo rosdep update"
  fi

  # 3.5 colcon build (incremental)
  cd "$WS"
  GZ_BUILD_FROM_SOURCE=1 colcon build \
      --merge-install --symlink-install \
      --packages-ignore lerobot_robot_aic \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --event-handlers console_direct+ 2>&1 | tail -60

  ok "workspace built at $WS/install"
  ok "source: source $WS/install/setup.bash"
}

# ============================================================================
# STEP 4: export /tmp/aic.sdf from aic_gz_bringup
# ============================================================================
step_sdf() {
  log "STEP 4: export /tmp/aic.sdf"

  # 4.1 detect source location
  if [[ -f /opt/ros/kilted/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/kilted/setup.bash
  fi
  if [[ -f "$HOME/aic_ws/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/aic_ws/install/setup.bash"
  fi
  if [[ -f "/ws_aic/install/setup.bash" ]] && [[ -d /ws_aic ]]; then
    # shellcheck disable=SC1091
    source /ws_aic/install/setup.bash
  fi

  if ! ros2 pkg list 2>/dev/null | grep -q aic_bringup; then
    fail "aic_bringup not found. Run step 3 (workspace) first."
  fi

  # 4.2 bringup
  mkdir -p /tmp
  rm -f /tmp/aic.sdf
  log "  launching aic_gz_bringup — wait for 'trial complete' in the log"
  log "  (this takes 30-90s and writes /tmp/aic.sdf when done)"

  cd /tmp
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
      start_aic_engine:=true \
      save_world:=/tmp/aic.sdf \
      spawn_task_board:=true \
      spawn_cable:=true \
      cable_type:=sfp_sc_cable \
      attach_cable_to_gripper:=true \
      ground_truth:=true \
      gazebo_gui:=false \
      launch_rviz:=false \
      headless_render:=true 2>&1 | tee /tmp/aic_bringup.log

  for i in {1..120}; do
    [[ -s /tmp/aic.sdf ]] && break
    sleep 1
  done

  if [[ ! -s /tmp/aic.sdf ]]; then
    fail "/tmp/aic.sdf was not written within 120s. See tail /tmp/aic_bringup.log:"
  fi

  # 4.3 sanity check
  MODELS=$(grep -c "<model" /tmp/aic.sdf || true)
  log "  /tmp/aic.sdf = $(wc -c < /tmp/aic.sdf) bytes, $MODELS <model> tags"
  if (( MODELS < 5 )); then
    fail "expected >5 <model> tags (robot + task_board + cable + ports), got $MODELS. check spawn_* flags."
  fi

  # 4.4 fix the two known SDF bugs before sdf2mjcf
  log "  patching /tmp/aic.sdf for the two known URI bugs"
  sed -i 's|file://<urdf-string>/model://|model://|g' /tmp/aic.sdf
  sed -i 's|file:///lc_plug_visual.glb|model://LC Plug/lc_plug_visual.glb|g' /tmp/aic.sdf
  sed -i 's|file:///sc_plug_visual.glb|model://SC Plug/sc_plug_visual.glb|g' /tmp/aic.sdf
  sed -i 's|file:///sfp_module_visual.glb|model://SFP Module/sfp_module_visual.glb|g' /tmp/aic.sdf

  ok "/tmp/aic.sdf exported"
}

# ============================================================================
# STEP 5: sdformat_mjcf + add_cable_plugin
# ============================================================================
step_mjcf() {
  log "STEP 5: sdformat_mjcf + add_cable_plugin"

  if [[ ! -s /tmp/aic.sdf ]]; then
    fail "no /tmp/aic.sdf — run step 4 (sdf) first"
  fi

  if [[ -f /opt/ros/kilted/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/kilted/setup.bash
  fi
  if [[ -f "$HOME/aic_ws/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/aic_ws/install/setup.bash"
  fi
  if [[ -f /ws_aic/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /ws_aic/install/setup.bash
  fi

  if ! command -v sdf2mjcf >/dev/null 2>&1; then
    fail "sdf2mjcf not found. Run step 3 (workspace) first to build gz-mujoco."
  fi

  DEST="$HOME/aic_mujoco_world"
  rm -rf "$DEST"
  mkdir -p "$DEST"

  log "  sdf2mjcf /tmp/aic.sdf -> $DEST"
  sdf2mjcf /tmp/aic.sdf "$DEST/aic_world.xml"

  log "  copy assets into aic_utils/aic_mujoco/mjcf/"
  MJCF_DIR="$REPO_ROOT/aic_utils/aic_mujoco/mjcf"
  cp "$DEST"/* "$MJCF_DIR/"
  ok "  copied $(ls "$DEST" | wc -l) files into $MJCF_DIR"

  log "  running add_cable_plugin.py"
  cd "$REPO_ROOT/aic_utils/aic_mujoco"
  python3 scripts/add_cable_plugin.py \
      --input mjcf/aic_world.xml \
      --output mjcf/aic_world.xml \
      --robot_output mjcf/aic_robot.xml \
      --scene_output mjcf/scene.xml

  log "  re-building aic_mujoco so the new xml gets installed"
  cd "$HOME/aic_ws"
  colcon build --packages-select aic_mujoco \
      --merge-install --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -30

  ok "scene.xml, aic_robot.xml, aic_world.xml ready"
}

# ============================================================================
# STEP 6: pixi env with mujoco-warp / mujoco.mjx
# ============================================================================
step_mjx() {
  log "STEP 6: pixi env with mujoco.mjx (GPU-backed)"

  cd "$REPO_ROOT"

  # 6.1 patch pixi.toml: bump mujoco to >=3.2 (mjx lives there) + add jax
  PATCHED=false
  if ! grep -q "mujoco = \">=3" pixi.toml; then
    sed -i 's|^mujoco = "==3\.5\.0"|mujoco = ">=3.2,<4"|' pixi.toml
    PATCHED=true
  fi
  if ! grep -q "jax " pixi.toml; then
    # add jax + jaxlib under [pypi-dependencies] — pinned to current stable
    sed -i '/^wandb = ">=0\.20"/a jax = { version = ">=0.4.35,<0.5", channel = "conda-forge" }\njaxlib = { version = ">=0.4.35,<0.5", channel = "conda-forge" }' pixi.toml
    PATCHED=true
  fi
  if $PATCHED; then
    ok "patched pixi.toml (mujoco>=3.2 + jax/jaxlib)"
  else
    ok "pixi.toml already has mujoco>=3.2 + jax"
  fi

  # 6.2 lock + install
  log "  pixi install (this resolves mujoco + jax + jaxlib + cuda)"
  pixi install 2>&1 | tail -20

  # 6.3 verify inside pixi env
  log "  verifying mujoco.mjx + jax + cuda"
  pixi run python - <<'PY'
import sys
print("python:", sys.version.split()[0])

import mujoco
print("mujoco:", mujoco.__version__)
assert tuple(int(x) for x in mujoco.__version__.split(".")[:2]) >= (3, 2), "need mujoco>=3.2 for mjx"

from mujoco import mjx
print("mjx ok, module:", mjx.__name__)

import jax, jaxlib
print("jax:", jax.__version__, "jaxlib:", jaxlib.__version__)
print("jax devices:", jax.devices())
PY

  ok "mujoco.mjx + jax installed and verified"
}

# ============================================================================
# STEP 7: smoke tests (sanity, runnable now)
# ============================================================================
step_smoke() {
  log "STEP 7: smoke tests"
  cd "$REPO_ROOT"

  log "  7.1 mujoco scene loads"
  pixi run python RL/scripts/setup_mujoco.py --steps 50 \
      || warn "scene.xml failed to load — run step 5 (mjcf) first"

  log "  7.2 wandb connectivity"
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    pixi run python RL/scripts/connect_wandb.py --steps 20 \
        --run-name pipeline_smoke || warn "wandb smoke failed"
  else
    warn "  WANDB_API_KEY not set — skipping wandb smoke test"
    echo "    set it with:  export WANDB_API_KEY=..."
  fi

  ok "smoke tests complete"
}

# ============================================================================
# dispatch
# ============================================================================
case "$PHASE" in
  all)
    step_deps
    step_ros
    step_workspace
    step_sdf
    step_mjcf
    step_mjx
    step_smoke
    ;;
  deps)      step_deps ;;
  ros)       step_ros ;;
  workspace) step_workspace ;;
  sdf)       step_sdf ;;
  mjcf)      step_mjcf ;;
  mjx)       step_mjx ;;
  smoke)     step_smoke ;;
  *)
    echo "Usage: $0 {all|deps|ros|workspace|sdf|mjcf|mjx|smoke}"
    exit 2
    ;;
esac

ok "phase '$PHASE' done"