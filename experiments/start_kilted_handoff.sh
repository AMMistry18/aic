#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
# A small source overlay pins Phase 1 geometry without rebuilding Gazebo.
prefix=/opt/research-phase1
mkdir -p "$prefix/share/ament_index/resource_index/packages"
for package in aic_assets aic_description aic_bringup; do
  touch "$prefix/share/ament_index/resource_index/packages/$package"
  ln -sfn "/research/.artifacts/upstream-phase1/$package" "$prefix/share/$package"
done
export AMENT_PREFIX_PATH="$prefix:$AMENT_PREFIX_PATH"
export GZ_SIM_RESOURCE_PATH="/research/.artifacts/upstream-phase1/aic_assets/models:${GZ_SIM_RESOURCE_PATH:-}"
export SDF_PATH="$GZ_SIM_RESOURCE_PATH"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/ws_aic/install/lib:/opt/ros/kilted/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=47
export GZ_PARTITION=research_lastinch_kilted
export LIBGL_ALWAYS_SOFTWARE=1
export DISPLAY=:99
cmake -S /research/experiments/gazebo -B /opt/research-grasp > /research/.artifacts/container/grasp_build.log 2>&1
cmake --build /opt/research-grasp -j 2 >> /research/.artifacts/container/grasp_build.log 2>&1
# Docker preserves /tmp across stops. A lock PID can be reused by an unrelated
# startup process, so Xvfb's stale-lock detection alone is insufficient. This
# display belongs to this isolated evaluator; never remove a live X server.
if pgrep -x Xvfb >/dev/null || pgrep -x Xorg >/dev/null; then
  echo 'Unexpected live X server in isolated evaluator; refusing second startup' >&2
  exit 1
fi
rm -f -- /tmp/.X99-lock /tmp/.X11-unix/X99
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp > /research/.artifacts/container/xvfb.log 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
for attempt in {1..50}; do
  if ! kill -0 "$xvfb_pid" 2>/dev/null; then cat /research/.artifacts/container/xvfb.log >&2; exit 1; fi
  if [ -S /tmp/.X11-unix/X99 ]; then break; fi
  sleep 0.1
done
if [ ! -S /tmp/.X11-unix/X99 ]; then echo 'Xvfb socket not ready' >&2; exit 1; fi
python3 /research/experiments/prepare_kilted_handoff.py \
 --scene "${AIC_RESEARCH_SCENE:-/research/.artifacts/lastinch_sc_native_v2}" --out /research/.artifacts/kilted_handoff \
 --camera-rate "${AIC_RESEARCH_CAMERA_RATE:-20}" --physics-step "${AIC_RESEARCH_PHYSICS_STEP:-0.002}" \
 --wrist-rate "${AIC_RESEARCH_WRIST_RATE:-100}"
exec ros2 launch /research/experiments/kilted_handoff.launch.py
