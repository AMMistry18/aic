#!/usr/bin/env bash
set -euo pipefail
cd /ws_aic/src/aic
export CONDA_PREFIX=/ws_aic/src/aic/.pixi/envs/default
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export AMENT_PREFIX_PATH="$CONDA_PREFIX"
export ROS_DISTRO=kilted
python /research/experiments/container_probe.py
