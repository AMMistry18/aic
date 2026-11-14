#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
python3 /research/experiments/container_probe.py
ros2 pkg prefix aic_controller
ros2 pkg prefix aic_assets
gz sim --versions
command -v Xvfb || true
sha256sum /ws_aic/install/share/aic_assets/models/sc_port/model.sdf
