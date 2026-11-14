#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
python3 /research/experiments/render_perception_training_case.py "$@"
