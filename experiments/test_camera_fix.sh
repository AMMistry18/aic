#!/usr/bin/env bash
set -eo pipefail
export CONDA_PREFIX=/ws_aic/src/aic/.pixi/envs/default
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export AMENT_PREFIX_PATH="$CONDA_PREFIX" ROS_DISTRO=kilted
export PYTHONPATH="/research/aic_model:/research/aic_example_policies:/research:${PYTHONPATH:-}"
cd /research
python -m pytest -q aic_model/test/test_camera_image_time.py aic_model/test/test_sc_controller.py aic_model/test/test_sc_plug_pose.py aic_model/test/test_sfp_plug_pose.py testing/sfp_v50_validation/tests experiments/tests
