#!/usr/bin/env bash
set -euo pipefail
cd /ws_aic/src/aic
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="/research/aic_model:/research/aic_example_policies:${PYTHONPATH:-}"
export CONDA_PREFIX=/ws_aic/src/aic/.pixi/envs/default
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export AMENT_PREFIX_PATH="$CONDA_PREFIX"
export ROS_DISTRO=kilted
python -m pytest -q --tb=short \
  /research/aic_model/test/test_sc_controller.py \
  /research/aic_model/test/test_sc_mouth_pose_geometry.py \
  /research/aic_model/test/test_sc_plug_pose.py \
  /research/aic_model/test/test_sc_plug_pose_geometry.py \
  /research/aic_model/test/test_sc_visual_alignment.py \
  /research/aic_model/test/test_sfp_plug_pose.py \
  /research/aic_model/test/test_sfp_plug_pose_trials.py \
  /research/aic_model/test/test_sfp_controller.py \
  /research/testing/sfp_v50_validation/tests
