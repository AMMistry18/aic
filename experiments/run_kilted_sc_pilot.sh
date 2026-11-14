#!/usr/bin/env bash
set -eo pipefail
export CONDA_PREFIX=/ws_aic/src/aic/.pixi/envs/default
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export AMENT_PREFIX_PATH="$CONDA_PREFIX" ROS_DISTRO=kilted
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=47
export PYTHONPATH="/research/aic_model:/research/aic_example_policies:${PYTHONPATH:-}"
export AIC_PERCEPTION_WEIGHTS_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/aic_example_policies/ros/weights"
export AIC_SFP_PLUG_POSE_WEIGHTS="$AIC_PERCEPTION_WEIGHTS_DIR/best_sfp_plug_pose.pt"
export AIC_SC_PLUG_POSE_WEIGHTS="$AIC_PERCEPTION_WEIGHTS_DIR/best_sc_plug_pose.pt"
export RL_INSERT_REPORT_MISS_AS_SUCCESS=0 RL_INSERT_SC_MISS_ABORTS=1
export RL_INSERT_ACTION_TIME_BUDGET_S=40 RL_INSERT_CALIB_DUMP=0 RL_INSERT_SC_CALIB_DUMP=0
python /research/experiments/run_kilted_sc_pilot.py
