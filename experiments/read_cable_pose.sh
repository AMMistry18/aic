#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=47 GZ_PARTITION=research_lastinch_kilted
setsid ros2 run ros_gz_bridge parameter_bridge '/model/cable_2/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V' > /research/.artifacts/kilted_handoff/pose_bridge.log 2>&1 &
bridge_pid=$!
trap 'kill -TERM -- -"$bridge_pid" 2>/dev/null || true' EXIT
python3 /research/experiments/read_cable_pose.py
