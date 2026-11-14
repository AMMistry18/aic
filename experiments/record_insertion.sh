#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=47 GZ_PARTITION=research_lastinch_kilted
active_cable=$(python3 -c "import json; print('cable_'+str(json.load(open('/research/.artifacts/kilted_handoff/scene.json'))['reset']['target_index']))")
setsid ros2 run ros_gz_bridge parameter_bridge \
 "/model/${active_cable}/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V" \
 '/lastinch/overview/image@sensor_msgs/msg/Image[gz.msgs.Image' \
 '/lastinch/target/image@sensor_msgs/msg/Image[gz.msgs.Image' \
 '/lastinch/board/image@sensor_msgs/msg/Image[gz.msgs.Image' \
 > /research/.artifacts/container/insertion_recorder_bridge.log 2>&1 &
bridge_pid=$!
trap 'kill -TERM -- -"$bridge_pid" 2>/dev/null || true' EXIT
python3 /research/experiments/record_insertion.py
