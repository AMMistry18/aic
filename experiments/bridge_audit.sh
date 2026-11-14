#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=47 GZ_PARTITION=research_lastinch_kilted
exec ros2 run ros_gz_bridge parameter_bridge \
 '/model/cable_2/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V' \
 '/model/ur5e/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
