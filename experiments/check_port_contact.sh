#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=47 GZ_PARTITION=research_lastinch_kilted
setsid ros2 run ros_gz_bridge parameter_bridge \
 '/world/aic_lastinch/model/cable_2/link/sc_plug_link/sensor/research_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts' \
 --ros-args -r /world/aic_lastinch/model/cable_2/link/sc_plug_link/sensor/research_contact/contact:=/research/contact_probe \
 > /research/.artifacts/kilted_handoff/contact_bridge.log 2>&1 &
bridge_pid=$!
trap 'kill -TERM -- -"$bridge_pid" 2>/dev/null || true' EXIT
python3 /research/experiments/check_port_contact.py
