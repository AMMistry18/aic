#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=47 GZ_PARTITION=research_lastinch_kilted
exec python3 /research/experiments/record_contact_witness.py "$@"
