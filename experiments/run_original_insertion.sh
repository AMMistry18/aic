#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=47
export PYTHONPATH="/research/aic_model:/research/aic_example_policies:/research/flowstate/aic_perception:${PYTHONPATH:-}"
export AIC_PERCEPTION_WEIGHTS_DIR=/research/aic_example_policies/aic_example_policies/ros/weights
export AIC_SFP_PLUG_POSE_WEIGHTS="${AIC_SFP_PLUG_POSE_WEIGHTS:-$AIC_PERCEPTION_WEIGHTS_DIR/best_sfp_plug_pose.pt}"
export AIC_SC_PLUG_POSE_WEIGHTS="${AIC_SC_PLUG_POSE_WEIGHTS:-$AIC_PERCEPTION_WEIGHTS_DIR/best_sc_plug_pose.pt}"
# Disable score-oriented success reporting and diagnostic simulator pose lookup.
export RL_INSERT_REPORT_MISS_AS_SUCCESS=0 RL_INSERT_SC_MISS_ABORTS=1
export RL_INSERT_CALIB_DUMP=0 RL_INSERT_SC_CALIB_DUMP=0
export AIC_INSERT_CONTROLLER_SIM_TIME=1 AIC_INSERT_OBSERVABLE_TERMINAL=1
export AIC_INSERT_HIGH_RATE_WRIST=1
# Research protocol: the legacy external ceiling is 8 N; a new evaluator-owned
# profile can prescribe 20 N. Internal conservative 6 N guards remain active.
# Increasing the ceiling alone does not request more insertion force. The original
# SC 7 N cap / 12 N sustained abort could reach it before recovery.  A 1 mm
# axial lead at 500 N/m requests 0.5 N elastic load; native feedback and cable
# loading are measured independently, never assumed equal to this request.
export RL_INSERT_SC_TARGET_FORCE_N=1.5 RL_INSERT_SC_SEAT_FORCE_CAP_N=3.0
export RL_INSERT_SC_FORCE_ABORT_N=6.0 RL_INSERT_SC_MAX_AXIAL_LEAD_M=0.001
export RL_INSERT_SC_CONTACT_FORCE_N=1.0 RL_INSERT_SC_STALL_TIMEOUT_S=1.0
export RL_INSERT_SC_SEAT_STALL_GRACE_S=0.0 RL_INSERT_SC_STALL_EVENT_DWELL_WALL_S=0.3
# Source-box audit: entry half-clearance is about 0.348 / 0.396 mm, not the
# keypoint cuboid's 1.2 / 0.85 mm. Sample a wider search at <=0.295 mm outer
# arc spacing and 0.3 mm turn pitch. These are development settings, not results.
export RL_INSERT_SC_SPIRAL_MAX_RADIUS_M=0.003 RL_INSERT_SC_SPIRAL_PITCH_M=0.0003
export RL_INSERT_SC_SPIRAL_STEPS_PER_REV=64 RL_INSERT_SC_SPIRAL_SETTLE_CYCLES=1
export RL_INSERT_SC_COMMAND_DT_S=0.025
if [[ "${AIC_RESEARCH_CONNECTOR:-sc}" == sfp ]]; then
  export RL_INSERT_V50_TARGET_FORCE_N=1.5 RL_INSERT_V50_SEAT_FORCE_CAP_N=3.0
  export RL_INSERT_FORCE_ABORT_N=6.0 RL_INSERT_V50_MAX_AXIAL_LEAD_M=0.001
  export RL_INSERT_V50_FREE_SPEED_M_S=0.003 RL_INSERT_V50_CONTACT_SPEED_M_S=0.001
  export RL_INSERT_V50_SEAT_OVERTRAVEL_M=0 RL_INSERT_V50_MAX_WEDGE_RETRIES=1
  export RL_INSERT_V50_RETRACT_PULL_LEAD_M=0.004
fi
/opt/research-perception/bin/python /research/experiments/run_original_insertion.py
