FROM docker.io/library/ros:kilted-ros-core

# Build from the exact upstream Phase 1 revision used by the validated image,
# independent of whichever branch contains this Dockerfile.
ARG AIC_PHASE1_COMMIT=1c1e018eaecadf1829e18603bb531af30e02fce2
RUN apt update && \
    apt install -y git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    git init /ws_aic/src/aic && \
    cd /ws_aic/src/aic && \
    git remote add origin https://github.com/intrinsic-dev/aic.git && \
    git fetch --depth 1 origin "$AIC_PHASE1_COMMIT" && \
    git checkout --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "$AIC_PHASE1_COMMIT"

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=0.67.2 bash
ENV PATH="/root/.pixi/bin:$PATH"

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache \
    --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --frozen

# Use the resolved environment directly. Running `pixi run` here can trigger a
# fresh manifest solve instead of using the already frozen environment.
RUN cd /ws_aic/src/aic && \
    .pixi/envs/default/bin/python -m ensurepip --upgrade && \
    .pixi/envs/default/bin/python -m pip install --no-cache-dir "ultralytics==8.4.48"

# Apply the exact V50 source and model weights after dependency layers so
# policy-only changes remain quick Docker rebuilds.
COPY docker/aic_model/v50_overlay /tmp/v50_overlay
RUN cd /ws_aic/src/aic && \
    SITE_PACKAGES="$(.pixi/envs/default/bin/python -c 'import site; print(site.getsitepackages()[0])')" && \
    cp /tmp/v50_overlay/aic_model/*.py /ws_aic/src/aic/aic_model/aic_model/ && \
    cp /tmp/v50_overlay/aic_model/*.py "$SITE_PACKAGES/aic_model/" && \
    cp /tmp/v50_overlay/aic_example_policies/ros/*.py \
       /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/ && \
    cp /tmp/v50_overlay/aic_example_policies/ros/*.py \
       "$SITE_PACKAGES/aic_example_policies/ros/" && \
    mkdir -p \
       /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/weights \
       "$SITE_PACKAGES/aic_example_policies/ros/weights" \
       "$SITE_PACKAGES/aic_perception" && \
    cp /tmp/v50_overlay/aic_example_policies/ros/weights/*.pt \
       /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/weights/ && \
    cp /tmp/v50_overlay/aic_example_policies/ros/weights/*.pt \
       "$SITE_PACKAGES/aic_example_policies/ros/weights/" && \
    cp /tmp/v50_overlay/aic_perception/*.py "$SITE_PACKAGES/aic_perception/" && \
    .pixi/envs/default/bin/python -m py_compile \
       /ws_aic/src/aic/aic_model/aic_model/RLInsert.py \
       /ws_aic/src/aic/aic_model/aic_model/v50_controller.py \
       /ws_aic/src/aic/aic_model/aic_model/sc_controller.py \
       /ws_aic/src/aic/aic_model/aic_model/sc_visual_alignment.py \
       /ws_aic/src/aic/aic_model/aic_model/sfp_plug_pose.py \
       /ws_aic/src/aic/aic_model/aic_model/aic_model.py && \
    .pixi/envs/default/bin/python -c \
       "import cv2; from ultralytics import YOLO; from aic_model.RLInsert import RLInsert, SC_WEIGHTS; from aic_model.sc_controller import run_sc_insertion; from aic_model.sc_visual_alignment import detect_sc_duplex_opening; from aic_model.v50_controller import run_v50_script; from aic_example_policies.ros.sc_mouth_pose_geometry import SC_MOUTH_KEYPOINT_COUNT; from aic_perception.gripper_masks import GripperMaskBank; model = YOLO(SC_WEIGHTS); assert model.model.kpt_shape == [SC_MOUTH_KEYPOINT_COUNT, 3]; GripperMaskBank(); print('SFP V50 + five-keypoint SC mouth imports OK')"

ENV AIC_SFP_PLUG_POSE_WEIGHTS=/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/weights/best_sfp_plug_pose.pt
ENV RL_INSERT_ACTION_TIME_BUDGET_S=720
# SC alignment closes lateral error at ~0.19 mm/s (1.5 mm segments, ~6.3 s each),
# so the 15 s default caps convergence at ~2.9 mm while SC_MAX_HANDOFF_LATERAL_M
# admits 10 mm.  The 2026-07-27 03:33 run started 4.90 mm out, was still
# converging (4.89 -> 3.27 -> 2.00 mm) when the timer fired, and seated 2 mm
# off-centre with 115 s of budget unused.  60 s covers the full admissible
# handoff range; _enforce_action_deadline inside the align loop remains the
# real bound, so this cannot overrun the action budget.
ENV RL_INSERT_SC_ALIGN_TIMEOUT_S=60
ENV RL_INSERT_VISUAL_GAP_WEDGE_ENABLE=1
ENV RL_INSERT_VISUAL_GAP_DEBUG=1
ENV RL_INSERT_V50_TARGET_FORCE_N=10.0
ENV RL_INSERT_V50_SEAT_FORCE_CAP_N=12.0
ENV RL_INSERT_FORCE_ABORT_N=18.0
ENV RL_INSERT_V50_LIFT_DISTANCE_M=0.006
ENV RL_INSERT_V50_MAX_VISUAL_RESCUES=1
ENV RL_INSERT_V50_MAX_LIFT_RECOVERIES=1
ENV RL_INSERT_CONTROL_MODE=script
ENV RL_INSERT_SCRIPT_BIAS_X_M=0
ENV RL_INSERT_SCRIPT_BIAS_Y_M=0
ENV RL_INSERT_SCRIPT_BIAS_RX_RAD=0
ENV RL_INSERT_MODEL=""
ENV RL_INSERT_SEAT_MODEL=""

WORKDIR /ws_aic/src/aic
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_model.RLInsert", "-p", "use_sim_time:=true"]
