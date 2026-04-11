FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

# Add other dependencies

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --locked
RUN cd /ws_aic/src/aic && pixi run python -m ensurepip --upgrade && \
    pixi run python -m pip install ultralytics
RUN cp /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/PerceptionInsert.py \
       /ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/ && \
    cp /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/perception_core.py \
       /ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/ && \
    mkdir -p /ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/weights && \
    cp /ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/weights/best.pt \
       /ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/weights/

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

# prepare the credentials file
if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# start the model
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
CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.PerceptionInsert", "-p", "use_sim_time:=true"]
