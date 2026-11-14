#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
root=/research/.artifacts/dev46_workspace
test -f "$root/experiments/gazebo/CMakeLists.txt"
test -x /opt/research-perception/bin/python
bash -n "$root/experiments/start_kilted_handoff.sh"
cmake -S "$root/experiments/gazebo" -B /opt/research-dev47-grasp
cmake --build /opt/research-dev47-grasp -j 1
