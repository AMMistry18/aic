#!/usr/bin/env bash
set -eo pipefail
export CONDA_PREFIX=/ws_aic/src/aic/.pixi/envs/default
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python /research/experiments/infer_saved_ports.py --images /research/experiments/evidence/perception_development --out /research/.artifacts/saved_port_inference
