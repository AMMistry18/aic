#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export PYTHONPATH="/research/aic_model:/research/aic_example_policies:/research/flowstate/aic_perception:${PYTHONPATH:-}"
/opt/research-perception/bin/python /research/experiments/evaluate_sfp_visual33.py "$@"
