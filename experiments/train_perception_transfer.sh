#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export PYTHONPATH="/research/aic_model:/research/aic_example_policies:${PYTHONPATH:-}"
/opt/research-perception/bin/python /research/experiments/train_perception_transfer.py "$@"
