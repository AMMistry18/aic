#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export PYTHONPATH="/research:/research/aic_model:/research/aic_example_policies:${PYTHONPATH:-}"
exec /opt/research-perception/bin/python /research/experiments/replay_sc_live_screen30.py "$@"
