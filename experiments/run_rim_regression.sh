#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
export PYTHONPATH="/research/aic_model:/research/aic_example_policies:${PYTHONPATH:-}"
python3 -m pytest -q /research/aic_model/test/test_sc_controller.py /research/aic_model/test/test_sc_wrench_frame.py /research/aic_model/test/test_camera_image_time.py /research/aic_model/test/test_visual_reacquisition.py /research/aic_model/test/test_sc_rim_refinement.py /research/aic_model/test/test_contact_registration.py /research/aic_model/test/test_sc_rim_tracking.py /research/aic_model/test/test_wrist_observation.py /research/aic_model/test/test_contact_capture.py /research/aic_model/test/test_native_wrench_gain.py /research/aic_model/test/test_sc_body_refinement.py /research/aic_model/test/test_tracking_compensation.py
