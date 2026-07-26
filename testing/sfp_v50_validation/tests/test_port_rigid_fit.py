from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# RLInsert imports rclpy/aic_task_interfaces/etc, but it happens to import
# cleanly under this test env (no ROS runtime needed for the pure-math port
# fit); pull the helpers straight from the shipped module rather than forking
# them into a copy, so this test exercises the exact code that runs.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "aic_model"))

from aic_model.RLInsert import (  # noqa: E402
    LOCAL_SFP_PORT_KPS,
    RLInsert,
    _axis_angle_to_R,
    _q_to_R,
    _weighted_kabsch_fit,
)


def _orientation(kp_3d):
    # _estimate_sfp_port_orientation never touches self; RLInsert's own
    # test_sc_controller.py reuses it the same unbound way.
    return RLInsert._estimate_sfp_port_orientation(None, kp_3d)


def test_noise_free_fit_recovers_center_and_yaw_exactly():
    yaw_true = 0.41
    t_true = np.array([0.52, -0.18, 0.11])
    R_true = _axis_angle_to_R(np.array([0.0, 0.0, yaw_true]))
    world_pts = (R_true @ LOCAL_SFP_PORT_KPS.T).T + t_true

    R_fit, t_fit = _weighted_kabsch_fit(LOCAL_SFP_PORT_KPS, world_pts, np.ones(4))

    assert np.allclose(R_fit, R_true, atol=1e-9)
    assert np.allclose(t_fit, t_true, atol=1e-9)

    fitted_kp_3d = (R_fit @ LOCAL_SFP_PORT_KPS.T).T + t_fit
    q_wxyz, yaw_fit = _orientation(fitted_kp_3d)
    assert q_wxyz is not None
    assert np.isclose(yaw_fit, yaw_true, atol=1e-9)


def test_low_weight_on_displaced_corner_beats_unweighted_average():
    t_true = np.array([0.50, 0.0, 0.12])
    R_true = _axis_angle_to_R(np.array([0.0, 0.0, 0.15]))
    world_pts = (R_true @ LOCAL_SFP_PORT_KPS.T).T + t_true

    displaced = world_pts.copy()
    displaced[1] += np.array([0.006, -0.004, 0.003])  # one bad corner, ~7-8mm off

    naive_center = displaced.mean(axis=0)
    weights = np.array([1.0, 0.05, 1.0, 1.0])  # the bad corner gets almost no vote
    _, t_fit = _weighted_kabsch_fit(LOCAL_SFP_PORT_KPS, displaced, weights)

    err_fit = np.linalg.norm(t_fit - t_true)
    err_naive = np.linalg.norm(naive_center - t_true)
    assert err_fit < err_naive


def test_three_corner_fit_recovers_center():
    t_true = np.array([-0.1, 0.3, 0.2])
    R_true = _axis_angle_to_R(np.array([0.0, 0.0, -0.6]))
    world_pts = (R_true @ LOCAL_SFP_PORT_KPS.T).T + t_true

    idx = [0, 1, 3]  # corner 2 missing/failed to triangulate
    R_fit, t_fit = _weighted_kabsch_fit(
        LOCAL_SFP_PORT_KPS[idx], world_pts[idx], np.ones(3))

    assert np.allclose(t_fit, t_true, atol=1e-9)
    fitted_kp_3d = (R_fit @ LOCAL_SFP_PORT_KPS.T).T + t_fit
    assert np.allclose(fitted_kp_3d[2], world_pts[2], atol=1e-9)  # dropped corner still recovered


def test_two_corner_input_signals_fallback():
    t_true = np.array([0.1, 0.1, 0.1])
    world_pts = LOCAL_SFP_PORT_KPS + t_true
    idx = [0, 1]
    with pytest.raises(ValueError):
        _weighted_kabsch_fit(LOCAL_SFP_PORT_KPS[idx], world_pts[idx], np.ones(2))


def test_orientation_axis_forced_to_world_negative_z():
    # Even a tilted raw rectangle must report insertion axis = world -Z; this
    # convention is deliberate (see _estimate_sfp_port_orientation) and must
    # not change under the rigid-fit refactor.
    R_true = _axis_angle_to_R(np.array([0.05, -0.03, 0.7]))
    t_true = np.array([0.2, -0.05, 0.09])
    world_pts = (R_true @ LOCAL_SFP_PORT_KPS.T).T + t_true

    R_fit, t_fit = _weighted_kabsch_fit(LOCAL_SFP_PORT_KPS, world_pts, np.ones(4))
    fitted_kp_3d = (R_fit @ LOCAL_SFP_PORT_KPS.T).T + t_fit
    q_wxyz, _ = _orientation(fitted_kp_3d)
    assert q_wxyz is not None
    R_tip = _q_to_R(*q_wxyz)
    assert np.allclose(R_tip[:, 2], [0.0, 0.0, -1.0], atol=1e-9)
