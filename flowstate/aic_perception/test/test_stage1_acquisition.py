from __future__ import annotations

import math

import numpy as np
import pytest

from aic_perception.arm_ik import UR5eArm, _rot_z
from aic_perception.board_stage2 import Transform
from aic_perception.purple_insignia import analyze_purple
from aic_perception.stage1_acquisition import (
    MAX_TOTAL_JOINT_TRAVEL_RAD,
    MAX_WORST_JOINT_TRAVEL_RAD,
    OBSERVATION_JOINTS_RAD,
    interpolated_joint_waypoints,
    joint_travel,
    validate_observation_path,
)


HOME = np.radians([-9.15, -77.59, -95.39, -97.02, 90.01, 80.84])


def _arm():
    return UR5eArm(
        flange_T_tcp=Transform(np.eye(3), np.array([0.0, 0.0, 0.1971])),
        base=Transform(_rot_z(math.pi), np.zeros(3)),
    )


def test_purple_detector_finds_full_and_clipped_blobs():
    full = np.zeros((120, 160, 3), dtype=np.uint8)
    full[40:80, 60:100] = (180, 40, 160)
    report = analyze_purple(full, margin_px=5)
    assert report.seen and report.full
    assert report.center_error == pytest.approx((0.0, 0.0), abs=0.02)

    clipped = np.zeros((120, 160, 3), dtype=np.uint8)
    clipped[:30, :40] = (180, 40, 160)
    report = analyze_purple(clipped, margin_px=10)
    assert report.seen and not report.full
    assert {"left", "top"}.issubset(report.edges)


def test_observation_target_is_a_small_measured_move_from_flowstate_home():
    travel = joint_travel(HOME, OBSERVATION_JOINTS_RAD)
    assert math.degrees(travel.worst_rad) == pytest.approx(60.91, abs=0.05)
    assert math.degrees(travel.total_rad) == pytest.approx(125.93, abs=0.05)
    assert travel.worst_rad < MAX_WORST_JOINT_TRAVEL_RAD
    assert travel.total_rad < MAX_TOTAL_JOINT_TRAVEL_RAD


def test_501_degree_reconfiguration_is_rejected_by_total_cap():
    bad_start = np.asarray(OBSERVATION_JOINTS_RAD) + np.radians(
        [138.0, 16.9, 94.5, 56.1, -176.1, 19.6]
    )
    result = validate_observation_path(_arm(), bad_start)
    assert not result.safe
    assert "total joint travel" in result.reason


def test_waypoints_bound_every_guarded_transaction():
    target = np.asarray(OBSERVATION_JOINTS_RAD)
    waypoints = interpolated_joint_waypoints(
        HOME, target, max_segment_joint_rad=math.radians(12.0)
    )
    previous = HOME
    assert len(waypoints) == 6
    for waypoint in waypoints:
        assert joint_travel(previous, waypoint).worst_rad <= math.radians(12.0)
        previous = np.asarray(waypoint)
    assert waypoints[-1] == pytest.approx(OBSERVATION_JOINTS_RAD)


def test_nominal_home_to_observation_path_passes_workspace_gates():
    result = validate_observation_path(_arm(), HOME)
    assert result.safe, result.reason
    assert result.min_tcp_height_m > 0.32
    assert result.max_tcp_reach_m < 0.70
