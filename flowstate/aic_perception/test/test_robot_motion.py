from __future__ import annotations

import math

import numpy as np
import pytest

from aic_perception.robot_motion import (
    ControllerPose,
    RobotMotion,
    interpolated_poses,
    interpolated_positions,
    minimum_jerk,
    normalize_quaternion,
    quaternion_angular_distance,
    quaternion_slerp,
)


def test_minimum_jerk_is_clamped_monotonic_and_has_smooth_endpoints():
    samples = np.asarray([minimum_jerk(value) for value in np.linspace(0, 1, 101)])
    assert minimum_jerk(-1.0) == 0.0
    assert minimum_jerk(2.0) == 1.0
    assert samples[0] == 0.0
    assert samples[-1] == 1.0
    assert np.all(np.diff(samples) >= 0.0)
    assert samples[1] < 2e-5
    assert 1.0 - samples[-2] < 2e-5


def test_interpolated_positions_preserve_endpoints_without_overshoot():
    positions = interpolated_positions((0.0, 1.0, 2.0), (0.02, 0.98, 2.01), 21)
    np.testing.assert_allclose(positions[0], (0.0, 1.0, 2.0))
    np.testing.assert_allclose(positions[-1], (0.02, 0.98, 2.01))
    values = np.asarray(positions)
    assert np.all((0.0 <= values[:, 0]) & (values[:, 0] <= 0.02))
    assert np.all((0.98 <= values[:, 1]) & (values[:, 1] <= 1.0))


def test_interpolated_positions_reject_bad_inputs():
    with pytest.raises(ValueError):
        interpolated_positions((0, 0, 0), (1, 1, 1), 1)
    with pytest.raises(ValueError):
        interpolated_positions((0, 0), (1, 1, 1), 2)


def test_quaternion_normalization_and_distance_treat_sign_as_equivalent():
    np.testing.assert_allclose(normalize_quaternion((0.0, 0.0, 0.0, 2.0)), (0, 0, 0, 1))
    assert quaternion_angular_distance(
        (0, 0, 0, 1), (0, 0, 0, -1)
    ) == pytest.approx(0.0)
    assert quaternion_angular_distance(
        (0, 0, 0, 1), (0, 0, 1, 0)
    ) == pytest.approx(math.pi)
    with pytest.raises(ValueError):
        normalize_quaternion((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        normalize_quaternion((0.0, np.nan, 0.0, 1.0))


def test_quaternion_slerp_uses_shortest_path_and_preserves_unit_norm():
    identity = (0.0, 0.0, 0.0, 1.0)
    target = (0.0, 0.0, -math.sin(math.pi / 4), -math.cos(math.pi / 4))
    midpoint = quaternion_slerp(identity, target, 0.5)
    assert np.linalg.norm(midpoint) == pytest.approx(1.0)
    assert quaternion_angular_distance(identity, midpoint) == pytest.approx(math.pi / 4)
    assert quaternion_angular_distance(midpoint, target) == pytest.approx(math.pi / 4)
    assert quaternion_slerp(identity, target, -1.0) == pytest.approx(identity)
    assert quaternion_angular_distance(
        quaternion_slerp(identity, target, 2.0), target
    ) == pytest.approx(0.0)


def test_interpolated_poses_use_minimum_jerk_for_translation_and_rotation():
    poses = interpolated_poses(
        (0.0, 0.0, 0.0),
        (0.02, -0.01, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8)),
        5,
    )
    np.testing.assert_allclose(poses[0][0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(poses[-1][0], (0.02, -0.01, 0.0))
    assert quaternion_angular_distance(
        poses[0][1], (0.0, 0.0, 0.0, 1.0)
    ) == pytest.approx(0.0)
    assert quaternion_angular_distance(
        poses[-1][1], (0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))
    ) == pytest.approx(0.0)
    assert poses[1][0][0] == pytest.approx(0.02 * minimum_jerk(0.25))
    assert quaternion_angular_distance(poses[0][1], poses[1][1]) == pytest.approx(
        (math.pi / 4) * minimum_jerk(0.25)
    )


def test_retreat_restores_translation_and_orientation(monkeypatch):
    motion = RobotMotion.__new__(RobotMotion)
    start = ControllerPose(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        speed_mps=0.0,
        received_at=0.0,
    )
    current = ControllerPose(
        position=(0.01, -0.02, 0.0),
        orientation=(0.0, math.sin(0.1), 0.0, math.cos(0.1)),
        speed_mps=0.0,
        received_at=0.0,
    )
    captured = {}
    monkeypatch.setattr(motion, "_current_state", lambda _: current)

    def capture_profile(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(motion, "_publish_profile", capture_profile)
    motion._retreat_to_step_start(start, publish_hz=20.0)

    assert captured["args"][0] == current.position
    assert captured["args"][1] == start.position
    assert captured["args"][2] == current.orientation
    assert captured["args"][3] == start.orientation
    assert captured["args"][5] == 20.0


def test_move_smooth_settles_rotation_without_angular_velocity(monkeypatch):
    class Publisher:
        def __init__(self):
            self.messages = []

        @staticmethod
        def get_subscription_count():
            return 1

        def publish(self, message):
            self.messages.append(message)

    class CameraRig:
        @staticmethod
        def latest_force_xyz(max_age_sec):
            assert max_age_sec == 0.25
            return (0.0, 0.0, 0.0)

    motion = RobotMotion.__new__(RobotMotion)
    motion._publisher = Publisher()
    motion._camera_rig = CameraRig()
    monkeypatch.setattr(motion, "_ensure_cartesian_mode", lambda _: (True, ""))
    monkeypatch.setattr(
        motion, "_command", lambda position, orientation: (position, orientation)
    )
    monkeypatch.setattr("aic_perception.robot_motion.time.sleep", lambda _: None)

    start = ControllerPose(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        speed_mps=0.0,
        received_at=0.0,
    )
    target_orientation = (0.0, math.sin(0.1), 0.0, math.cos(0.1))
    settled = ControllerPose(
        position=start.position,
        orientation=target_orientation,
        speed_mps=0.0,
        received_at=0.0,
        angular_speed_radps=None,
    )
    states = iter((start, settled, settled, settled))
    monkeypatch.setattr(motion, "_current_state", lambda _: next(states))
    captured = {}

    def complete_profile(*args, **kwargs):
        captured["args"] = args
        return True

    monkeypatch.setattr(motion, "_publish_profile", complete_profile)
    outcome = motion.move_smooth(
        start.position,
        target_orientation=target_orientation,
        max_speed_mps=0.025,
        max_angular_speed_radps=0.25,
        publish_hz=20.0,
        settle_tolerance_m=0.003,
        settle_angular_tolerance_rad=0.02,
        settle_angular_speed_radps=0.05,
        timeout_sec=2.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert outcome.success
    assert outcome.distance_m == pytest.approx(0.0)
    assert outcome.angular_distance_rad == pytest.approx(0.2)
    assert captured["args"][2] == start.orientation
    assert quaternion_angular_distance(
        captured["args"][3], target_orientation
    ) == pytest.approx(0.0)


def test_force_guard_checks_absolute_and_change_from_baseline():
    assert not RobotMotion._force_exceeded((1.0, 2.0, 1.0), (1.0, 2.0, 1.0), 12.0, 5.0)
    assert RobotMotion._force_exceeded((12.0, 0.0, 0.0), (0.0, 0.0, 0.0), 12.0, 5.0)
    assert RobotMotion._force_exceeded((6.1, 0.0, 0.0), (1.0, 0.0, 0.0), 12.0, 5.0)
    assert not RobotMotion._force_exceeded(None, None, 12.0, 5.0)
