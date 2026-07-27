from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from aic_perception.robot_motion import (
    base_yaw_target_pose,
    contact_force_n,
    force_guard_tripped,
    FORCE_RUNAWAY_CEILING_N,
    STATIC_FORCE_ENVELOPE_HI_N,
    STATIC_FORCE_ENVELOPE_LO_N,
    ControllerPose,
    JointReference,
    MotionFailure,
    MotionOutcome,
    RobotMotion,
    interpolated_joint_positions,
    interpolated_poses,
    interpolated_positions,
    minimum_jerk,
    normalize_quaternion,
    quaternion_angular_distance,
    quaternion_slerp,
)


def test_motion_outcome_target_reached_defaults_false_for_compatibility():
    outcome = MotionOutcome(True, "legacy safe completion")

    assert outcome.success
    assert not outcome.target_reached


def test_base_yaw_target_pose_rotates_position_and_orientation_together():
    position, orientation = base_yaw_target_pose(
        (1.0, 0.0, 0.5),
        (0.0, 0.0, 0.0, 1.0),
        math.pi / 2.0,
    )

    assert position == pytest.approx((0.0, 1.0, 0.5), abs=1e-9)
    assert orientation == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    assert quaternion_angular_distance(
        (0.0, 0.0, 0.0, 1.0), orientation
    ) == pytest.approx(math.pi / 2.0)


def test_base_yaw_target_pose_is_reversible_and_validates_inputs():
    start_position = (0.55, -0.25, 0.8)
    start_orientation = normalize_quaternion((0.2, -0.3, 0.1, 0.9))
    moved_position, moved_orientation = base_yaw_target_pose(
        start_position, start_orientation, 0.05
    )
    restored_position, restored_orientation = base_yaw_target_pose(
        moved_position, moved_orientation, -0.05
    )

    assert restored_position == pytest.approx(start_position, abs=1e-12)
    assert quaternion_angular_distance(
        restored_orientation, start_orientation
    ) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        base_yaw_target_pose((0.0, 0.0), start_orientation, 0.1)
    with pytest.raises(ValueError):
        base_yaw_target_pose(start_position, start_orientation, math.nan)


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


def test_joint_profile_changes_only_the_requested_joint_without_overshoot():
    samples = interpolated_joint_positions(
        (0.2, -0.4, 0.6, -0.8, 1.0, -1.2),
        (0.3, -0.4, 0.6, -0.8, 1.0, -1.2),
        21,
    )
    values = np.asarray(samples)
    assert values[0, 0] == pytest.approx(0.2)
    assert values[-1, 0] == pytest.approx(0.3)
    assert np.all((0.2 <= values[:, 0]) & (values[:, 0] <= 0.3))
    np.testing.assert_allclose(
        values[:, 1:], np.repeat(values[:1, 1:], len(values), axis=0)
    )


def test_callbacks_pair_measured_named_joints_with_controller_mode():
    vector = lambda **values: SimpleNamespace(x=0.0, y=0.0, z=0.0, **values)
    motion = RobotMotion.__new__(RobotMotion)
    motion._condition = threading.Condition()
    motion._state = None
    motion._joint_reference = None
    motion._target_mode = None
    motion._target_mode_received_at = -math.inf
    message = SimpleNamespace(
        tcp_pose=SimpleNamespace(
            position=vector(),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        tcp_velocity=SimpleNamespace(linear=vector(), angular=vector()),
        target_mode=SimpleNamespace(mode=np.uint8(2)),
    )

    motion._on_controller_state(message)

    # ControllerState carries the command mode but its reference positions are
    # empty in impedance control.  The actual arm pose comes from /joint_states
    # and is reordered by name into controller command order.
    assert motion._joint_reference is None
    motion._on_joint_state(
        SimpleNamespace(
            name=[
                "wrist_3_joint",
                "elbow_joint",
                "shoulder_pan_joint",
                "gripper_joint",
                "wrist_1_joint",
                "shoulder_lift_joint",
                "wrist_2_joint",
            ],
            position=[-1.2, 0.6, 0.1, 0.02, -0.8, -0.4, 1.0],
        )
    )

    assert motion._joint_reference is not None
    assert motion._joint_reference.positions == pytest.approx(
        (0.1, -0.4, 0.6, -0.8, 1.0, -1.2)
    )
    assert motion._joint_reference.target_mode == 2
    assert type(motion._joint_reference.target_mode) is int


def test_controller_handoff_holds_the_reported_mode_at_measured_state():
    motion = RobotMotion.__new__(RobotMotion)
    calls = []
    motion._reported_target_mode = lambda: 2
    motion._publish_mode_hold_command = lambda mode: calls.append(mode) or True

    assert motion.prepare_controller_handoff()
    assert calls == [2]


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
        def wait_for_force_xyz(timeout_sec, max_age_sec):
            assert timeout_sec == 1.0
            assert max_age_sec == 0.5
            return (0.0, 0.0, 0.0)

        @staticmethod
        def latest_force_xyz(max_age_sec):
            assert max_age_sec == 0.5
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
    assert outcome.target_reached
    assert outcome.distance_m == pytest.approx(0.0)
    assert outcome.angular_distance_rad == pytest.approx(0.2)
    assert captured["args"][2] == start.orientation
    assert quaternion_angular_distance(
        captured["args"][3], target_orientation
    ) == pytest.approx(0.0)


def test_move_smooth_replans_from_safe_measured_pose_when_controller_stops_short(
    monkeypatch,
):
    class Publisher:
        def __init__(self):
            self.messages = []

        @staticmethod
        def get_subscription_count():
            return 1

        def publish(self, message):
            self.messages.append(message)

    motion = RobotMotion.__new__(RobotMotion)
    motion._publisher = Publisher()
    motion._camera_rig = SimpleNamespace(
        wait_for_force_xyz=lambda **_kwargs: (0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(motion, "_ensure_cartesian_mode", lambda _: (True, ""))
    monkeypatch.setattr(
        motion, "_command", lambda position, orientation: (position, orientation)
    )
    monkeypatch.setattr(motion, "_publish_profile", lambda *args, **kwargs: True)

    start = ControllerPose(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        speed_mps=0.0,
        received_at=0.0,
    )
    stopped_short = ControllerPose(
        position=(0.004, 0.0, 0.0),
        orientation=start.orientation,
        speed_mps=0.0,
        received_at=0.0,
        angular_speed_radps=0.0,
    )
    states = iter((start, stopped_short))
    monkeypatch.setattr(motion, "_current_state", lambda _: next(states))
    times = iter((0.0, 1.0))
    monkeypatch.setattr(
        "aic_perception.robot_motion.time.monotonic", lambda: next(times)
    )

    outcome = motion.move_smooth(
        (0.01, 0.0, 0.0),
        target_orientation=start.orientation,
        max_speed_mps=0.025,
        publish_hz=20.0,
        settle_tolerance_m=0.001,
        timeout_sec=0.5,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert outcome.success
    assert not outcome.target_reached
    assert "replanning" in outcome.message
    assert outcome.distance_m == pytest.approx(0.004)
    assert motion._publisher.messages[-1][0] == stopped_short.position


def test_move_smooth_reverses_when_stopped_pose_misses_requested_orientation(
    monkeypatch,
):
    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

        @staticmethod
        def publish(_message):
            pass

    motion = RobotMotion.__new__(RobotMotion)
    motion._publisher = Publisher()
    motion._camera_rig = SimpleNamespace(
        wait_for_force_xyz=lambda **_kwargs: (0.0, 0.0, 0.0),
        latest_force_xyz=lambda **_kwargs: (0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(motion, "_ensure_cartesian_mode", lambda _: (True, ""))
    monkeypatch.setattr(
        motion, "_command", lambda position, orientation: (position, orientation)
    )
    monkeypatch.setattr(motion, "_publish_profile", lambda *args, **kwargs: True)

    start = ControllerPose(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        speed_mps=0.0,
        received_at=0.0,
    )
    stopped_wrong_orientation = ControllerPose(
        position=(0.004, 0.0, 0.0),
        orientation=start.orientation,
        speed_mps=0.0,
        received_at=0.0,
        angular_speed_radps=0.0,
    )
    states = iter((start, stopped_wrong_orientation))
    monkeypatch.setattr(motion, "_current_state", lambda _: next(states))
    times = iter((0.0, 1.0))
    monkeypatch.setattr(
        "aic_perception.robot_motion.time.monotonic", lambda: next(times)
    )
    retreats = []
    monkeypatch.setattr(
        motion,
        "_retreat_to_step_start",
        lambda saved, hz: retreats.append((saved, hz)),
    )
    target_orientation = (0.0, math.sin(0.1), 0.0, math.cos(0.1))

    outcome = motion.move_smooth(
        (0.01, 0.0, 0.0),
        target_orientation=target_orientation,
        max_speed_mps=0.025,
        max_angular_speed_radps=0.5,
        publish_hz=20.0,
        settle_tolerance_m=0.001,
        settle_angular_tolerance_rad=0.02,
        timeout_sec=0.5,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert not outcome.success
    assert not outcome.target_reached
    assert "did not settle" in outcome.message
    assert retreats and retreats[0][0] is start


def test_move_smooth_waits_for_fresh_force_before_publishing(monkeypatch):
    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

    motion = RobotMotion.__new__(RobotMotion)
    motion._publisher = Publisher()
    motion._camera_rig = SimpleNamespace(
        wait_for_force_xyz=lambda **_kwargs: None
    )
    monkeypatch.setattr(motion, "_ensure_cartesian_mode", lambda _: (True, ""))
    monkeypatch.setattr(
        motion,
        "_current_state",
        lambda _: ControllerPose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            speed_mps=0.0,
            received_at=0.0,
        ),
    )
    monkeypatch.setattr(
        motion,
        "_publish_profile",
        lambda *_args, **_kwargs: pytest.fail(
            "Cartesian setpoints must not publish without fresh force"
        ),
    )

    outcome = motion.move_smooth(
        (0.01, 0.0, 0.0),
        max_speed_mps=0.025,
        publish_hz=20.0,
        settle_tolerance_m=0.001,
        timeout_sec=2.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert not outcome.success
    assert "fresh wrist-force" in outcome.message


@pytest.mark.parametrize(
    ("method_name", "joint_index"),
    (("move_joint1_yaw", 0), ("move_joint6_yaw", 5)),
)
def test_single_joint_yaw_preserves_all_other_controller_references(
    monkeypatch, method_name, joint_index
):
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
        def wait_for_force_xyz(timeout_sec, max_age_sec):
            assert timeout_sec == 1.0
            assert max_age_sec == 0.5
            return (0.0, 0.0, 0.0)

        @staticmethod
        def latest_force_xyz(max_age_sec):
            assert max_age_sec == 0.5
            return (0.0, 0.0, 0.0)

    motion = RobotMotion.__new__(RobotMotion)
    motion._joint_publisher = Publisher()
    motion._camera_rig = CameraRig()
    motion._mode_joint = 2
    monkeypatch.setattr(motion, "_ensure_joint_mode", lambda _: (True, ""))
    restores = []
    monkeypatch.setattr(
        motion,
        "_ensure_cartesian_mode",
        lambda timeout: (restores.append(timeout) is None, ""),
    )
    monkeypatch.setattr(motion, "_joint_command", lambda positions: positions)
    monkeypatch.setattr("aic_perception.robot_motion.time.sleep", lambda _: None)

    start = JointReference(
        positions=(0.2, -0.4, 0.6, -0.8, 1.0, -1.2),
        tcp_speed_mps=0.0,
        received_at=1.0,
        target_mode=1,
    )
    joint_mode_start = JointReference(start.positions, 0.0, 2.0, target_mode=2)
    target_values = list(start.positions)
    target_values[joint_index] += 0.06
    target = tuple(target_values)
    references = iter(
        [joint_mode_start]
        + [
            JointReference(target, 0.0, stamp, target_mode=2)
            for stamp in (3.0, 4.0, 5.0)
        ]
    )

    def current_reference(_timeout, target_mode=None):
        return joint_mode_start if target_mode == 2 else start

    monkeypatch.setattr(motion, "_current_joint_reference", current_reference)
    # An aged/unknown asynchronous controller-mode report must not invalidate
    # an otherwise fresh measured joint stream after mode was confirmed.
    monkeypatch.setattr(motion, "_reported_target_mode", lambda: None)
    monkeypatch.setattr(
        motion,
        "_next_joint_reference",
        lambda _after, _timeout, target_mode=None: next(references),
    )
    captured = {}

    def complete_profile(*args, **kwargs):
        captured["args"] = args
        return True, args[1]

    monkeypatch.setattr(motion, "_publish_joint_profile", complete_profile)
    outcome = getattr(motion, method_name)(
        0.06,
        max_speed_radps=0.12,
        publish_hz=20.0,
        timeout_sec=2.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert outcome.success
    assert outcome.target_reached
    assert outcome.joint_distance_rad == pytest.approx(0.06)
    assert captured["args"][0] == start.positions
    assert captured["args"][1] == pytest.approx(target)
    assert captured["args"][2] == pytest.approx(1.875 * 0.06 / 0.12)
    assert captured["args"][1][joint_index] == pytest.approx(
        start.positions[joint_index] + 0.06
    )
    assert all(
        captured["args"][1][index] == pytest.approx(value)
        for index, value in enumerate(start.positions)
        if index != joint_index
    )
    assert motion._joint_publisher.messages[-1] == pytest.approx(target)
    assert restores == [2.0]


def test_joint_profile_outlives_mode_timestamp_when_joint_feedback_is_fresh(
    monkeypatch,
):
    """Regression for the live 0.525 s reversal of a 0.981 s J1 profile."""

    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

        @staticmethod
        def publish(_message):
            pass

    class CameraRig:
        @staticmethod
        def wait_for_force_xyz(timeout_sec, max_age_sec):
            return (0.0, 0.0, 0.0)

        @staticmethod
        def latest_force_xyz(max_age_sec):
            return (0.0, 0.0, 0.0)

    motion = RobotMotion.__new__(RobotMotion)
    motion._joint_publisher = Publisher()
    motion._camera_rig = CameraRig()
    motion._mode_joint = 2
    monkeypatch.setattr(motion, "_ensure_joint_mode", lambda _: (True, ""))
    monkeypatch.setattr(
        motion, "_ensure_cartesian_mode", lambda _: (True, "")
    )
    monkeypatch.setattr(motion, "_joint_command", lambda positions: positions)
    monkeypatch.setattr("aic_perception.robot_motion.time.sleep", lambda _: None)

    start = JointReference(
        (0.2, -0.4, 0.6, -0.8, 1.0, -1.2),
        0.0,
        1.0,
        target_mode=None,
    )
    confirmed = JointReference(start.positions, 0.0, 2.0, target_mode=2)
    target = list(start.positions)
    target[0] += 0.0628
    target = tuple(target)
    settling = iter(
        JointReference(target, 0.0, stamp, target_mode=None)
        for stamp in (3.0, 4.0, 5.0)
    )

    # Fresh measured joints continue while the independent mode report has
    # aged out to None.  This is the exact asynchronous condition from r20.
    monkeypatch.setattr(
        motion,
        "_current_joint_reference",
        lambda _timeout, target_mode=None: start if target_mode is None else None,
    )
    monkeypatch.setattr(motion, "_reported_target_mode", lambda: None)

    def next_reference(_after, _timeout, target_mode=None):
        if target_mode == 2:
            return confirmed
        return next(settling)

    monkeypatch.setattr(motion, "_next_joint_reference", next_reference)

    def profile(_start, profile_target, duration, _hz, stop):
        assert duration == pytest.approx(1.875 * 0.0628 / 0.12)
        assert duration > 0.5
        for _ in range(30):
            assert not stop()
        return True, profile_target

    monkeypatch.setattr(motion, "_publish_joint_profile", profile)

    outcome = motion.move_joint1_yaw(
        0.0628,
        max_speed_radps=0.12,
        publish_hz=20.0,
        timeout_sec=3.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert outcome.success
    assert outcome.target_reached
    assert outcome.failure is None


def test_joint_profile_reverses_on_explicit_fresh_mode_change(monkeypatch):
    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

    class CameraRig:
        @staticmethod
        def wait_for_force_xyz(timeout_sec, max_age_sec):
            return (0.0, 0.0, 0.0)

        @staticmethod
        def latest_force_xyz(max_age_sec):
            return (0.0, 0.0, 0.0)

    motion = RobotMotion.__new__(RobotMotion)
    motion._joint_publisher = Publisher()
    motion._camera_rig = CameraRig()
    motion._mode_joint = 2
    start = JointReference((0.2, -0.4, 0.6), 0.0, 1.0)
    confirmed = JointReference(start.positions, 0.0, 2.0, target_mode=2)
    monkeypatch.setattr(motion, "_ensure_joint_mode", lambda _: (True, ""))
    monkeypatch.setattr(
        motion, "_ensure_cartesian_mode", lambda _: (True, "")
    )
    monkeypatch.setattr(
        motion, "_current_joint_reference", lambda *_args, **_kwargs: start
    )
    monkeypatch.setattr(
        motion, "_next_joint_reference", lambda *_args, **_kwargs: confirmed
    )
    monkeypatch.setattr(motion, "_reported_target_mode", lambda: 1)
    monkeypatch.setattr(
        motion,
        "_publish_joint_profile",
        lambda *args: (not args[4](), args[0]),
    )
    monkeypatch.setattr(motion, "_retreat_joint_target", lambda *args: None)

    outcome = motion.move_joint1_yaw(
        0.05,
        max_speed_radps=0.20,
        publish_hz=20.0,
        timeout_sec=2.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert not outcome.success
    assert outcome.failure is MotionFailure.MODE_CHANGED
    assert "left joint target mode" in outcome.message


def test_joint1_yaw_rejects_speed_above_documented_limit():
    motion = RobotMotion.__new__(RobotMotion)
    outcome = motion.move_joint1_yaw(
        0.05,
        max_speed_radps=0.201,
        publish_hz=20.0,
        timeout_sec=2.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )
    assert not outcome.success
    assert "0.20" in outcome.message
    assert outcome.failure is MotionFailure.INVALID_REQUEST


def test_full_joint_target_settles_exact_measured_vector(monkeypatch):
    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

        @staticmethod
        def publish(_message):
            pass

    class CameraRig:
        @staticmethod
        def wait_for_force_xyz(timeout_sec, max_age_sec):
            return (0.0, 0.0, 0.0)

        @staticmethod
        def latest_force_xyz(max_age_sec):
            return (0.0, 0.0, 0.0)

    motion = RobotMotion.__new__(RobotMotion)
    motion._joint_publisher = Publisher()
    motion._camera_rig = CameraRig()
    motion._mode_joint = 2
    start = JointReference(
        (0.2, -0.4, 0.6, -0.8, 1.0, -1.2),
        0.0,
        1.0,
        target_mode=1,
    )
    joint_start = JointReference(start.positions, 0.0, 2.0, target_mode=2)
    target = (0.25, -0.35, 0.55, -0.75, 0.95, -1.15)
    settled = JointReference(target, 0.0, 3.0, target_mode=2)
    monkeypatch.setattr(motion, "_ensure_joint_mode", lambda _: (True, ""))
    monkeypatch.setattr(
        motion, "_ensure_cartesian_mode", lambda _: (True, "")
    )
    monkeypatch.setattr(
        motion, "_current_joint_reference", lambda *_args, **_kwargs: start
    )
    monkeypatch.setattr(
        motion,
        "_next_joint_reference",
        lambda _after, _timeout, target_mode=None: (
            joint_start if target_mode == 2 else settled
        ),
    )
    monkeypatch.setattr(
        motion,
        "_publish_joint_profile",
        lambda _start, requested, *_args: (True, requested),
    )
    monkeypatch.setattr(motion, "_reported_target_mode", lambda: 2)
    monkeypatch.setattr(
        motion, "_joint_command", lambda positions: positions
    )

    outcome = motion.move_joint_target(
        target,
        max_speed_radps=0.20,
        publish_hz=50.0,
        timeout_sec=2.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert outcome.success and outcome.target_reached
    assert outcome.joint_distance_rad == pytest.approx(0.05)


def test_full_joint_target_force_abort_reverses_complete_vector(monkeypatch):
    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

    class CameraRig:
        @staticmethod
        def wait_for_force_xyz(timeout_sec, max_age_sec):
            # Above the untared free-space envelope (see
            # STATIC_FORCE_ENVELOPE_HI_N): 19 N is not contact on this sensor.
            return (34.0, 0.0, 0.0)

        @staticmethod
        def latest_force_xyz(max_age_sec):
            # Above the untared free-space envelope (see
            # STATIC_FORCE_ENVELOPE_HI_N): 19 N is not contact on this sensor.
            return (34.0, 0.0, 0.0)

    motion = RobotMotion.__new__(RobotMotion)
    motion._joint_publisher = Publisher()
    motion._camera_rig = CameraRig()
    motion._mode_joint = 2
    start = JointReference(
        (0.2, -0.4, 0.6, -0.8, 1.0, -1.2),
        0.0,
        1.0,
        target_mode=1,
    )
    joint_start = JointReference(start.positions, 0.0, 2.0, target_mode=2)
    monkeypatch.setattr(motion, "_ensure_joint_mode", lambda _: (True, ""))
    monkeypatch.setattr(
        motion, "_ensure_cartesian_mode", lambda _: (True, "")
    )
    monkeypatch.setattr(
        motion, "_current_joint_reference", lambda *_args, **_kwargs: start
    )
    monkeypatch.setattr(
        motion,
        "_next_joint_reference",
        lambda _after, _timeout, target_mode=None: joint_start,
    )
    monkeypatch.setattr(motion, "_reported_target_mode", lambda: 2)

    def guarded_profile(start_positions, _target, _duration, _hz, stop):
        assert stop()
        return False, start_positions

    monkeypatch.setattr(
        motion, "_publish_joint_profile", guarded_profile
    )
    reversed_to = {}
    monkeypatch.setattr(
        motion,
        "_retreat_joint_target",
        lambda saved, last, speed, hz: reversed_to.update(
            saved=saved.positions, last=last, speed=speed, hz=hz
        ),
    )

    outcome = motion.move_joint_target(
        (0.25, -0.35, 0.55, -0.75, 0.95, -1.15),
        max_speed_radps=0.20,
        publish_hz=20.0,
        timeout_sec=2.0,
        baseline_force_xyz=(13.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert not outcome.success and outcome.force_abort
    assert outcome.failure is MotionFailure.FORCE_LIMIT
    assert reversed_to["saved"] == start.positions
    assert reversed_to["last"] == start.positions


def test_joint1_yaw_requires_newer_reference_confirming_joint_mode(monkeypatch):
    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

    motion = RobotMotion.__new__(RobotMotion)
    motion._joint_publisher = Publisher()
    motion._mode_joint = 2
    pre_switch = JointReference(
        (0.2, -0.4, 0.6), 0.0, 4.0, target_mode=1
    )
    monkeypatch.setattr(
        motion, "_current_joint_reference", lambda *_args, **_kwargs: pre_switch
    )
    monkeypatch.setattr(motion, "_ensure_joint_mode", lambda _: (True, ""))
    restored = []
    monkeypatch.setattr(
        motion,
        "_ensure_cartesian_mode",
        lambda timeout: (restored.append(timeout) is None, ""),
    )

    def no_joint_confirmation(after, timeout, target_mode=None):
        assert after == pre_switch.received_at
        assert target_mode == 2
        return None

    monkeypatch.setattr(
        motion, "_next_joint_reference", no_joint_confirmation
    )
    monkeypatch.setattr(
        motion,
        "_publish_joint_profile",
        lambda *args, **kwargs: pytest.fail(
            "must not command before a fresh joint-mode reference"
        ),
    )

    outcome = motion.move_joint1_yaw(
        0.05,
        max_speed_radps=0.10,
        publish_hz=20.0,
        timeout_sec=2.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert not outcome.success
    assert "newer" in outcome.message and "joint target mode" in outcome.message
    assert restored == [2.0]


def test_joint1_yaw_force_guard_reverses_to_entire_starting_target(monkeypatch):
    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

    class CameraRig:
        @staticmethod
        def wait_for_force_xyz(timeout_sec, max_age_sec):
            # Above the untared free-space envelope (see
            # STATIC_FORCE_ENVELOPE_HI_N): 19 N is not contact on this sensor.
            return (34.0, 0.0, 0.0)

        @staticmethod
        def latest_force_xyz(max_age_sec):
            # Above the untared free-space envelope (see
            # STATIC_FORCE_ENVELOPE_HI_N): 19 N is not contact on this sensor.
            return (34.0, 0.0, 0.0)

    motion = RobotMotion.__new__(RobotMotion)
    motion._joint_publisher = Publisher()
    motion._camera_rig = CameraRig()
    motion._mode_joint = 2
    monkeypatch.setattr(motion, "_ensure_joint_mode", lambda _: (True, ""))
    monkeypatch.setattr(motion, "_ensure_cartesian_mode", lambda _: (True, ""))
    start = JointReference(
        positions=(0.2, -0.4, 0.6, -0.8, 1.0, -1.2),
        tcp_speed_mps=0.0,
        received_at=1.0,
        target_mode=1,
    )
    joint_mode_start = JointReference(start.positions, 0.0, 2.0, target_mode=2)
    monkeypatch.setattr(
        motion,
        "_current_joint_reference",
        lambda _timeout, target_mode=None: (
            joint_mode_start if target_mode == 2 else start
        ),
    )
    monkeypatch.setattr(
        motion,
        "_next_joint_reference",
        lambda _after, _timeout, target_mode=None: joint_mode_start,
    )

    def guarded_profile(*args):
        stop = args[4]
        assert stop()
        return False, args[0]

    monkeypatch.setattr(motion, "_publish_joint_profile", guarded_profile)
    reversed_to = {}
    monkeypatch.setattr(
        motion,
        "_retreat_joint_target",
        lambda saved, last, speed, hz: reversed_to.update(
            saved=saved.positions, last=last, speed=speed, hz=hz
        ),
    )
    outcome = motion.move_joint1_yaw(
        -0.05,
        max_speed_radps=0.10,
        publish_hz=20.0,
        timeout_sec=2.0,
        baseline_force_xyz=(13.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert not outcome.success and outcome.force_abort
    assert "reversed" in outcome.message
    assert reversed_to["saved"] == start.positions
    assert reversed_to["last"] == start.positions


def test_joint1_yaw_settle_timeout_reverses_and_restores_cartesian(monkeypatch):
    class Publisher:
        @staticmethod
        def get_subscription_count():
            return 1

        @staticmethod
        def publish(_message):
            pass

    class CameraRig:
        @staticmethod
        def wait_for_force_xyz(timeout_sec, max_age_sec):
            return (0.0, 0.0, 0.0)

        @staticmethod
        def latest_force_xyz(max_age_sec):
            return (0.0, 0.0, 0.0)

    motion = RobotMotion.__new__(RobotMotion)
    motion._joint_publisher = Publisher()
    motion._camera_rig = CameraRig()
    motion._mode_joint = 2
    monkeypatch.setattr(motion, "_ensure_joint_mode", lambda _: (True, ""))
    restored = []
    monkeypatch.setattr(
        motion,
        "_ensure_cartesian_mode",
        lambda timeout: (restored.append(timeout) is None, ""),
    )
    start = JointReference((0.2, -0.4, 0.6), 0.0, 1.0, target_mode=1)
    joint_mode_start = JointReference(start.positions, 0.0, 2.0, target_mode=2)
    monkeypatch.setattr(
        motion,
        "_current_joint_reference",
        lambda _timeout, target_mode=None: (
            joint_mode_start if target_mode == 2 else start
        ),
    )
    monkeypatch.setattr(
        motion,
        "_next_joint_reference",
        lambda _after, _timeout, target_mode=None: joint_mode_start,
    )
    monkeypatch.setattr(
        motion,
        "_publish_joint_profile",
        lambda *args, **kwargs: (True, args[1]),
    )
    reversed_to = []
    monkeypatch.setattr(
        motion,
        "_retreat_joint_target",
        lambda saved, last, speed, hz: reversed_to.append(
            (saved.positions, last, speed, hz)
        ),
    )
    times = iter((0.0, 10.0))
    monkeypatch.setattr(
        "aic_perception.robot_motion.time.monotonic", lambda: next(times)
    )

    outcome = motion.move_joint1_yaw(
        0.06,
        max_speed_radps=0.12,
        publish_hz=20.0,
        timeout_sec=2.0,
        baseline_force_xyz=(0.0, 0.0, 0.0),
        max_force_n=18.0,
        force_delta_n=5.0,
        cancelled=lambda: False,
    )

    assert not outcome.success
    assert "timeout" in outcome.message and "reversed" in outcome.message
    assert reversed_to[0][0] == start.positions
    assert reversed_to[0][1][0] == pytest.approx(0.26)
    assert restored == [2.0]


def test_current_joint1_is_none_until_full_reference_is_fresh(monkeypatch):
    motion = RobotMotion.__new__(RobotMotion)
    monkeypatch.setattr(motion, "_current_joint_reference", lambda *_args, **_kwargs: None)
    assert motion.current_joint1() is None
    reference = JointReference((0.42, 1.0), 0.0, 1.0)
    monkeypatch.setattr(
        motion,
        "_current_joint_reference",
        lambda *_args, **_kwargs: reference,
    )
    assert motion.current_joint1() == pytest.approx(0.42)


def test_force_guard_checks_unexplained_force_not_a_raw_ceiling():
    """A caller-supplied ceiling below the static envelope is not honoured.

    It cannot be: on an untared sensor a 12 N raw limit only reports which way
    the wrist is pointing.  Contact is force the static load cannot explain.
    """
    assert not RobotMotion._force_exceeded((1.0, 2.0, 1.0), (1.0, 2.0, 1.0), 12.0, 5.0)
    assert not RobotMotion._force_exceeded((12.0, 0.0, 0.0), (0.0, 0.0, 0.0), 12.0, 5.0)
    assert RobotMotion._force_exceeded(
        (STATIC_FORCE_ENVELOPE_HI_N + 5.0, 0.0, 0.0), (0.0, 0.0, 0.0), 12.0, 5.0
    )
    assert not RobotMotion._force_exceeded(None, None, 12.0, 5.0)


def test_force_guard_ignores_gravity_rotating_in_the_sensor_frame():
    # A J5/J6 reorientation rotates the tool weight in the sensor frame, which
    # moves the vector *and* the magnitude, because the constant bias does not
    # rotate with it.  Neither a vector delta nor a magnitude delta survives
    # that; only force outside the measured free-space envelope does.
    assert not RobotMotion._force_exceeded(
        (0.0, 14.0, 0.0), (14.0, 0.0, 0.0), 18.0, 5.0
    )
    # 13 -> 19 N is an ordinary free-space swing on this sensor and must stay
    # quiet: trusting it is what force-aborted four consecutive live runs.
    assert not RobotMotion._force_exceeded(
        (19.0, 0.0, 0.0), (13.0, 0.0, 0.0), 18.0, 5.0
    )
    # Genuine contact leaves the envelope and must still trip.
    assert RobotMotion._force_exceeded(
        (0.0, STATIC_FORCE_ENVELOPE_HI_N + 5.0, 0.0), (14.0, 0.0, 0.0), 25.0, 5.0
    )


# ---------------------------------------------------------------------------
# Wrist force guard on an untared sensor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_n",
    (13.59, 14.31, 14.73, 17.06, 17.79, 25.72),
)
def test_free_space_hardware_readings_are_not_contact(raw_n):
    """Every one of these tripped a guard on hardware while in free space.

    The wrist FTS is untared: it reports a ~19.7 N constant bias plus a ~6.1 N
    tool weight that rotates in the sensor frame, so the free-space magnitude
    sweeps 13.6-25.7 N with wrist orientation alone.  25.72 N is the reading
    that refused deterministic Stage 1 before it moved a single joint.
    """
    force = (raw_n, 0.0, 0.0)
    assert contact_force_n(force) == 0.0
    assert not force_guard_tripped(force, None, 18.0, 5.0)


def test_reorientation_across_the_whole_envelope_is_not_contact():
    """The old magnitude-delta guard fired here; the swing is free space."""
    low = (STATIC_FORCE_ENVELOPE_LO_N, 0.0, 0.0)
    high = (STATIC_FORCE_ENVELOPE_HI_N, 0.0, 0.0)
    assert not force_guard_tripped(high, low, 18.0, 5.0)
    assert not force_guard_tripped(low, high, 18.0, 5.0)


def test_force_beyond_the_static_envelope_is_contact():
    over = STATIC_FORCE_ENVELOPE_HI_N + 6.0
    assert contact_force_n((over, 0.0, 0.0)) == pytest.approx(6.0)
    assert force_guard_tripped((over, 0.0, 0.0), None, 18.0, 5.0)


def test_a_reading_below_the_envelope_is_not_treated_as_contact():
    """A lightly-loaded or tared sensor sits below the envelope.

    Calling that "contact" would fire the guard on a healthy zero reading, so
    only force *above* the envelope counts.
    """
    under = STATIC_FORCE_ENVELOPE_LO_N - 6.0
    assert contact_force_n((under, 0.0, 0.0)) == 0.0
    assert not force_guard_tripped((under, 0.0, 0.0), None, 18.0, 5.0)
    assert contact_force_n((0.0, 0.0, 0.0)) == 0.0


def test_a_change_larger_than_the_whole_swing_is_contact():
    """Even inside the envelope, no reorientation explains this much change."""
    swing = STATIC_FORCE_ENVELOPE_HI_N - STATIC_FORCE_ENVELOPE_LO_N
    baseline = (STATIC_FORCE_ENVELOPE_LO_N, 0.0, 0.0)
    force = (STATIC_FORCE_ENVELOPE_LO_N + swing + 5.0, 0.0, 0.0)
    assert force_guard_tripped(force, baseline, 18.0, 5.0)


def test_raw_ceiling_below_the_envelope_is_ignored():
    """A ceiling inside the envelope only reports where the wrist points.

    The shipped 18 N absolute limit sits inside [13.6, 25.7] N, so honouring it
    literally is what refused Stage 1 in free space.
    """
    assert STATIC_FORCE_ENVELOPE_LO_N < 18.0 < STATIC_FORCE_ENVELOPE_HI_N
    assert not force_guard_tripped((25.72, 0.0, 0.0), None, 18.0, 5.0)
    # The runaway backstop still applies.
    assert force_guard_tripped(
        (FORCE_RUNAWAY_CEILING_N + 1.0, 0.0, 0.0), None, 18.0, 5.0
    )


def test_missing_or_nonfinite_force_never_trips():
    assert not force_guard_tripped(None, None, 18.0, 5.0)
    assert contact_force_n(None) == 0.0
    assert contact_force_n((float("nan"), 0.0, 0.0)) == 0.0
