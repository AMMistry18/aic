"""Guarded Cartesian motion through the documented AIC controller interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable

import numpy as np

from .config import PerceptionConfig


@dataclass(frozen=True)
class ControllerPose:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    speed_mps: float
    received_at: float
    angular_speed_radps: float | None = None


@dataclass(frozen=True)
class MotionOutcome:
    success: bool
    message: str
    distance_m: float = 0.0
    angular_distance_rad: float = 0.0
    force_abort: bool = False
    cancelled: bool = False


def minimum_jerk(alpha: float) -> float:
    """Quintic interpolation with zero endpoint velocity and acceleration."""
    value = min(1.0, max(0.0, float(alpha)))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def normalize_quaternion(
    quaternion: tuple[float, float, float, float] | np.ndarray,
) -> tuple[float, float, float, float]:
    """Return a finite unit quaternion in ROS ``(x, y, z, w)`` order."""
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("quaternion norm is invalid")
    return tuple(float(value) for value in values / norm)


def quaternion_angular_distance(
    first: tuple[float, float, float, float] | np.ndarray,
    second: tuple[float, float, float, float] | np.ndarray,
) -> float:
    """Shortest orientation distance in radians, treating ``q`` and ``-q`` alike."""
    first_unit = np.asarray(normalize_quaternion(first), dtype=float)
    second_unit = np.asarray(normalize_quaternion(second), dtype=float)
    dot = float(np.clip(abs(np.dot(first_unit, second_unit)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def quaternion_slerp(
    start: tuple[float, float, float, float] | np.ndarray,
    target: tuple[float, float, float, float] | np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Shortest-path spherical interpolation between ROS-order quaternions."""
    start_unit = np.asarray(normalize_quaternion(start), dtype=float)
    target_unit = np.asarray(normalize_quaternion(target), dtype=float)
    value = min(1.0, max(0.0, float(alpha)))

    dot = float(np.dot(start_unit, target_unit))
    if dot < 0.0:
        target_unit = -target_unit
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))

    # Linear interpolation is better conditioned for nearly identical poses.
    if dot > 0.9995:
        interpolated = start_unit + value * (target_unit - start_unit)
        return normalize_quaternion(interpolated)

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    start_scale = math.sin((1.0 - value) * theta) / sin_theta
    target_scale = math.sin(value * theta) / sin_theta
    return normalize_quaternion(start_scale * start_unit + target_scale * target_unit)


def interpolated_positions(
    start: tuple[float, float, float],
    target: tuple[float, float, float],
    samples: int,
) -> list[tuple[float, float, float]]:
    if samples < 2:
        raise ValueError("samples must be at least two")
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if start_array.shape != (3,) or target_array.shape != (3,):
        raise ValueError("positions must have three elements")
    if not np.all(np.isfinite(np.concatenate((start_array, target_array)))):
        raise ValueError("positions must be finite")
    return [
        tuple(
            float(value)
            for value in start_array
            + minimum_jerk(i / (samples - 1)) * (target_array - start_array)
        )
        for i in range(samples)
    ]


def interpolated_poses(
    start_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    start_orientation: tuple[float, float, float, float],
    target_orientation: tuple[float, float, float, float],
    samples: int,
) -> list[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ]
]:
    """Minimum-jerk translation and shortest-path SLERP pose samples."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    positions = interpolated_positions(start_position, target_position, samples)
    start_unit = normalize_quaternion(start_orientation)
    target_unit = normalize_quaternion(target_orientation)
    return [
        (
            position,
            quaternion_slerp(
                start_unit,
                target_unit,
                minimum_jerk(index / (samples - 1)),
            ),
        )
        for index, position in enumerate(positions)
    ]


class RobotMotion:
    """Own one bounded, force-guarded Cartesian controller session."""

    def __init__(self, node: Any, camera_rig: Any, config: PerceptionConfig):
        from aic_control_interfaces.msg import ControllerState, MotionUpdate
        from aic_control_interfaces.srv import ChangeTargetMode
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._node = node
        self._camera_rig = camera_rig
        self._config = config
        self._MotionUpdate = MotionUpdate
        self._ChangeTargetMode = ChangeTargetMode
        self._condition = threading.Condition()
        self._state: ControllerPose | None = None
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = node.create_publisher(
            MotionUpdate, config.pose_command_topic, qos
        )
        self._state_subscription = node.create_subscription(
            ControllerState,
            config.controller_state_topic,
            self._on_controller_state,
            qos,
        )
        self._mode_client = node.create_client(
            ChangeTargetMode, config.change_target_mode_service
        )

    def _on_controller_state(self, message: Any) -> None:
        try:
            pose = message.tcp_pose
            velocity = message.tcp_velocity.linear
            position = (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            )
            orientation = (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
            speed = float(
                np.linalg.norm(
                    np.asarray((velocity.x, velocity.y, velocity.z), dtype=float)
                )
            )
            values = np.asarray((*position, *orientation, speed), dtype=float)
        except (AttributeError, TypeError, ValueError):
            return
        if not np.all(np.isfinite(values)):
            return
        try:
            normalized = normalize_quaternion(orientation)
        except ValueError:
            return
        angular_speed_radps = None
        try:
            angular = message.tcp_velocity.angular
            candidate = float(
                np.linalg.norm(
                    np.asarray((angular.x, angular.y, angular.z), dtype=float)
                )
            )
            if math.isfinite(candidate):
                angular_speed_radps = candidate
        except (AttributeError, TypeError, ValueError):
            # Older bridges may omit angular velocity. Orientation feedback is
            # still sufficient to settle a bounded position-mode rotation.
            pass
        with self._condition:
            self._state = ControllerPose(
                position=position,
                orientation=normalized,
                speed_mps=speed,
                received_at=time.monotonic(),
                angular_speed_radps=angular_speed_radps,
            )
            self._condition.notify_all()

    def _current_state(self, timeout_sec: float) -> ControllerPose | None:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while True:
                if (
                    self._state is not None
                    and time.monotonic() - self._state.received_at <= 0.5
                ):
                    return self._state
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def _ensure_cartesian_mode(self, timeout_sec: float) -> tuple[bool, str]:
        from aic_control_interfaces.msg import TargetMode

        if not self._mode_client.wait_for_service(timeout_sec=timeout_sec):
            return False, "Cartesian target-mode service is unavailable"
        request = self._ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN
        future = self._mode_client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            return False, "timed out switching to Cartesian target mode"
        try:
            response = future.result()
        except Exception as error:  # surfaced as a safe motion failure
            return False, f"Cartesian target-mode request failed: {error}"
        if response is None or not response.success:
            return False, "controller rejected Cartesian target mode"
        return True, ""

    @staticmethod
    def _diagonal(values: tuple[float, ...]) -> list[float]:
        matrix = np.zeros((6, 6), dtype=float)
        np.fill_diagonal(matrix, np.asarray(values, dtype=float))
        return [float(value) for value in matrix.reshape(-1)]

    def _command(
        self,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> Any:
        from aic_control_interfaces.msg import TrajectoryGenerationMode

        message = self._MotionUpdate()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = self._config.base_frame
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            position
        )
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = orientation
        # Conservative free-space impedance. The controller adds its own target
        # clamping, smoothing, tracking-error reset, and torque-level safety.
        message.target_stiffness = self._diagonal(
            (60.0, 60.0, 60.0, 40.0, 40.0, 40.0)
        )
        message.target_damping = self._diagonal(
            (35.0, 35.0, 35.0, 25.0, 25.0, 25.0)
        )
        message.feedforward_wrench_at_tip.force.x = 0.0
        message.feedforward_wrench_at_tip.force.y = 0.0
        message.feedforward_wrench_at_tip.force.z = 0.0
        message.feedforward_wrench_at_tip.torque.x = 0.0
        message.feedforward_wrench_at_tip.torque.y = 0.0
        message.feedforward_wrench_at_tip.torque.z = 0.0
        message.wrench_feedback_gains_at_tip = [0.0] * 6
        message.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_POSITION
        )
        return message

    def _publish_profile(
        self,
        start: tuple[float, float, float],
        target: tuple[float, float, float],
        start_orientation: tuple[float, float, float, float],
        target_orientation: tuple[float, float, float, float],
        duration_sec: float,
        publish_hz: float,
        stop: Callable[[], bool] | None = None,
    ) -> bool:
        samples = max(2, int(math.ceil(duration_sec * publish_hz)) + 1)
        period = 1.0 / publish_hz
        next_tick = time.monotonic()
        for position, orientation in interpolated_poses(
            start,
            target,
            start_orientation,
            target_orientation,
            samples,
        ):
            if stop is not None and stop():
                return False
            self._publisher.publish(self._command(position, orientation))
            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
        return True

    @staticmethod
    def _force_exceeded(
        force_xyz: tuple[float, float, float] | None,
        baseline_xyz: tuple[float, float, float] | None,
        max_force_n: float,
        force_delta_n: float,
    ) -> bool:
        if force_xyz is None:
            return False
        force = np.asarray(force_xyz, dtype=float)
        if float(np.linalg.norm(force)) >= max_force_n:
            return True
        if baseline_xyz is None:
            return False
        baseline = np.asarray(baseline_xyz, dtype=float)
        return float(np.linalg.norm(force - baseline)) >= force_delta_n

    def _retreat_to_step_start(
        self,
        start: ControllerPose,
        publish_hz: float,
    ) -> None:
        current = self._current_state(0.15) or start
        distance = float(
            np.linalg.norm(
                np.asarray(start.position, dtype=float)
                - np.asarray(current.position, dtype=float)
            )
        )
        angular_distance = quaternion_angular_distance(
            current.orientation, start.orientation
        )
        duration = min(
            1.0,
            max(0.2, distance / 0.04, angular_distance / 0.35),
        )
        self._publish_profile(
            current.position,
            start.position,
            current.orientation,
            start.orientation,
            duration,
            publish_hz,
        )

    def move_smooth(
        self,
        target_position: tuple[float, float, float],
        *,
        target_orientation: tuple[float, float, float, float] | None = None,
        max_speed_mps: float,
        max_angular_speed_radps: float = 0.25,
        publish_hz: float,
        settle_tolerance_m: float,
        settle_angular_tolerance_rad: float = 0.02,
        settle_angular_speed_radps: float = 0.05,
        timeout_sec: float,
        baseline_force_xyz: tuple[float, float, float] | None,
        max_force_n: float,
        force_delta_n: float,
        cancelled: Callable[[], bool],
    ) -> MotionOutcome:
        if self._publisher.get_subscription_count() < 1:
            deadline = time.monotonic() + min(timeout_sec, 2.0)
            while (
                self._publisher.get_subscription_count() < 1
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
        if self._publisher.get_subscription_count() < 1:
            return MotionOutcome(False, "Cartesian pose command has no subscriber")

        mode_ok, mode_error = self._ensure_cartesian_mode(min(timeout_sec, 2.0))
        if not mode_ok:
            return MotionOutcome(False, mode_error)
        start = self._current_state(min(timeout_sec, 2.0))
        if start is None:
            return MotionOutcome(False, "no fresh controller TCP state")

        target = np.asarray(target_position, dtype=float)
        start_position = np.asarray(start.position, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return MotionOutcome(False, "target position is invalid")
        orientation_requested = target_orientation is not None
        try:
            normalized_target_orientation = normalize_quaternion(
                start.orientation if target_orientation is None else target_orientation
            )
        except ValueError:
            return MotionOutcome(False, "target orientation is invalid")
        if (
            np.dot(
                np.asarray(start.orientation, dtype=float),
                np.asarray(normalized_target_orientation, dtype=float),
            )
            < 0.0
        ):
            # Keep the commanded quaternion in the start pose's hemisphere so
            # the final hold command cannot introduce a sign discontinuity.
            normalized_target_orientation = tuple(
                -value for value in normalized_target_orientation
            )
        numeric_parameters = (
            max_speed_mps,
            max_angular_speed_radps,
            publish_hz,
            settle_tolerance_m,
            settle_angular_tolerance_rad,
            settle_angular_speed_radps,
            timeout_sec,
        )
        if not all(math.isfinite(float(value)) for value in numeric_parameters):
            return MotionOutcome(False, "motion parameters must be finite")
        if max_speed_mps <= 0.0 or max_angular_speed_radps <= 0.0:
            return MotionOutcome(False, "motion speed limits must be positive")
        if publish_hz <= 0.0 or timeout_sec <= 0.0:
            return MotionOutcome(False, "publish rate and timeout must be positive")
        if (
            settle_tolerance_m < 0.0
            or settle_angular_tolerance_rad < 0.0
            or settle_angular_speed_radps < 0.0
        ):
            return MotionOutcome(False, "settling tolerances must be non-negative")
        distance = float(np.linalg.norm(target - start_position))
        angular_distance = (
            quaternion_angular_distance(
                start.orientation, normalized_target_orientation
            )
            if orientation_requested
            else 0.0
        )
        if (
            distance <= settle_tolerance_m
            and angular_distance <= settle_angular_tolerance_rad
        ):
            return MotionOutcome(
                True,
                "target already within settle tolerance",
                angular_distance_rad=angular_distance,
            )
        duration = max(
            0.35,
            distance / max_speed_mps,
            angular_distance / max_angular_speed_radps,
        )
        if duration >= timeout_sec:
            return MotionOutcome(
                False,
                "motion profile exceeds per-move timeout",
                angular_distance_rad=angular_distance,
            )

        force_abort = False
        was_cancelled = False
        force_feedback_lost = False

        def stop_requested() -> bool:
            nonlocal force_abort, was_cancelled, force_feedback_lost
            if cancelled():
                was_cancelled = True
                return True
            force_xyz = self._camera_rig.latest_force_xyz(max_age_sec=0.25)
            if force_xyz is None:
                force_feedback_lost = True
                return True
            force_abort = self._force_exceeded(
                force_xyz,
                baseline_force_xyz,
                max_force_n,
                force_delta_n,
            )
            return force_abort

        completed = self._publish_profile(
            start.position,
            tuple(float(value) for value in target),
            start.orientation,
            normalized_target_orientation,
            duration,
            publish_hz,
            stop_requested,
        )
        if not completed:
            self._retreat_to_step_start(start, publish_hz)
            if was_cancelled:
                return MotionOutcome(
                    False,
                    "motion cancelled and reversed",
                    angular_distance_rad=angular_distance,
                    cancelled=True,
                )
            if force_feedback_lost:
                return MotionOutcome(
                    False,
                    "wrist force feedback became stale; move reversed",
                    angular_distance_rad=angular_distance,
                )
            return MotionOutcome(
                False,
                "wrist force guard triggered; move reversed",
                angular_distance_rad=angular_distance,
                force_abort=True,
            )

        deadline = time.monotonic() + max(0.2, timeout_sec - duration)
        stable_samples = 0
        target_tuple = tuple(float(value) for value in target)
        while time.monotonic() < deadline:
            if stop_requested():
                self._retreat_to_step_start(start, publish_hz)
                if was_cancelled:
                    return MotionOutcome(
                        False,
                        "motion cancelled and reversed",
                        angular_distance_rad=angular_distance,
                        cancelled=True,
                    )
                if force_feedback_lost:
                    return MotionOutcome(
                        False,
                        "wrist force feedback became stale; move reversed",
                        angular_distance_rad=angular_distance,
                    )
                return MotionOutcome(
                    False,
                    "wrist force guard triggered while settling; move reversed",
                    angular_distance_rad=angular_distance,
                    force_abort=True,
                )
            self._publisher.publish(
                self._command(target_tuple, normalized_target_orientation)
            )
            state = self._current_state(0.1)
            if state is not None:
                position_error = float(
                    np.linalg.norm(
                        target - np.asarray(state.position, dtype=float)
                    )
                )
                orientation_error = (
                    quaternion_angular_distance(
                        state.orientation, normalized_target_orientation
                    )
                    if orientation_requested
                    else 0.0
                )
                angular_velocity_settled = (
                    not orientation_requested
                    or state.angular_speed_radps is None
                    or state.angular_speed_radps <= settle_angular_speed_radps
                )
                if (
                    position_error <= settle_tolerance_m
                    and state.speed_mps <= 0.01
                    and orientation_error <= settle_angular_tolerance_rad
                    and angular_velocity_settled
                ):
                    stable_samples += 1
                    if stable_samples >= 3:
                        return MotionOutcome(
                            True,
                            "measured TCP pose settled at target",
                            distance_m=distance,
                            angular_distance_rad=angular_distance,
                        )
                else:
                    stable_samples = 0
            time.sleep(max(0.02, 1.0 / publish_hz))

        self._retreat_to_step_start(start, publish_hz)
        return MotionOutcome(
            False,
            "measured TCP pose did not settle before timeout; move reversed",
            angular_distance_rad=angular_distance,
        )
