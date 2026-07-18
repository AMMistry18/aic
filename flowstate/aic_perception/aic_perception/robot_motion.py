"""Guarded Cartesian and joint motion through documented AIC interfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
import time
from typing import Any, Callable

import numpy as np

from .config import PerceptionConfig


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class ControllerPose:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    speed_mps: float
    received_at: float
    angular_speed_radps: float | None = None


@dataclass(frozen=True)
class JointReference:
    """Fresh measured arm joints paired with TCP speed and controller mode."""

    positions: tuple[float, ...]
    tcp_speed_mps: float
    received_at: float
    # ``aic_control_interfaces/msg/TargetMode.mode`` is uint8.  Store it as an
    # ordinary Python int so mode comparisons remain explicit and testable.
    target_mode: int | None = None


class MotionFailure(str, Enum):
    """Machine-readable reason why a motion transaction did not complete."""

    INVALID_REQUEST = "invalid_request"
    CONTROLLER_UNAVAILABLE = "controller_unavailable"
    MODE_UNAVAILABLE = "mode_unavailable"
    MODE_CHANGED = "mode_changed"
    JOINT_FEEDBACK_STALE = "joint_feedback_stale"
    FORCE_FEEDBACK_STALE = "force_feedback_stale"
    FORCE_LIMIT = "force_limit"
    CANCELLED = "cancelled"
    TARGET_TIMEOUT = "target_timeout"


@dataclass(frozen=True)
class MotionOutcome:
    success: bool
    message: str
    distance_m: float = 0.0
    angular_distance_rad: float = 0.0
    joint_distance_rad: float = 0.0
    force_abort: bool = False
    cancelled: bool = False
    failure: MotionFailure | None = None
    # ``success`` means the motion primitive completed safely.  A Cartesian
    # controller may stop at a safe, stationary pose before the requested
    # target and return success so the viewpoint planner can replan.  Keep
    # that case distinguishable from a target that was actually reached.
    target_reached: bool = False


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


def base_yaw_target_pose(
    position: tuple[float, float, float] | np.ndarray,
    orientation: tuple[float, float, float, float] | np.ndarray,
    angle_rad: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    """Return the TCP pose induced by shoulder pan about base-frame +Z.

    Rotating position and orientation together is the rigid forward-kinematics
    effect of joint 1.  It lets the accepted Cartesian controller execute the
    horizontal search without switching into a rejected joint target mode.
    """

    xyz = np.asarray(position, dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("position must contain three finite values")
    if not math.isfinite(float(angle_rad)):
        raise ValueError("yaw angle must be finite")
    current = normalize_quaternion(orientation)
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    target_position = (
        float(cosine * xyz[0] - sine * xyz[1]),
        float(sine * xyz[0] + cosine * xyz[1]),
        float(xyz[2]),
    )
    half = 0.5 * float(angle_rad)
    x2, y2, z2, w2 = current
    z1 = math.sin(half)
    w1 = math.cos(half)
    target_orientation = normalize_quaternion(
        (
            w1 * x2 - z1 * y2,
            w1 * y2 + z1 * x2,
            w1 * z2 + z1 * w2,
            w1 * w2 - z1 * z2,
        )
    )
    return target_position, target_orientation


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


def interpolated_joint_positions(
    start: tuple[float, ...],
    target: tuple[float, ...],
    samples: int,
) -> list[tuple[float, ...]]:
    """Minimum-jerk samples for an arbitrary fixed-size joint vector."""

    if samples < 2:
        raise ValueError("samples must be at least two")
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if (
        start_array.ndim != 1
        or target_array.shape != start_array.shape
        or start_array.size == 0
    ):
        raise ValueError("joint vectors must be non-empty and equal-sized")
    if not np.all(np.isfinite(np.concatenate((start_array, target_array)))):
        raise ValueError("joint positions must be finite")
    return [
        tuple(
            float(value)
            for value in start_array
            + minimum_jerk(index / (samples - 1))
            * (target_array - start_array)
        )
        for index in range(samples)
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
    """Own bounded, force-guarded Cartesian and joint controller sessions."""

    def __init__(self, node: Any, camera_rig: Any, config: PerceptionConfig):
        from aic_control_interfaces.msg import (
            ControllerState,
            JointMotionUpdate,
            MotionUpdate,
            TargetMode,
        )
        from aic_control_interfaces.srv import ChangeTargetMode
        from rclpy.qos import (
            DurabilityPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from sensor_msgs.msg import JointState

        self._node = node
        self._camera_rig = camera_rig
        self._config = config
        self._MotionUpdate = MotionUpdate
        self._JointMotionUpdate = JointMotionUpdate
        self._mode_joint = int(TargetMode.MODE_JOINT)
        self._ChangeTargetMode = ChangeTargetMode
        self._condition = threading.Condition()
        self._state: ControllerPose | None = None
        self._joint_reference: JointReference | None = None
        self._target_mode: int | None = None
        self._target_mode_received_at = -math.inf
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = node.create_publisher(
            MotionUpdate, config.pose_command_topic, qos
        )
        self._joint_publisher = node.create_publisher(
            JointMotionUpdate, config.joint_command_topic, qos
        )
        self._state_subscription = node.create_subscription(
            ControllerState,
            config.controller_state_topic,
            self._on_controller_state,
            qos,
        )
        self._joint_state_subscription = node.create_subscription(
            JointState,
            config.joint_state_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self._mode_client = node.create_client(
            ChangeTargetMode, config.change_target_mode_service
        )

    def _on_controller_state(self, message: Any) -> None:
        received_at = time.monotonic()
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
        target_mode = None
        try:
            target_mode = int(message.target_mode.mode)
        except (AttributeError, TypeError, ValueError):
            # TCP feedback remains useful if an older bridge omits mode.
            pass
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
                received_at=received_at,
                angular_speed_radps=angular_speed_radps,
            )
            if target_mode is not None:
                self._target_mode = target_mode
                self._target_mode_received_at = received_at
            self._condition.notify_all()

    def _on_joint_state(self, message: Any) -> None:
        """Cache the six measured arm joints in controller command order.

        ``ControllerState.reference_joint_state`` is a commanded reference and
        has no positions in the workcell's impedance mode.  ``/joint_states``
        is the documented measured source; names are used so gripper joints or
        publisher ordering can never move the wrong arm axis.
        """

        received_at = time.monotonic()
        try:
            names = tuple(str(name) for name in message.name)
            positions = tuple(float(value) for value in message.position)
        except (AttributeError, TypeError, ValueError):
            return
        if len(names) != len(positions) or len(set(names)) != len(names):
            return
        by_name = dict(zip(names, positions))
        if not all(name in by_name for name in ARM_JOINT_NAMES):
            return
        ordered = tuple(by_name[name] for name in ARM_JOINT_NAMES)
        if not np.all(np.isfinite(ordered)):
            return

        with self._condition:
            state = self._state
            tcp_speed = (
                state.speed_mps
                if state is not None
                and received_at - state.received_at <= 0.5
                else math.inf
            )
            target_mode = (
                self._target_mode
                if received_at - self._target_mode_received_at <= 0.5
                else None
            )
            self._joint_reference = JointReference(
                positions=ordered,
                tcp_speed_mps=float(tcp_speed),
                received_at=received_at,
                target_mode=target_mode,
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

    def _current_joint_reference(
        self,
        timeout_sec: float,
        target_mode: int | None = None,
    ) -> JointReference | None:
        """Return a fresh reference for every joint, or ``None`` when stale."""

        deadline = time.monotonic() + max(0.0, timeout_sec)
        with self._condition:
            while True:
                if (
                    self._joint_reference is not None
                    and time.monotonic() - self._joint_reference.received_at <= 0.5
                    and (
                        target_mode is None
                        or self._joint_reference.target_mode == target_mode
                    )
                ):
                    return self._joint_reference
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def _next_joint_reference(
        self,
        after_received_at: float,
        timeout_sec: float,
        target_mode: int | None = None,
    ) -> JointReference | None:
        """Wait for a new, fresh controller sample after ``after_received_at``."""

        deadline = time.monotonic() + max(0.0, timeout_sec)
        with self._condition:
            while True:
                reference = self._joint_reference
                if (
                    reference is not None
                    and reference.received_at > after_received_at
                    and time.monotonic() - reference.received_at <= 0.5
                    and (
                        target_mode is None
                        or reference.target_mode == target_mode
                    )
                ):
                    return reference
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def current_joint(
        self, joint_index: int, timeout_sec: float = 0.25
    ) -> float | None:
        """Expose one measured arm joint after validating its vector index."""

        if joint_index < 0:
            raise ValueError("joint_index must be non-negative")
        reference = self._current_joint_reference(timeout_sec)
        if reference is None or joint_index >= len(reference.positions):
            return None
        return float(reference.positions[joint_index])

    def current_joint1(self, timeout_sec: float = 0.25) -> float | None:
        """Expose measured shoulder-pan position as a readiness signal."""

        return self.current_joint(0, timeout_sec)

    def _reported_target_mode(self, max_age_sec: float = 0.5) -> int | None:
        """Latest controller-reported target mode, or None when stale."""

        now = time.monotonic()
        with self._condition:
            if now - self._target_mode_received_at <= max_age_sec:
                return self._target_mode
        return None

    def _await_reported_target_mode(
        self, mode: int, timeout_sec: float
    ) -> bool:
        """Wait until /aic_controller/controller_state reports ``mode``."""

        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while True:
                if (
                    self._target_mode == mode
                    and time.monotonic() - self._target_mode_received_at <= 0.5
                ):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)

    def _publish_mode_hold_command(self, reported_mode: int | None) -> bool:
        """Re-target the controller at its own measured state, once.

        The deployed controller build can reject target-mode changes while it
        considers an execution in flight (the repository source carries that
        exact todo).  Commanding "stay where you are" in the mode the
        controller reports completes any leftover execution immediately and
        is the documented no-risk command in that mode.
        """

        if reported_mode == self._mode_joint:
            reference = self._current_joint_reference(0.3)
            if reference is None or not reference.positions:
                return False
            self._joint_publisher.publish(
                self._joint_command(reference.positions)
            )
            return True
        state = self._current_state(0.3)
        if state is None:
            return False
        self._publisher.publish(
            self._command(state.position, state.orientation)
        )
        return True

    def prepare_controller_handoff(self) -> bool:
        """End this skill's command stream at the measured arm state.

        This does not own or release the Flowstate ``arm`` lease; that lease is
        held by the AIC controller bridge and is released by the immediately
        following ``Switch To Default Controller`` process node.  Publishing a
        final measured-state target ensures there is no trajectory in flight
        when that controller switch begins.
        """

        return self._publish_mode_hold_command(self._reported_target_mode())

    def _request_target_mode(
        self,
        mode: int,
        label: str,
        timeout_sec: float,
        *,
        min_attempts: int = 4,
    ) -> tuple[bool, str]:
        """Drive the controller's *reported* mode to ``mode``.

        The service response alone is not trustworthy in either direction:
        the repository controller source never rejects a valid mode, yet the
        live build returned rejections around in-flight executions, and a
        first-of-execution request has been observed to time out while still
        taking effect.  The documented controller_state topic reports the
        active mode, so that report is the acceptance criterion.  Requests
        already satisfied by the report are skipped entirely; after two
        failed attempts a measured-state hold command is published in the
        reported mode to complete any leftover execution that may be
        blocking the switch.
        """

        if self._reported_target_mode() == mode:
            return True, ""
        overall_deadline = time.monotonic() + max(float(timeout_sec), 8.0)
        logger = self._node.get_logger()
        attempt = 0
        last_error = f"{label} target-mode request was never attempted"
        while attempt < min_attempts or time.monotonic() < overall_deadline:
            attempt += 1
            remaining = max(0.5, overall_deadline - time.monotonic())
            response_accepted = False
            if not self._mode_client.wait_for_service(
                timeout_sec=min(1.5, remaining)
            ):
                last_error = f"{label} target-mode service is unavailable"
            else:
                request = self._ChangeTargetMode.Request()
                request.target_mode.mode = mode
                future = self._mode_client.call_async(request)
                call_deadline = time.monotonic() + min(1.5, remaining)
                while not future.done() and time.monotonic() < call_deadline:
                    time.sleep(0.01)
                if not future.done():
                    future.cancel()
                    last_error = f"timed out switching to {label} target mode"
                else:
                    try:
                        response = future.result()
                    except Exception as error:
                        last_error = (
                            f"{label} target-mode request failed: {error}"
                        )
                    else:
                        if response is not None and response.success:
                            response_accepted = True
                        else:
                            last_error = (
                                f"controller rejected {label} target mode"
                            )
            # The reported mode is the acceptance criterion regardless of
            # what the service said.
            if self._await_reported_target_mode(
                mode, 1.5 if response_accepted else 0.8
            ):
                return True, ""
            reported = self._reported_target_mode()
            if response_accepted and reported is None:
                # No fresh state report exists to contradict an accepted
                # request; trust the response rather than stalling.
                return True, ""
            if response_accepted:
                last_error = (
                    f"state report stayed in mode {reported} while "
                    f"confirming {label} target mode"
                )
            logger.warning(
                f"{label} target-mode attempt {attempt} unresolved: "
                f"{last_error} (controller reports mode {reported})"
            )
            if attempt >= 2:
                self._publish_mode_hold_command(reported)
            if attempt >= min_attempts and time.monotonic() >= overall_deadline:
                break
            time.sleep(0.25)
        reported = self._reported_target_mode()
        return False, f"{last_error} (controller reports mode {reported})"

    def _ensure_cartesian_mode(self, timeout_sec: float) -> tuple[bool, str]:
        from aic_control_interfaces.msg import TargetMode

        return self._request_target_mode(
            TargetMode.MODE_CARTESIAN, "Cartesian", timeout_sec
        )

    def _ensure_joint_mode(self, timeout_sec: float) -> tuple[bool, str]:
        """Switch the documented AIC controller to joint target mode."""

        from aic_control_interfaces.msg import TargetMode

        return self._request_target_mode(
            TargetMode.MODE_JOINT, "joint", timeout_sec
        )

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

    def _joint_command(self, positions: tuple[float, ...]) -> Any:
        """Build one documented position-mode JointMotionUpdate command."""

        from aic_control_interfaces.msg import TrajectoryGenerationMode

        values = tuple(float(value) for value in positions)
        if not values or not np.all(np.isfinite(values)):
            raise ValueError("joint command positions must be finite and non-empty")
        message = self._JointMotionUpdate()
        message.target_state.positions = list(values)
        if len(values) == 6:
            # Match the controller's own configured joint_impedance defaults
            # (aic_ros2_controllers.yaml): stiff arm joints, light wrists.
            # The earlier uniform 85/75 gains over-damped wrist_3 five-fold
            # against its configured 15 and produced a visible jerk on the
            # first joint-mode command of every J6 attempt.
            message.target_stiffness = [
                100.0, 100.0, 100.0, 50.0, 50.0, 50.0
            ]
            message.target_damping = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        else:
            message.target_stiffness = [85.0] * len(values)
            message.target_damping = [75.0] * len(values)
        message.target_feedforward_torque = [0.0] * len(values)
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

    def _publish_joint_profile(
        self,
        start: tuple[float, ...],
        target: tuple[float, ...],
        duration_sec: float,
        publish_hz: float,
        stop: Callable[[], bool] | None = None,
    ) -> tuple[bool, tuple[float, ...]]:
        """Publish a minimum-jerk full-vector joint profile."""

        samples = max(2, int(math.ceil(duration_sec * publish_hz)) + 1)
        period = 1.0 / publish_hz
        next_tick = time.monotonic()
        last = tuple(float(value) for value in start)
        for positions in interpolated_joint_positions(start, target, samples):
            if stop is not None and stop():
                return False, last
            self._joint_publisher.publish(self._joint_command(positions))
            last = positions
            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
        return True, last

    def _retreat_joint_target(
        self,
        start: JointReference,
        last_commanded: tuple[float, ...],
        max_speed_radps: float,
        publish_hz: float,
    ) -> None:
        """Best-effort return to the full measured arm pose saved before yaw."""

        # Joint-state freshness and controller-mode freshness are independent
        # streams.  A valid measured arm vector remains the best retreat start
        # even when the latest controller-state mode report has aged out.
        current = self._current_joint_reference(0.15)
        current_positions = (
            current.positions
            if current is not None
            and len(current.positions) == len(start.positions)
            else last_commanded
        )
        distance = float(
            np.max(
                np.abs(
                    np.asarray(start.positions, dtype=float)
                    - np.asarray(current_positions, dtype=float)
                )
            )
        )
        # A quintic minimum-jerk profile has a peak normalized derivative of
        # 1.875, so scale duration by that factor to honor the requested limit.
        duration = max(0.2, 1.875 * distance / max_speed_radps)
        self._publish_joint_profile(
            current_positions,
            start.positions,
            duration,
            publish_hz,
        )

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
        # The wrist sensor reports its static ~14 N gravity/bias load in the
        # sensor frame, so any tool reorientation rotates that vector even in
        # free space (a 0.30 rad wrist yaw moves it by ~4 N against a 5 N
        # threshold).  Compare magnitudes instead: the norm of a constant
        # load is rotation-invariant, while genuine contact changes it.
        baseline = np.asarray(baseline_xyz, dtype=float)
        return (
            abs(
                float(np.linalg.norm(force))
                - float(np.linalg.norm(baseline))
            )
            >= force_delta_n
        )

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

    def move_joint1_yaw(
        self,
        delta_rad: float,
        *,
        max_speed_radps: float = 0.20,
        publish_hz: float,
        settle_tolerance_rad: float = 0.01,
        settle_tcp_speed_mps: float = 0.02,
        timeout_sec: float,
        baseline_force_xyz: tuple[float, float, float] | None,
        max_force_n: float,
        force_delta_n: float,
        cancelled: Callable[[], bool],
        _joint_index: int = 0,
        _joint_label: str = "joint 1",
    ) -> MotionOutcome:
        """Move one yaw joint while preserving every other measured reference.

        The public J1 entry point uses index 0. ``move_joint6_yaw`` below uses
        index 5 for UR5e ``wrist_3_joint``. The primitive reads the complete
        measured vector from documented ``/joint_states`` and changes only the
        requested element. It never invents targets for the remaining joints.
        A force, cancel, stale-force, or stale-controller guard returns the full
        vector to the saved measured start before reporting failure.
        """

        numeric_parameters = (
            delta_rad,
            max_speed_radps,
            publish_hz,
            settle_tolerance_rad,
            settle_tcp_speed_mps,
            timeout_sec,
            max_force_n,
            force_delta_n,
        )
        if not all(math.isfinite(float(value)) for value in numeric_parameters):
            return MotionOutcome(False, "joint-yaw parameters must be finite")
        if _joint_index < 0:
            return MotionOutcome(False, "joint-yaw index must be non-negative")
        if not 0.0 < max_speed_radps <= 0.20:
            return MotionOutcome(
                False,
                f"{_joint_label} yaw speed must be in (0, 0.20] rad/s",
                failure=MotionFailure.INVALID_REQUEST,
            )
        if publish_hz <= 0.0 or timeout_sec <= 0.0:
            return MotionOutcome(False, "publish rate and timeout must be positive")
        if settle_tolerance_rad < 0.0 or settle_tcp_speed_mps < 0.0:
            return MotionOutcome(False, "joint settling tolerances must be non-negative")
        if self._joint_publisher.get_subscription_count() < 1:
            deadline = time.monotonic() + min(timeout_sec, 2.0)
            while (
                self._joint_publisher.get_subscription_count() < 1
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
        if self._joint_publisher.get_subscription_count() < 1:
            return MotionOutcome(
                False,
                "joint pose command has no subscriber",
                failure=MotionFailure.CONTROLLER_UNAVAILABLE,
            )

        # Record fresh measured joints before changing modes.  Commanding from
        # an older Cartesian-mode sample could otherwise pull unrelated joints
        # toward a stale posture.
        pre_switch = self._current_joint_reference(min(timeout_sec, 2.0))
        if pre_switch is None:
            return MotionOutcome(
                False,
                "no fresh measured arm joint state",
                failure=MotionFailure.JOINT_FEEDBACK_STALE,
            )
        mode_ok, mode_error = self._ensure_joint_mode(min(timeout_sec, 2.0))
        if not mode_ok:
            return MotionOutcome(
                False, mode_error, failure=MotionFailure.MODE_UNAVAILABLE
            )

        def finish(outcome: MotionOutcome) -> MotionOutcome:
            """Never leave the shared controller in joint target mode."""

            restored, restore_error = self._ensure_cartesian_mode(
                min(timeout_sec, 2.0)
            )
            if restored:
                return outcome
            return replace(
                outcome,
                success=False,
                message=f"{outcome.message}; {restore_error}",
                failure=MotionFailure.MODE_UNAVAILABLE,
            )

        start = self._next_joint_reference(
            pre_switch.received_at,
            min(timeout_sec, 2.0),
            target_mode=self._mode_joint,
        )
        if start is None:
            return finish(
                MotionOutcome(
                    False,
                    "no newer measured arm state confirming joint target mode",
                    failure=MotionFailure.MODE_UNAVAILABLE,
                )
            )
        if not start.positions:
            return finish(
                MotionOutcome(
                    False,
                    "measured arm joint state is empty",
                    failure=MotionFailure.JOINT_FEEDBACK_STALE,
                )
            )
        if _joint_index >= len(start.positions):
            return finish(
                MotionOutcome(
                    False,
                    f"measured arm vector has no {_joint_label} index "
                    f"{_joint_index}",
                    failure=MotionFailure.INVALID_REQUEST,
                )
            )
        if abs(delta_rad) <= settle_tolerance_rad:
            return finish(
                MotionOutcome(
                    True,
                    f"{_joint_label} target already within tolerance",
                    target_reached=True,
                )
            )

        target_values = list(start.positions)
        target_values[_joint_index] += float(delta_rad)
        target = tuple(target_values)
        # Peak velocity for minimum-jerk interpolation is 1.875 * distance / T.
        duration = max(0.35, 1.875 * abs(delta_rad) / max_speed_radps)
        if duration >= timeout_sec:
            return finish(
                MotionOutcome(
                    False,
                    f"{_joint_label} motion profile exceeds per-move timeout",
                    joint_distance_rad=abs(delta_rad),
                    failure=MotionFailure.TARGET_TIMEOUT,
                )
            )

        if self._camera_rig.wait_for_force_xyz(
            timeout_sec=min(1.0, timeout_sec), max_age_sec=0.5
        ) is None:
            return finish(
                MotionOutcome(
                    False,
                    f"no fresh wrist-force sample before {_joint_label} yaw; "
                    "motion refused",
                    joint_distance_rad=abs(delta_rad),
                    failure=MotionFailure.FORCE_FEEDBACK_STALE,
                )
            )

        guard_reason: str | None = None

        def stop_requested() -> bool:
            nonlocal guard_reason
            if cancelled():
                guard_reason = "cancelled"
                return True
            force_xyz = self._camera_rig.latest_force_xyz(max_age_sec=0.5)
            if force_xyz is None:
                guard_reason = "stale_force"
                return True
            if self._force_exceeded(
                force_xyz,
                baseline_force_xyz,
                max_force_n,
                force_delta_n,
            ):
                guard_reason = "force"
                return True
            # Do not couple fresh measured /joint_states to the independent
            # controller-state timestamp.  The transaction already confirmed
            # joint mode at its start.  During the profile, abort only for a
            # genuinely stale measured vector or an explicit fresh report that
            # the controller changed away from joint mode.  A temporarily
            # unknown/stale mode report is not stale joint feedback.
            if self._current_joint_reference(0.0) is None:
                guard_reason = "stale_joints"
                return True
            reported_mode = self._reported_target_mode()
            if reported_mode is not None and reported_mode != self._mode_joint:
                guard_reason = "mode_changed"
                return True
            return False

        completed, last_commanded = self._publish_joint_profile(
            start.positions,
            target,
            duration,
            publish_hz,
            stop_requested,
        )
        if not completed:
            self._retreat_joint_target(
                start, last_commanded, max_speed_radps, publish_hz
            )
            messages = {
                "cancelled": f"{_joint_label} yaw cancelled and reversed",
                "stale_force": (
                    f"wrist force feedback became stale; {_joint_label} yaw reversed"
                ),
                "stale_joints": (
                    f"measured joint feedback became stale; {_joint_label} yaw reversed"
                ),
                "mode_changed": (
                    f"controller left joint target mode; {_joint_label} yaw reversed"
                ),
                "force": f"wrist force guard triggered; {_joint_label} yaw reversed",
            }
            failures = {
                "cancelled": MotionFailure.CANCELLED,
                "stale_force": MotionFailure.FORCE_FEEDBACK_STALE,
                "stale_joints": MotionFailure.JOINT_FEEDBACK_STALE,
                "mode_changed": MotionFailure.MODE_CHANGED,
                "force": MotionFailure.FORCE_LIMIT,
            }
            return finish(
                MotionOutcome(
                    False,
                    messages.get(
                        guard_reason or "", f"{_joint_label} yaw stopped and reversed"
                    ),
                    joint_distance_rad=abs(delta_rad),
                    force_abort=guard_reason == "force",
                    cancelled=guard_reason == "cancelled",
                    failure=failures.get(guard_reason or ""),
                )
            )

        deadline = time.monotonic() + max(0.2, timeout_sec - duration)
        stable_samples = 0
        last_received_at = start.received_at
        while time.monotonic() < deadline:
            if stop_requested():
                self._retreat_joint_target(
                    start, target, max_speed_radps, publish_hz
                )
                messages = {
                    "cancelled": (
                        f"{_joint_label} yaw cancelled while settling and reversed"
                    ),
                    "stale_force": (
                        f"wrist force feedback became stale; {_joint_label} yaw reversed"
                    ),
                    "stale_joints": (
                        f"measured joint feedback became stale; {_joint_label} yaw reversed"
                    ),
                    "mode_changed": (
                        f"controller left joint target mode; {_joint_label} yaw reversed"
                    ),
                    "force": (
                        "wrist force guard triggered while settling; "
                        f"{_joint_label} yaw reversed"
                    ),
                }
                failures = {
                    "cancelled": MotionFailure.CANCELLED,
                    "stale_force": MotionFailure.FORCE_FEEDBACK_STALE,
                    "stale_joints": MotionFailure.JOINT_FEEDBACK_STALE,
                    "mode_changed": MotionFailure.MODE_CHANGED,
                    "force": MotionFailure.FORCE_LIMIT,
                }
                return finish(
                    MotionOutcome(
                        False,
                        messages.get(
                            guard_reason or "",
                            f"{_joint_label} yaw stopped and reversed",
                        ),
                        joint_distance_rad=abs(delta_rad),
                        force_abort=guard_reason == "force",
                        cancelled=guard_reason == "cancelled",
                        failure=failures.get(guard_reason or ""),
                    )
                )

            self._joint_publisher.publish(self._joint_command(target))
            remaining = max(0.0, deadline - time.monotonic())
            reference = self._next_joint_reference(
                last_received_at,
                min(0.15, remaining),
            )
            if reference is not None:
                last_received_at = reference.received_at
                if len(reference.positions) != len(target):
                    self._retreat_joint_target(
                        start, target, max_speed_radps, publish_hz
                    )
                    return finish(
                        MotionOutcome(
                            False,
                            "measured joint vector size changed; "
                            f"{_joint_label} yaw reversed",
                            joint_distance_rad=abs(delta_rad),
                            failure=MotionFailure.JOINT_FEEDBACK_STALE,
                        )
                    )
                if (
                    abs(
                        reference.positions[_joint_index]
                        - target[_joint_index]
                    )
                    <= settle_tolerance_rad
                    and reference.tcp_speed_mps <= settle_tcp_speed_mps
                ):
                    stable_samples += 1
                    if stable_samples >= 3:
                        return finish(
                            MotionOutcome(
                                True,
                                f"measured {_joint_label} settled at yaw target",
                                joint_distance_rad=abs(delta_rad),
                                target_reached=True,
                            )
                        )
                else:
                    stable_samples = 0
            time.sleep(max(0.02, 1.0 / publish_hz))

        self._retreat_joint_target(start, target, max_speed_radps, publish_hz)
        return finish(
            MotionOutcome(
                False,
                f"measured {_joint_label} did not settle before timeout; "
                "yaw reversed",
                joint_distance_rad=abs(delta_rad),
                failure=MotionFailure.TARGET_TIMEOUT,
            )
        )

    def move_joint6_yaw(
        self,
        delta_rad: float,
        *,
        max_speed_radps: float = 0.20,
        publish_hz: float,
        settle_tolerance_rad: float = 0.01,
        settle_tcp_speed_mps: float = 0.02,
        timeout_sec: float,
        baseline_force_xyz: tuple[float, float, float] | None,
        max_force_n: float,
        force_delta_n: float,
        cancelled: Callable[[], bool],
    ) -> MotionOutcome:
        """Yaw UR5e wrist joint 6 while holding joints 1-5 measured targets."""

        return self.move_joint1_yaw(
            delta_rad,
            max_speed_radps=max_speed_radps,
            publish_hz=publish_hz,
            settle_tolerance_rad=settle_tolerance_rad,
            settle_tcp_speed_mps=settle_tcp_speed_mps,
            timeout_sec=timeout_sec,
            baseline_force_xyz=baseline_force_xyz,
            max_force_n=max_force_n,
            force_delta_n=force_delta_n,
            cancelled=cancelled,
            _joint_index=5,
            _joint_label="joint 6 (wrist_3_joint)",
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
                target_reached=True,
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


        if self._camera_rig.wait_for_force_xyz(
            timeout_sec=min(1.0, timeout_sec), max_age_sec=0.5
        ) is None:
            return MotionOutcome(
                False,
                "no fresh wrist-force sample before Cartesian motion; "
                "motion refused",
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
            # Match CameraRig.grab's freshness contract.  The F/T broadcaster
            # runs at 50 Hz, but brief best-effort ROS/Zenoh jitter exceeded
            # the old motion-only 0.25 s cutoff and caused a false rollback.
            # Force magnitude and baseline-delta checks still run on every
            # accepted sample and a 0.5 s loss still aborts the move.
            force_xyz = self._camera_rig.latest_force_xyz(max_age_sec=0.5)
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
                    and state.speed_mps <= 0.02
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
                            target_reached=True,
                        )
                else:
                    stable_samples = 0
            time.sleep(max(0.02, 1.0 / publish_hz))

        # AIC can keep reporting a small nonzero TCP velocity while holding a
        # valid target.  Preserve the safety margin, but do not reverse a move
        # that demonstrably arrived at the viewpoint merely because the last
        # few velocity samples did not become exactly quiet.
        state = self._current_state(0.15)
        timeout_diagnostic = "no fresh controller TCP state at timeout"
        if state is not None:
            position_error = float(
                np.linalg.norm(target - np.asarray(state.position, dtype=float))
            )
            orientation_error = (
                quaternion_angular_distance(
                    state.orientation, normalized_target_orientation
                )
                if orientation_requested
                else 0.0
            )
            if (
                position_error <= settle_tolerance_m
                and orientation_error <= settle_angular_tolerance_rad
            ):
                return MotionOutcome(
                    True,
                    "measured TCP reached target with residual controller motion",
                    distance_m=distance,
                    angular_distance_rad=angular_distance,
                    target_reached=True,
                )

            angular_velocity_safe = (
                state.angular_speed_radps is None
                or state.angular_speed_radps <= max(0.08, settle_angular_speed_radps)
            )
            angular_speed_text = (
                "unknown"
                if state.angular_speed_radps is None
                else f"{state.angular_speed_radps:.4f}rad/s"
            )
            timeout_diagnostic = (
                f"position_error={position_error:.4f}m "
                f"orientation_error={orientation_error:.4f}rad "
                f"linear_speed={state.speed_mps:.4f}m/s "
                f"angular_speed={angular_speed_text}"
            )
            partial_orientation_safe = (
                not orientation_requested
                or orientation_error
                <= max(0.08, 2.0 * settle_angular_tolerance_rad)
            )
            if (
                state.speed_mps <= 0.02
                and angular_velocity_safe
                and partial_orientation_safe
            ):
                # A joint/workspace limit can make the controller stop short
                # of a Cartesian request even though the reached pose is safe.
                # Hold that measured pose and let camera feedback choose a new
                # direction instead of reversing and failing the whole search.
                self._publisher.publish(
                    self._command(state.position, state.orientation)
                )
                measured_distance = float(
                    np.linalg.norm(
                        np.asarray(state.position, dtype=float) - start_position
                    )
                )
                measured_angle = quaternion_angular_distance(
                    start.orientation, state.orientation
                )
                return MotionOutcome(
                    True,
                    "controller stopped short; replanning from measured TCP",
                    distance_m=measured_distance,
                    angular_distance_rad=measured_angle,
                    target_reached=False,
                )

        self._retreat_to_step_start(start, publish_hz)
        return MotionOutcome(
            False,
            "measured TCP pose did not settle before timeout "
            f"({timeout_diagnostic}); move reversed",
            angular_distance_rad=angular_distance,
        )
