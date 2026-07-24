"""Plug-relative deterministic SFP controller for the v50 Flowstate image.

This module is intentionally separate from the deployed v49 ``RLInsert.py``.
The v50 image installs it next to both runtime copies of ``RLInsert`` and applies
a small, hash-gated dispatch patch.  Keeping the state machine here makes the
new behavior testable without importing ROS.

The controller has one non-negotiable perception contract: a fresh, validated
plug pose is required before motion and after every lift recovery.  The old
fixed cable-angle bias is never used as a fallback.  Once a fresh plug pose has
been observed, the measured TCP-to-plug transform is used kinematically while
the cameras become occluded during seating.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import time
from typing import Optional

import numpy as np


INSERT_DEPTH_M = 0.0458
LOCAL_SFP_PORT_KPS = np.array(
    [
        [0.00685, 0.0043, 0.0],
        [-0.00685, 0.0043, 0.0],
        [-0.00685, -0.0043, 0.0],
        [0.00685, -0.0043, 0.0],
    ],
    dtype=np.float64,
)

SEATED = "seated"
STALLED = "stalled"
HARD_FAILURE = "hard_failure"
# To activate the correction, set FORCE_GAIN ~0.00015 (m/N) and MOMENT_GAIN
# ~0.02 (rad/N·m) after verifying sign/frame from SEAT_WRENCH logs; these are
# starting points to bench-tune.
# Activate P3 after reviewing SEAT_SLOPE/SEAT_WRENCH logs: set
# MOUTH_SPEED_SCALE~0.5 and STALL_GRACE_S~3.0 (starting points, bench-tune).
SEAT_ALIGN_OBSERVE_FORCE_GAIN = 0.00015
SEAT_ALIGN_OBSERVE_MOMENT_GAIN = 0.02
SEAT_WRENCH_LOG_PERIOD_S = 0.2
SEAT_SLOPE_LOG_PERIOD_S = 0.2
SEAT_SLOPE_WINDOW_S = 0.5


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class V50Config:
    """Safety and timing limits, expressed in wall time where applicable."""

    command_dt_sim_s: float = 0.05
    align_timeout_wall_s: float = 15.0
    align_lateral_tol_m: float = 0.0010
    align_rotation_tol_rad: float = np.deg2rad(1.5)
    align_max_lateral_step_m: float = 0.0015
    align_max_rotation_step_rad: float = np.deg2rad(1.5)
    stall_timeout_wall_s: float = 2.5
    stall_progress_m: float = 0.0008
    free_speed_m_s: float = 0.015
    contact_speed_m_s: float = 0.006
    contact_force_n: float = 3.0
    target_axial_force_n: float = 8.0
    seat_force_cap_n: float = 10.0
    force_abort_n: float = 18.0
    force_abort_wall_s: float = 0.25
    axial_stiffness_n_m: float = 500.0
    max_axial_lead_m: float = 0.020
    lateral_safety_m: float = 0.006
    rotation_safety_rad: float = np.deg2rad(15.0)
    seat_align_enable: bool = True
    seat_align_force_gain: float = 0.00003
    seat_align_moment_gain: float = 0.004
    seat_align_max_lat_m: float = 0.0004
    seat_align_max_tilt_rad: float = 0.0087
    seat_align_release_decay: float = 0.7
    seat_mouth_zone_m: float = 0.006
    seat_mouth_speed_scale: float = 0.25
    seat_stall_grace_s: float = 1.5
    seat_overtravel_m: float = 0.005
    seat_candidate_depth_m: float = 0.0445
    insertion_event_timeout_wall_s: float = 6.0
    plug_max_age_s: float = 0.35

    @classmethod
    def from_env(cls) -> "V50Config":
        return cls(
            command_dt_sim_s=_env_float("RL_INSERT_V50_COMMAND_DT_S", 0.05),
            align_timeout_wall_s=_env_float("RL_INSERT_V50_ALIGN_TIMEOUT_S", 15.0),
            stall_timeout_wall_s=_env_float("RL_INSERT_V50_STALL_TIMEOUT_S", 2.5),
            stall_progress_m=_env_float("RL_INSERT_V50_STALL_PROGRESS_M", 0.0008),
            free_speed_m_s=_env_float("RL_INSERT_V50_FREE_SPEED_M_S", 0.015),
            contact_speed_m_s=_env_float("RL_INSERT_V50_CONTACT_SPEED_M_S", 0.006),
            contact_force_n=_env_float("RL_INSERT_V50_CONTACT_FORCE_N", 3.0),
            target_axial_force_n=_env_float("RL_INSERT_V50_TARGET_FORCE_N", 8.0),
            seat_force_cap_n=_env_float("RL_INSERT_V50_SEAT_FORCE_CAP_N", 10.0),
            force_abort_n=_env_float("RL_INSERT_FORCE_ABORT_N", 18.0),
            force_abort_wall_s=_env_float("RL_INSERT_V50_FORCE_ABORT_DWELL_S", 0.25),
            axial_stiffness_n_m=_env_float("RL_INSERT_V50_AXIAL_STIFFNESS_N_M", 500.0),
            max_axial_lead_m=_env_float("RL_INSERT_V50_MAX_AXIAL_LEAD_M", 0.020),
            seat_align_enable=_env_bool("RL_INSERT_V50_SEAT_ALIGN_ENABLE", True),
            seat_align_force_gain=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_FORCE_GAIN", 0.00003
            ),
            seat_align_moment_gain=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_MOMENT_GAIN", 0.004
            ),
            seat_align_max_lat_m=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_MAX_LAT_M", 0.0004
            ),
            seat_align_max_tilt_rad=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_MAX_TILT_RAD", 0.0087
            ),
            seat_align_release_decay=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_RELEASE_DECAY", 0.7
            ),
            seat_mouth_zone_m=_env_float("RL_INSERT_V50_SEAT_MOUTH_ZONE_M", 0.006),
            seat_mouth_speed_scale=_env_float(
                "RL_INSERT_V50_SEAT_MOUTH_SPEED_SCALE", 0.25
            ),
            seat_stall_grace_s=_env_float("RL_INSERT_V50_SEAT_STALL_GRACE_S", 1.5),
            seat_overtravel_m=_env_float("RL_INSERT_V50_SEAT_OVERTRAVEL_M", 0.005),
            insertion_event_timeout_wall_s=_env_float(
                "RL_INSERT_V50_EVENT_TIMEOUT_S", 6.0
            ),
            plug_max_age_s=_env_float("RL_INSERT_V50_PLUG_MAX_AGE_S", 0.35),
        ).validated()

    def validated(self) -> "V50Config":
        if not 0.0 < self.target_axial_force_n < self.seat_force_cap_n:
            raise ValueError("v50 target force must be below the seat force cap")
        if not self.seat_force_cap_n < self.force_abort_n <= 18.0:
            raise ValueError("v50 force cap must be below the <=18 N hard abort")
        if self.axial_stiffness_n_m <= 0.0:
            raise ValueError("v50 axial stiffness must be positive")
        if not 0.0 <= self.seat_overtravel_m <= 0.008:
            raise ValueError("v50 seat overtravel must stay within 0-8 mm")
        if self.seat_align_max_lat_m < 0.0 or self.seat_align_max_tilt_rad < 0.0:
            raise ValueError("v50 seat alignment correction caps must be non-negative")
        if not 0.0 <= self.seat_align_release_decay <= 1.0:
            raise ValueError("v50 seat alignment release decay must be within 0-1")
        if (
            self.seat_mouth_zone_m < 0.0
            or self.seat_mouth_speed_scale < 0.0
            or self.seat_stall_grace_s < 0.0
        ):
            raise ValueError("v50 P3 observe-first parameters must be non-negative")
        return self

    @property
    def force_lead_m(self) -> float:
        return min(
            self.max_axial_lead_m,
            self.target_axial_force_n / self.axial_stiffness_n_m,
        )


def quaternion_to_matrix(quaternion_wxyz) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s,
                          (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                          (R[1, 2] + R[2, 1]) / s])
        else:
            s = np.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s,
                          (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0.0:
        q = -q
    return q


def axis_angle(rotation) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))
    if angle <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    denom = 2.0 * np.sin(angle)
    if abs(denom) <= 1e-9:
        values, vectors = np.linalg.eigh((R + np.eye(3)) * 0.5)
        axis = vectors[:, int(np.argmax(values))]
    else:
        axis /= denom
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return axis * angle


def rotation_from_axis_angle(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle <= 1e-12:
        return np.eye(3)
    axis = v / angle
    K = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def clamp_vector_norm(vector, max_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm):
        return np.zeros_like(value)
    if max_norm <= 0.0:
        return np.zeros_like(value)
    if norm > max_norm:
        return value * (max_norm / norm)
    return value


def solve_tip_in_tcp(tcp_pos, tcp_quat, tip_pos, tip_rotation):
    """Return the measured rigid plug-tip transform in the current TCP frame."""

    tcp_pos = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
    tip_pos = np.asarray(tip_pos, dtype=np.float64).reshape(3)
    R_tcp = quaternion_to_matrix(tcp_quat)
    R_tip = np.asarray(tip_rotation, dtype=np.float64).reshape(3, 3)
    return R_tcp.T @ (tip_pos - tcp_pos), R_tcp.T @ R_tip


def tip_from_tcp_transform(tcp_pos, tcp_quat, tip_in_tcp_pos, tip_in_tcp_rotation):
    R_tcp = quaternion_to_matrix(tcp_quat)
    tip_pos = (
        np.asarray(tcp_pos, dtype=np.float64).reshape(3)
        + R_tcp @ np.asarray(tip_in_tcp_pos, dtype=np.float64).reshape(3)
    )
    tip_rotation = R_tcp @ np.asarray(tip_in_tcp_rotation, dtype=np.float64).reshape(3, 3)
    return tip_pos, tip_rotation


def tcp_for_tip_transform(tip_pos, tip_rotation, tip_in_tcp_pos, tip_in_tcp_rotation):
    R_tip = np.asarray(tip_rotation, dtype=np.float64).reshape(3, 3)
    R_rel = np.asarray(tip_in_tcp_rotation, dtype=np.float64).reshape(3, 3)
    R_tcp = R_tip @ R_rel.T
    tcp_pos = (
        np.asarray(tip_pos, dtype=np.float64).reshape(3)
        - R_tcp @ np.asarray(tip_in_tcp_pos, dtype=np.float64).reshape(3)
    )
    return tcp_pos, matrix_to_quaternion(R_tcp)


def v50_tip_from_tcp(policy, tcp_pos, tcp_quat):
    """Use the per-run visual grasp transform once it has been established.

    The legacy transform remains available only before v50 control starts so the
    old port detector can rank multiple port candidates.  ``run_v50_script``
    refuses to move unless ``_v50_grasp_transform`` has been populated from a
    fresh visual plug observation.
    """

    transform = getattr(policy, "_v50_grasp_transform", None)
    if transform is None:
        from .rl_insert_contract import sfp_tip_pose_from_tcp

        return sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)
    return tip_from_tcp_transform(tcp_pos, tcp_quat, *transform)


def v50_tcp_pose_for_tip(policy, tip_pos, tip_rotation):
    transform = getattr(policy, "_v50_grasp_transform", None)
    if transform is None:
        from .rl_insert_contract import tcp_pose_for_sfp_tip

        return tcp_pose_for_sfp_tip(tip_pos, tip_rotation)
    return tcp_for_tip_transform(tip_pos, tip_rotation, *transform)


def next_persistent_depth(
    current_depth: float,
    commanded_depth: float,
    elapsed_wall_s: float,
    force_n: float,
    config: V50Config,
) -> float:
    """Advance an absolute seat setpoint while retaining bounded axial lead."""

    current_depth = float(current_depth)
    commanded_depth = max(float(commanded_depth), current_depth)
    if np.isfinite(force_n) and force_n >= config.seat_force_cap_n:
        candidate = commanded_depth
    else:
        speed = (
            config.contact_speed_m_s
            if np.isfinite(force_n) and force_n >= config.contact_force_n
            else config.free_speed_m_s
        )
        candidate = commanded_depth + speed * max(0.0, float(elapsed_wall_s))
    # This is a persistent absolute setpoint.  It grows while the plug is stuck,
    # unlike v49's ``current_tip + one_step`` command.  The lead cap bounds the
    # nominal impedance demand to approximately target_axial_force_n.
    return min(
        INSERT_DEPTH_M + config.seat_overtravel_m,
        candidate,
        current_depth + config.force_lead_m,
    )


@dataclass
class WallProgressWatch:
    progress_m: float
    timeout_s: float
    best_depth: float
    progress_time: float

    @classmethod
    def start(cls, depth: float, now: float, config: V50Config):
        return cls(config.stall_progress_m, config.stall_timeout_wall_s, depth, now)

    def stalled(self, depth: float, now: float) -> bool:
        if float(depth) >= self.best_depth + self.progress_m:
            self.best_depth = float(depth)
            self.progress_time = float(now)
        return float(now) - self.progress_time >= self.timeout_s


def configure_v50(policy) -> None:
    """Load the dedicated plug estimator and initialize per-action state."""

    if getattr(policy, "_v50_configured", False):
        return
    from .sfp_plug_pose import SfpPlugPoseEstimator

    weights = Path(
        os.environ.get("AIC_SFP_PLUG_POSE_WEIGHTS", "/models/sfp_plug_pose_best.pt")
    )
    if not weights.is_file():
        raise FileNotFoundError(
            f"v50 plug-pose weights missing at {weights}; no fixed-bias fallback is allowed"
        )
    policy._v50_config = V50Config.from_env()
    policy._v50_plug_estimator = SfpPlugPoseEstimator(
        str(weights),
        imgsz=_env_int("RL_INSERT_V50_PLUG_IMGSZ", 960),
        conf_threshold=_env_float("RL_INSERT_V50_PLUG_CONF", 0.25),
    )
    policy._v50_grasp_transform = None
    policy._v50_pending_world_plug = None
    _warm_v50_perception(policy)
    policy._v50_configured = True
    policy.get_logger().info(
        f"[v50] plug-relative controller ready; weights={weights} "
        f"force_target={policy._v50_config.target_axial_force_n:.1f}N "
        f"force_abort={policy._v50_config.force_abort_n:.1f}N"
    )


def _warm_v50_perception(policy) -> None:
    """Pay both YOLO first-inference costs during lifecycle configuration."""

    from .sfp_plug_pose import PlugPoseView

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    # A no-detection result is expected for black pixels; an exception is not.
    policy._pc.detect_nic(image, conf_thresh=0.99)
    policy._v50_plug_estimator.detect_views(
        [
            PlugPoseView(
                camera_name="v50_warmup",
                image_bgr=image,
                K=np.array(
                    [[500.0, 0.0, 320.0], [0.0, 500.0, 320.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                T_world_from_camera=np.eye(4),
                stamp_s=0.0,
                frame_id="v50-warmup",
            )
        ]
    )
    policy.get_logger().info(
        "[v50] port and plug YOLO first-inference warmup completed during configure"
    )


def _message_stamp(message):
    header = getattr(message, "header", None)
    return getattr(header, "stamp", None)


def _observation_stamp_s(observation) -> Optional[float]:
    from .sfp_plug_pose import stamp_to_seconds

    values = []
    for name in ("left_image", "center_image", "right_image"):
        try:
            value = stamp_to_seconds(_message_stamp(getattr(observation, name, None)))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(float(value))
    return max(values) if values else None


def _normalize_event(value: object) -> str:
    return str(value or "").strip().strip("/")


def _plug_views_from_observation(policy, observation):
    from .sfp_plug_pose import PlugPoseView, stamp_to_seconds

    messages = {
        "left_camera": getattr(observation, "left_image", None),
        "center_camera": getattr(observation, "center_image", None),
        "right_camera": getattr(observation, "right_image", None),
    }
    result = []
    for camera, (image_bgr, K, T_cam_from_base) in policy._build_views(
        observation
    ).items():
        try:
            stamp_s = stamp_to_seconds(_message_stamp(messages[camera]))
        except (TypeError, ValueError):
            continue
        result.append(
            PlugPoseView(
                camera_name=camera,
                image_bgr=image_bgr,
                K=K,
                T_world_from_camera=policy._pc.invert_transform(T_cam_from_base),
                stamp_s=float(stamp_s),
                frame_id=f"{camera}:{float(stamp_s):.9f}",
            )
        )
    return result


def prime_v50_plug_pose(policy, get_observation, move_robot) -> bool:
    """Observe the plug before port selection and calibrate its TCP transform.

    Port consensus ranks multiple detected cages by distance to the plug.  v50
    therefore primes the direct plug estimate first, so even candidate selection
    uses measured plug geometry rather than the legacy grasp assumption.  The
    robot remains stationary while the subsequent port consensus is collected,
    making this same fresh world plug pose valid for initial relative alignment.
    """

    policy._v50_grasp_transform = None
    policy._v50_pending_world_plug = None
    deadline = time.monotonic() + 2.0
    observation = None
    while time.monotonic() < deadline:
        policy._enforce_action_deadline(move_robot)
        candidate = get_observation()
        if candidate is not None and _observation_stamp_s(candidate) is not None:
            observation = candidate
            break
        time.sleep(0.02)
    if observation is None:
        policy.get_logger().error("[v50] no timestamped observation for plug priming")
        return False
    views = _plug_views_from_observation(policy, observation)
    if len(views) < 2:
        policy.get_logger().error("[v50] fewer than two views for plug priming")
        return False
    from .sfp_plug_pose import stamp_to_seconds

    now_s = stamp_to_seconds(policy._parent_node.get_clock().now())
    estimate = policy._v50_plug_estimator.estimate_multiview(
        views,
        now_s=now_s,
        max_age_s=policy._v50_config.plug_max_age_s,
    )
    if estimate is None:
        policy.get_logger().error(
            "[v50] direct plug priming failed; no fixed-grasp control fallback"
        )
        return False
    tcp_pos, tcp_quat = policy._tcp()
    policy._v50_grasp_transform = solve_tip_in_tcp(
        tcp_pos,
        tcp_quat,
        estimate.position_world,
        estimate.rotation_world_from_plug,
    )
    policy._v50_pending_world_plug = estimate
    policy.get_logger().info(
        f"[v50] direct plug primed before port selection: confidence="
        f"{estimate.confidence:.3f} views={estimate.view_count} "
        f"reproj={estimate.reprojection_error_px:.2f}px "
        f"frames={list(estimate.source_frame_ids)}"
    )
    return True


class PlugRelativeV50Controller:
    """Bounded visual -> persistent-seat -> lift/re-perceive state machine."""

    STIFFNESS = [350.0, 350.0, 500.0, 60.0, 60.0, 60.0]
    DAMPING = [90.0, 90.0, 100.0, 25.0, 25.0, 25.0]
    HOLD_STIFFNESS = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    HOLD_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

    def __init__(
        self,
        policy,
        task,
        get_observation,
        move_robot,
        send_feedback,
        *,
        port_pos,
        port_quat,
        Rp,
    ):
        self.policy = policy
        self.task = task
        self.get_observation = get_observation
        self.move_robot = move_robot
        self.send_feedback = send_feedback
        self.config = getattr(policy, "_v50_config", V50Config.from_env())
        self.port_pos = np.asarray(port_pos, dtype=np.float64).reshape(3)
        self.port_quat = np.asarray(port_quat, dtype=np.float64).reshape(4)
        self.Rp = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
        self.log = policy.get_logger()
        parent = policy._parent_node
        self.event_generation = int(getattr(parent, "_insertion_event_generation", 0))
        self.expected_event = _normalize_event(
            f"{getattr(task, 'target_module_name', '')}/{getattr(task, 'port_name', '')}"
        )
        self.last_observation_stamp = None
        self.last_accepted_plug_stamp = None

    def _event(self):
        parent = self.policy._parent_node
        generation = int(getattr(parent, "_insertion_event_generation", 0))
        value = _normalize_event(getattr(parent, "_insertion_event_value", ""))
        if generation <= self.event_generation:
            return None
        return value

    def _event_status(self):
        value = self._event()
        if value is None:
            return None
        if value == self.expected_event:
            return SEATED
        self.log.error(
            f"[v50] insertion event was for wrong port '{value}', expected "
            f"'{self.expected_event}'"
        )
        return HARD_FAILURE

    def _force_magnitude(self, observation) -> float:
        if observation is None:
            return float("nan")
        wrench = self.policy._wrench_vector(observation) - self.policy._wrench_baseline
        return float(np.linalg.norm(wrench[:3]))

    def _wrench_plug_frame(self, observation):
        if observation is None:
            nan3 = np.full(3, np.nan, dtype=np.float64)
            return nan3, nan3.copy()
        wrench_wrist = (
            self.policy._wrench_vector(observation) - self.policy._wrench_baseline
        )
        _, tcp_quat = self.policy._tcp()
        R_tcp = quaternion_to_matrix(tcp_quat)
        wrist_to_plug = self.Rp.T @ R_tcp
        return wrist_to_plug @ wrench_wrist[:3], wrist_to_plug @ wrench_wrist[3:]

    def _seat_alignment_sample(self, observation, depth, force, acc_lat, acc_tilt):
        f_plug, m_plug = self._wrench_plug_frame(observation)
        contact = bool(np.isfinite(force) and force >= self.config.contact_force_n)
        finite_wrench = bool(np.all(np.isfinite(f_plug[:2])) and np.all(np.isfinite(m_plug[:2])))
        if contact and finite_wrench:
            log_force_gain = (
                self.config.seat_align_force_gain
                if self.config.seat_align_force_gain != 0.0
                else SEAT_ALIGN_OBSERVE_FORCE_GAIN
            )
            log_moment_gain = (
                self.config.seat_align_moment_gain
                if self.config.seat_align_moment_gain != 0.0
                else SEAT_ALIGN_OBSERVE_MOMENT_GAIN
            )
            d_lat_would = -log_force_gain * f_plug[:2]
            d_tilt_would = -log_moment_gain * m_plug[:2]
            d_lat_applied = -self.config.seat_align_force_gain * f_plug[:2]
            d_tilt_applied = -self.config.seat_align_moment_gain * m_plug[:2]
        else:
            d_lat_would = np.zeros(2, dtype=np.float64)
            d_tilt_would = np.zeros(2, dtype=np.float64)
            d_lat_applied = np.zeros(2, dtype=np.float64)
            d_tilt_applied = np.zeros(2, dtype=np.float64)
            decay = self.config.seat_align_release_decay
            acc_lat = np.asarray(acc_lat, dtype=np.float64).reshape(2) * decay
            acc_tilt = np.asarray(acc_tilt, dtype=np.float64).reshape(2) * decay
        acc_lat = clamp_vector_norm(
            np.asarray(acc_lat, dtype=np.float64).reshape(2) + d_lat_applied,
            self.config.seat_align_max_lat_m,
        )
        acc_tilt = clamp_vector_norm(
            np.asarray(acc_tilt, dtype=np.float64).reshape(2) + d_tilt_applied,
            self.config.seat_align_max_tilt_rad,
        )
        return acc_lat, acc_tilt, (depth, f_plug, m_plug, d_lat_would, d_tilt_would)

    def _log_seat_wrench(self, sample, acc_lat, acc_tilt, *, summary: Optional[str] = None):
        depth, f_plug, m_plug, d_lat_would, d_tilt_would = sample
        suffix = f" summary={summary}" if summary else ""
        self.log.info(
            f"[v50] SEAT_WRENCH depth={depth*1000.0:.1f}mm "
            f"axial_N={f_plug[2]:.2f} "
            f"lat_N={np.round(f_plug[:2], 2).tolist()} "
            f"|lat|={np.linalg.norm(f_plug[:2]):.2f} "
            f"moment_Nm={np.round(m_plug[:2], 3).tolist()} "
            f"|M|={np.linalg.norm(m_plug[:2]):.3f} "
            f"nudge_would_mm={np.round(d_lat_would * 1000.0, 3).tolist()} "
            f"tilt_would_deg={np.degrees(np.linalg.norm(d_tilt_would)):.3f} "
            f"nudge_applied_mm={np.round(acc_lat * 1000.0, 3).tolist()} "
            f"tilt_applied_deg={np.degrees(np.linalg.norm(acc_tilt)):.3f}"
            f"{suffix}"
        )

    def _hold_tip(self, tip_pos, tip_rotation) -> None:
        self.policy.set_pose_target(
            self.move_robot,
            self.policy._tcp_target_for_tip(tip_pos, tip_rotation),
            stiffness=self.HOLD_STIFFNESS,
            damping=self.HOLD_DAMPING,
        )

    def _wait_new_observation(self, *, after_stamp, timeout_wall_s):
        deadline = time.monotonic() + timeout_wall_s
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            observation = self.get_observation()
            if observation is not None:
                stamp = _observation_stamp_s(observation)
                if stamp is not None and (after_stamp is None or stamp > after_stamp + 1e-9):
                    self.last_observation_stamp = stamp
                    return observation
            time.sleep(0.02)
        return None

    def _plug_views(self, observation):
        return _plug_views_from_observation(self.policy, observation)

    def _activate_plug_pose(self, observation) -> bool:
        views = self._plug_views(observation)
        if len(views) < 2:
            self.log.error("[v50] fewer than two fresh plug camera views; fail closed")
            return False
        from .sfp_plug_pose import stamp_to_seconds

        # Image age must be measured in the same ROS/simulation clock domain as
        # the image headers, never against time.monotonic().
        now_s = stamp_to_seconds(self.policy._parent_node.get_clock().now())
        estimate = self.policy._v50_plug_estimator.estimate_relative_to_port(
            views,
            self.port_pos,
            self.Rp,
            now_s=now_s,
            max_age_s=self.config.plug_max_age_s,
            min_stamp_s=self.last_accepted_plug_stamp,
        )
        if estimate is None:
            self.log.error(
                "[v50] fresh plug pose unavailable; refusing fixed-bias fallback"
            )
            return False
        tip_pos = self.port_pos + self.Rp @ np.asarray(
            estimate.translation_port, dtype=np.float64
        ).reshape(3)
        tip_rotation = self.Rp @ np.asarray(
            estimate.rotation_port_from_plug, dtype=np.float64
        ).reshape(3, 3)
        tcp_pos, tcp_quat = self.policy._tcp()
        self.policy._v50_grasp_transform = solve_tip_in_tcp(
            tcp_pos, tcp_quat, tip_pos, tip_rotation
        )
        self.last_accepted_plug_stamp = float(estimate.stamp_s)
        self.log.info(
            "[v50] fresh plug pose accepted: "
            f"delta_port_mm={np.round(estimate.translation_port * 1000.0, 2).tolist()} "
            f"confidence={estimate.confidence:.3f} views={estimate.view_count} "
            f"reproj={estimate.reprojection_error_px:.2f}px "
            f"frames={list(estimate.source_frame_ids)}"
        )
        return True

    def _activate_initial_plug_pose(self) -> bool:
        estimate = getattr(self.policy, "_v50_pending_world_plug", None)
        if estimate is None or getattr(self.policy, "_v50_grasp_transform", None) is None:
            self.log.error("[v50] no fresh primed plug pose for initial control")
            return False
        relative = self.Rp.T @ (
            np.asarray(estimate.position_world, dtype=np.float64) - self.port_pos
        )
        self.last_accepted_plug_stamp = float(estimate.stamp_s)
        self.policy._v50_pending_world_plug = None
        self.log.info(
            f"[v50] initial plug-to-port delta_mm="
            f"{np.round(relative * 1000.0, 2).tolist()} from direct plug pose"
        )
        return True

    def _tip_pose(self):
        tcp_pos, tcp_quat = self.policy._tcp()
        return self.policy._tip_from_tcp(tcp_pos, tcp_quat)

    def _errors(self):
        tip_pos, tip_rotation = self._tip_pose()
        delta = self.Rp.T @ (tip_pos - self.port_pos)
        rotation_error = axis_angle(self.Rp.T @ tip_rotation)
        return (
            float(delta[2]),
            delta[:2],
            rotation_error,
            tip_pos,
            tip_rotation,
        )

    def _align(self) -> bool:
        start = time.monotonic()
        depth, _, _, _, _ = self._errors()
        align_depth = depth
        while time.monotonic() - start < self.config.align_timeout_wall_s:
            self.policy._enforce_action_deadline(self.move_robot)
            depth, lateral_xy, rotation_error, tip_pos, _ = self._errors()
            lateral = float(np.linalg.norm(lateral_xy))
            rotation = float(np.linalg.norm(rotation_error))
            if (
                lateral <= self.config.align_lateral_tol_m
                and rotation <= self.config.align_rotation_tol_rad
            ):
                self.log.info(
                    f"[v50] plug-relative alignment valid: lateral={lateral*1000:.2f}mm "
                    f"rotation={np.degrees(rotation):.2f}deg depth={depth*1000:.1f}mm"
                )
                return True
            lateral_step = -lateral_xy
            lateral_norm = float(np.linalg.norm(lateral_step))
            if lateral_norm > self.config.align_max_lateral_step_m:
                lateral_step *= self.config.align_max_lateral_step_m / lateral_norm
            target_tip = (
                tip_pos
                + self.Rp[:, 0] * lateral_step[0]
                + self.Rp[:, 1] * lateral_step[1]
                + self.Rp[:, 2] * float(np.clip(align_depth - depth, -0.002, 0.002))
            )
            if rotation > self.config.align_max_rotation_step_rad:
                remaining = 1.0 - self.config.align_max_rotation_step_rad / rotation
                target_rotation = self.Rp @ rotation_from_axis_angle(
                    remaining * rotation_error
                )
            else:
                target_rotation = self.Rp
            self.policy.set_pose_target(
                self.move_robot,
                self.policy._tcp_target_for_tip(target_tip, target_rotation),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.error("[v50] plug-relative alignment exceeded wall-time budget")
        return False

    def _wait_for_insertion_event(self, fixed_tip) -> str:
        deadline = time.monotonic() + self.config.insertion_event_timeout_wall_s
        hard_force_since = None
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            event_status = self._event_status()
            if event_status is not None:
                return event_status
            observation = self.get_observation()
            force = self._force_magnitude(observation)
            tip_pos, _ = self._tip_pose()
            if np.isfinite(force) and force > self.config.force_abort_n:
                self._hold_tip(tip_pos, self.Rp)
                hard_force_since = hard_force_since or time.monotonic()
                if time.monotonic() - hard_force_since >= self.config.force_abort_wall_s:
                    self.log.error(
                        f"[v50] >{self.config.force_abort_n:.1f}N during event dwell"
                    )
                    return HARD_FAILURE
            else:
                hard_force_since = None
                self.policy.set_pose_target(
                    self.move_robot,
                    self.policy._tcp_target_for_tip(fixed_tip, self.Rp),
                    stiffness=self.STIFFNESS,
                    damping=self.DAMPING,
                )
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.warn("[v50] seated geometry produced no matching insertion event")
        return STALLED

    def _seat(self) -> str:
        depth, _, _, _, _ = self._errors()
        command_depth = depth
        now = time.monotonic()
        last_command_time = now
        progress = WallProgressWatch.start(depth, now, self.config)
        hard_force_since = None
        acc_lat = np.zeros(2, dtype=np.float64)
        acc_tilt = np.zeros(2, dtype=np.float64)
        last_wrench_log_time = 0.0
        slope_samples = []
        last_slope_log_time = 0.0
        stall_grace_deadline = None

        def log_seat_slope(*, summary: Optional[str] = None):
            if not slope_samples:
                return
            _, first_depth, first_force = slope_samples[0]
            _, last_depth, last_force = slope_samples[-1]
            d_depth_mm = (last_depth - first_depth) * 1000.0
            d_force = last_force - first_force
            if np.isfinite(d_depth_mm) and abs(d_depth_mm) > 1e-6:
                d_force_per_mm = d_force / d_depth_mm
            elif np.isfinite(d_force) and abs(d_force) > 1e-9:
                d_force_per_mm = float(np.copysign(np.inf, d_force))
            else:
                d_force_per_mm = 0.0
            suffix = f" summary={summary}" if summary else ""
            self.log.info(
                f"[v50] SEAT_SLOPE depth={last_depth*1000.0:.1f}mm "
                f"axial_N={last_force:.2f} dDepth_mm={d_depth_mm:.3f} "
                f"dForce_N={d_force:.2f} dForce_per_mm={d_force_per_mm:.2f}"
                f"{suffix}"
            )

        while True:
            self.policy._enforce_action_deadline(self.move_robot)
            event_status = self._event_status()
            if event_status is not None:
                return event_status
            observation = self.get_observation()
            depth, lateral_xy, rotation_error, tip_pos, _ = self._errors()
            lateral = float(np.linalg.norm(lateral_xy))
            rotation = float(np.linalg.norm(rotation_error))
            force = self._force_magnitude(observation)
            now = time.monotonic()
            seat_wrench_sample = None
            if self.config.seat_align_enable:
                acc_lat, acc_tilt, seat_wrench_sample = self._seat_alignment_sample(
                    observation, depth, force, acc_lat, acc_tilt
                )
                if now - last_wrench_log_time >= SEAT_WRENCH_LOG_PERIOD_S:
                    self._log_seat_wrench(seat_wrench_sample, acc_lat, acc_tilt)
                    last_wrench_log_time = now
            if seat_wrench_sample is not None:
                axial_force = float(seat_wrench_sample[1][2])
            else:
                f_plug, _ = self._wrench_plug_frame(observation)
                axial_force = float(f_plug[2])
            slope_samples.append((now, depth, axial_force))
            while slope_samples and now - slope_samples[0][0] > SEAT_SLOPE_WINDOW_S:
                slope_samples.pop(0)
            if now - last_slope_log_time >= SEAT_SLOPE_LOG_PERIOD_S:
                log_seat_slope()
                last_slope_log_time = now

            if np.isfinite(force) and force > self.config.force_abort_n:
                self._hold_tip(tip_pos, self.Rp)
                hard_force_since = hard_force_since or now
                if now - hard_force_since >= self.config.force_abort_wall_s:
                    self.log.error(
                        f"[v50] sustained force {force:.1f}N exceeds "
                        f"{self.config.force_abort_n:.1f}N; held and aborted"
                    )
                    return HARD_FAILURE
                self.policy.sleep_for(self.config.command_dt_sim_s)
                continue
            hard_force_since = None

            if lateral > self.config.lateral_safety_m or rotation > self.config.rotation_safety_rad:
                self.log.warn(
                    f"[v50] wedge geometry: lateral={lateral*1000:.1f}mm "
                    f"rotation={np.degrees(rotation):.1f}deg"
                )
                if seat_wrench_sample is not None:
                    self._log_seat_wrench(
                        seat_wrench_sample, acc_lat, acc_tilt, summary="stall"
                    )
                log_seat_slope(summary="stall")
                return STALLED

            if depth >= self.config.seat_candidate_depth_m:
                fixed_tip = self.port_pos + self.Rp[:, 2] * INSERT_DEPTH_M
                return self._wait_for_insertion_event(fixed_tip)

            stalled_now = progress.stalled(depth, now)
            if stall_grace_deadline is not None and not stalled_now:
                self.log.info(
                    f"[v50] stall grace recovered: depth={depth*1000:.1f}mm "
                    f"best={progress.best_depth*1000:.1f}mm"
                )
                stall_grace_deadline = None
            if stalled_now:
                if self.config.seat_stall_grace_s > 0.0:
                    if stall_grace_deadline is None:
                        stall_grace_deadline = now + self.config.seat_stall_grace_s
                        self.log.warn(
                            f"[v50] wall-time stall grace: depth={depth*1000:.1f}mm "
                            f"best={progress.best_depth*1000:.1f}mm force={force:.2f}N "
                            f"grace_s={self.config.seat_stall_grace_s:.1f}"
                        )
                    if now >= stall_grace_deadline:
                        self.log.warn(
                            f"[v50] wall-time stall: depth={depth*1000:.1f}mm "
                            f"best={progress.best_depth*1000:.1f}mm force={force:.2f}N"
                        )
                        if seat_wrench_sample is not None:
                            self._log_seat_wrench(
                                seat_wrench_sample, acc_lat, acc_tilt, summary="stall"
                            )
                        log_seat_slope(summary="stall")
                        return STALLED
                else:
                    self.log.warn(
                        f"[v50] wall-time stall: depth={depth*1000:.1f}mm "
                        f"best={progress.best_depth*1000:.1f}mm force={force:.2f}N"
                    )
                    if seat_wrench_sample is not None:
                        self._log_seat_wrench(
                            seat_wrench_sample, acc_lat, acc_tilt, summary="stall"
                        )
                    log_seat_slope(summary="stall")
                    return STALLED

            depth_config = self.config
            if (
                self.config.seat_mouth_speed_scale != 1.0
                and depth < self.config.seat_mouth_zone_m
            ):
                depth_config = replace(
                    self.config,
                    free_speed_m_s=(
                        self.config.free_speed_m_s
                        * self.config.seat_mouth_speed_scale
                    ),
                    contact_speed_m_s=(
                        self.config.contact_speed_m_s
                        * self.config.seat_mouth_speed_scale
                    ),
                )
            command_depth = next_persistent_depth(
                depth,
                command_depth,
                now - last_command_time,
                force,
                depth_config,
            )
            last_command_time = now
            target_tip = self.port_pos + self.Rp[:, 2] * command_depth
            target_rotation = self.Rp
            if self.config.seat_align_enable and (
                np.any(acc_lat != 0.0) or np.any(acc_tilt != 0.0)
            ):
                target_tip = (
                    target_tip
                    + self.Rp[:, 0] * acc_lat[0]
                    + self.Rp[:, 1] * acc_lat[1]
                )
                target_rotation = self.Rp @ rotation_from_axis_angle(
                    np.array([acc_tilt[0], acc_tilt[1], 0.0], dtype=np.float64)
                )
            self.policy.set_pose_target(
                self.move_robot,
                self.policy._tcp_target_for_tip(target_tip, target_rotation),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)

    def _hold_legacy_safe_pose(self):
        """Hold the TCP itself when no visually calibrated plug transform exists."""

        tcp_pos, tcp_quat = self.policy._tcp()
        from .rl_insert_contract import sfp_tip_pose_from_tcp

        tip_pos, tip_rotation = sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)
        self._hold_tip(tip_pos, tip_rotation)

    def run(self) -> bool:
        self.send_feedback("v50 fresh plug-to-port perception")
        if not self._activate_initial_plug_pose():
            self._hold_legacy_safe_pose()
            return False

        if not self._align():
            return False
        self.send_feedback("v50 persistent force-regulated seating")
        outcome = self._seat()
        if outcome == SEATED:
            self.log.info(
                f"[v50] matching insertion event confirmed for {self.expected_event}"
            )
            self.send_feedback("correct-port insertion event confirmed")
            return True
        return False


def run_v50_script(
    policy,
    task,
    get_observation,
    move_robot,
    send_feedback,
    *,
    port_pos,
    port_quat,
    Rp,
) -> bool:
    return PlugRelativeV50Controller(
        policy,
        task,
        get_observation,
        move_robot,
        send_feedback,
        port_pos=port_pos,
        port_quat=port_quat,
        Rp=Rp,
    ).run()


__all__ = [
    "HARD_FAILURE",
    "INSERT_DEPTH_M",
    "PlugRelativeV50Controller",
    "SEATED",
    "STALLED",
    "V50Config",
    "WallProgressWatch",
    "configure_v50",
    "next_persistent_depth",
    "prime_v50_plug_pose",
    "run_v50_script",
    "solve_tip_in_tcp",
    "tcp_for_tip_transform",
    "tip_from_tcp_transform",
    "v50_tcp_pose_for_tip",
    "v50_tip_from_tcp",
]
