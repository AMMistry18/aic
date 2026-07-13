"""Pure numerical contract shared by SFP distillation and ROS deployment.

This module intentionally depends only on NumPy.  Keeping pose reconstruction,
the 69-vector layout, and action conversion here prevents the MuJoCo wrapper and
Gazebo policy from independently choosing frame, quaternion, or sign conventions.
"""
from __future__ import annotations

import numpy as np


OBS_DIM = 69
ACTION_DIM = 6
HOME_QPOS = np.array(
    [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110],
    dtype=np.float64,
)
DEPLOY_POS_SCALE = np.array([0.0015, 0.0015, 0.0035], dtype=np.float64)
DEPLOY_ROT_SCALE = np.array([0.08, 0.08, 0.12], dtype=np.float64)
SFP_TIP_IN_TCP_POS = np.array(
    [-0.0017771781, -0.0188744563, 0.0547221980], dtype=np.float64
)
# Refined 2026-07-13 from two Flowstate calib dumps (q_tcp^-1 * q_port averaged),
# a ~1.16 deg nudge toward "kinematic tip square to the perceived port at
# handoff". Prior: [0.9852867415, 0.1688620346, -0.0042579615, -0.0260292145].
SFP_TIP_IN_TCP_QUAT = np.array(
    [0.9863980666, 0.1620801968, 0.0030413, -0.0271958552],
    dtype=np.float64,
)


def canonicalize_quaternion(q) -> np.ndarray:
    """Normalize a wxyz quaternion and choose one deterministic sign."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"invalid quaternion {q}")
    q = q / norm
    for value in q:
        if abs(float(value)) > 1e-12:
            if value < 0.0:
                q = -q
            break
    return q


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = canonicalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    w = np.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    if w > 1e-7:
        q = np.array(
            [
                w,
                (R[2, 1] - R[1, 2]) / (4.0 * w),
                (R[0, 2] - R[2, 0]) / (4.0 * w),
                (R[1, 0] - R[0, 1]) / (4.0 * w),
            ]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        v = np.zeros(3)
        v[i] = s / 4.0
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = np.array([(R[k, j] - R[j, k]) / s, *v])
    return canonicalize_quaternion(q)


def matrix_to_rotvec(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-6:
        q = matrix_to_quat(R)
        axis = q[1:]
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis * angle


def port_frame(port_quat) -> np.ndarray:
    """Return canonical columns ``[lat_x, lat_y, inward_insert_axis]``."""
    return quat_to_matrix(port_quat)


def sfp_tip_pose_from_tcp(tcp_pos, tcp_quat) -> tuple[np.ndarray, np.ndarray]:
    R_tcp = quat_to_matrix(tcp_quat)
    R_tip = R_tcp @ quat_to_matrix(SFP_TIP_IN_TCP_QUAT)
    tip_pos = np.asarray(tcp_pos, dtype=np.float64).reshape(3) + R_tcp @ SFP_TIP_IN_TCP_POS
    return tip_pos, R_tip


def tcp_pose_for_sfp_tip(tip_pos, tip_rotation) -> tuple[np.ndarray, np.ndarray]:
    R_tip = np.asarray(tip_rotation, dtype=np.float64).reshape(3, 3)
    R_tcp = R_tip @ quat_to_matrix(SFP_TIP_IN_TCP_QUAT).T
    tcp_pos = np.asarray(tip_pos, dtype=np.float64).reshape(3) - R_tcp @ SFP_TIP_IN_TCP_POS
    return tcp_pos, matrix_to_quat(R_tcp)


def deploy_action_delta(action, port_quat) -> tuple[np.ndarray, np.ndarray]:
    """Convert one deploy action to world translation and rotation-vector deltas."""
    action = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
    frame = port_frame(port_quat)
    return (
        frame @ (action[:3] * DEPLOY_POS_SCALE),
        frame @ (action[3:] * DEPLOY_ROT_SCALE),
    )


def guided_tip_target(
    *,
    port_pos,
    port_quat,
    current_depth,
    planned_depth,
    insert_depth=0.045,
    outside_step=0.0015,
    inside_step=0.00075,
    outside_switch_depth=-0.003,
    max_lead=0.004,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return an absolute, centered tip target for guarded insertion.

    The target remains on the perceived port axis with the port orientation.
    Planned depth advances more slowly around contact and cannot lead the
    measured tip by more than ``max_lead``. This avoids command integration and
    lateral/rotation drift while still allowing the impedance controller to
    settle behind the trajectory.
    """
    current_depth = float(current_depth)
    planned_depth = float(planned_depth)
    insert_depth = float(insert_depth)
    max_lead = float(max_lead)
    values = np.array(
        [current_depth, planned_depth, insert_depth, outside_step, inside_step, max_lead],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or outside_step <= 0 or inside_step <= 0:
        raise ValueError(f"invalid guided insertion parameters: {values}")
    if max_lead <= 0:
        raise ValueError("max_lead must be positive")

    step = outside_step if current_depth < outside_switch_depth else inside_step
    next_plan = min(max(planned_depth, current_depth) + step, insert_depth)
    target_depth = min(next_plan, current_depth + max_lead, insert_depth)
    frame = port_frame(port_quat)
    target_pos = (
        np.asarray(port_pos, dtype=np.float64).reshape(3)
        + frame[:, 2] * target_depth
    )
    return target_pos, frame, float(next_plan), float(target_depth)


def build_observation69(
    *,
    joint_pos,
    joint_vel,
    tcp_pos,
    tcp_quat,
    tcp_linear_velocity_world,
    tcp_angular_velocity_world,
    port_pos,
    port_quat,
    tip_pos,
    tip_rotation,
    wrench,
    last_action,
    home_qpos=HOME_QPOS,
    wrench_mode="zero",
    wrench_baseline=None,
) -> np.ndarray:
    """Assemble the complete state-student observation in one canonical place."""
    if wrench_mode not in ("raw", "zero", "baseline"):
        raise ValueError("wrench_mode must be raw, zero, or baseline")

    frame = port_frame(port_quat)
    tcp_quat = canonicalize_quaternion(tcp_quat)
    port_quat = canonicalize_quaternion(port_quat)
    tip_rotation = np.asarray(tip_rotation, dtype=np.float64).reshape(3, 3)
    port_pos = np.asarray(port_pos, dtype=np.float64).reshape(3)
    tip_pos = np.asarray(tip_pos, dtype=np.float64).reshape(3)

    delta = frame.T @ (tip_pos - port_pos)
    relative_rotation = frame.T @ tip_rotation
    rotation_error = matrix_to_rotvec(relative_rotation)
    hint = np.clip(
        np.concatenate(
            [
                [-6.0 * delta[0], -6.0 * delta[1], -8.0 * delta[2]],
                -3.0 * rotation_error,
            ]
        ),
        -1.0,
        1.0,
    )

    wrench = np.asarray(wrench, dtype=np.float64).reshape(6).copy()
    if wrench_mode == "zero":
        wrench[:] = 0.0
    elif wrench_mode == "baseline":
        if wrench_baseline is None:
            raise ValueError("wrench_baseline is required in baseline mode")
        wrench -= np.asarray(wrench_baseline, dtype=np.float64).reshape(6)

    obs = np.concatenate(
        [
            np.asarray(joint_pos, dtype=np.float64).reshape(6)
            - np.asarray(home_qpos, dtype=np.float64).reshape(6),
            np.asarray(joint_vel, dtype=np.float64).reshape(6),
            np.asarray(tcp_pos, dtype=np.float64).reshape(3),
            tcp_quat,
            frame.T @ np.asarray(tcp_linear_velocity_world, dtype=np.float64).reshape(3),
            frame.T @ np.asarray(tcp_angular_velocity_world, dtype=np.float64).reshape(3),
            port_pos,
            port_quat,
            delta,
            rotation_error,
            hint,
            hint,
            [1.0],
            0.1 * wrench,
            np.asarray(last_action, dtype=np.float64).reshape(6),
            relative_rotation[:, 0],
            relative_rotation[:, 2],
        ]
    ).astype(np.float32)
    if obs.shape != (OBS_DIM,) or not np.all(np.isfinite(obs)):
        raise ValueError(f"invalid deploy observation: shape={obs.shape} obs={obs}")
    return obs


__all__ = [
    "ACTION_DIM",
    "DEPLOY_POS_SCALE",
    "DEPLOY_ROT_SCALE",
    "HOME_QPOS",
    "OBS_DIM",
    "SFP_TIP_IN_TCP_POS",
    "SFP_TIP_IN_TCP_QUAT",
    "build_observation69",
    "canonicalize_quaternion",
    "deploy_action_delta",
    "guided_tip_target",
    "matrix_to_quat",
    "matrix_to_rotvec",
    "port_frame",
    "quat_to_matrix",
    "sfp_tip_pose_from_tcp",
    "tcp_pose_for_sfp_tip",
]
