# MDP functions for insertion: port-frame observations, shaped rewards,
# success/failure terminations.
#
# All functions take ManagerBasedRLEnv and return tensors of shape (num_envs, ...).
# Port pose is read from env.scene[scene_name].data.root_state_w which is the
# live PhysX state -> always matches the current randomized spawn.

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Target registry: scene_name + insertion depth per target key.
# Mirrors TARGETS in aic_insert_env_cfg.py. Kept in sync manually; small enough
# not to be a burden.
# ---------------------------------------------------------------------------

_TARGETS = {
    "sc": {
        "scene_name": "sc_port",
        "depth": 0.016,
        "port_type": "sc",
        "entrance_parent_pos": (0.0, -0.002, 0.0),
        "entrance_parent_rpy": (1.5708, 3.14159, 0.0),
        "entrance_local_pos": (0.0, 0.0, -0.01564),
        "entrance_local_rpy": (0.0, 0.0, 0.0),
        "tip_body_name": "sc_tip_link",
    },
    "sc2": {
        "scene_name": "sc_port_2",
        "depth": 0.016,
        "port_type": "sc",
        "entrance_parent_pos": (0.0, -0.002, 0.0),
        "entrance_parent_rpy": (1.5708, 3.14159, 0.0),
        "entrance_local_pos": (0.0, 0.0, -0.01564),
        "entrance_local_rpy": (0.0, 0.0, 0.0),
        "tip_body_name": "sc_tip_link",
    },
    "sfp": {
        "scene_name": "nic_card",
        "depth": 0.046,
        "port_type": "sfp",
        "entrance_parent_pos": (0.01295, -0.031572, 0.00501),
        "entrance_parent_rpy": (4.69895, 0.0, 0.0),
        "entrance_local_pos": (0.0, 0.0, -0.0458),
        "entrance_local_rpy": (0.0, 0.0, 0.0),
        "tip_body_name": "sfp_tip_link",
    },
}

# Exact tip frame used by the Isaac controller body_offset.
_TIP_IN_WRIST = {
    "sc": {
        "pos": (0.04026, 0.00907, 0.14939),
        "quat_wxyz": (0.85472, 0.01261, -0.51889, -0.00920),
    },
    "sfp": {
        "pos": (0.05631, 0.00137, 0.14857),
        "quat_wxyz": (0.86526, 0.01717, -0.50059, -0.02774),
    },
}

# Arm-only pre-insertion joint seeds (UR5e 6-DoF arm). These are warm-start
# poses near the task board, not final insertion poses.
_PREINSERT_ARM_JOINTS = {
    "sc":  [0.22, -1.15, -1.78, -1.65, 1.57, 1.42],
    "sc2": [0.28, -1.13, -1.76, -1.63, 1.57, 1.40],
    "sfp": [0.12, -1.08, -1.70, -1.55, 1.57, 1.35],
}

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _seated_tip_quat_port(num_envs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Desired tip orientation in port frame for a seated connector.

    The SC tip frame's +Z points back along the connector body, so in a safe
    seated pose tip +Z points out of the port while the tip body extends along
    local -Z into the receptacle. Tip +X stays aligned with port +X to fix twist.
    """
    return torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype).expand(num_envs, -1)


def _port_pose_w(env: "ManagerBasedRLEnv", target_key: str):
    """Returns the true port entrance pose in world frame."""
    cfg = _TARGETS[target_key]
    port: RigidObject = env.scene[cfg["scene_name"]]
    state = port.data.root_state_w
    root_pos_w = state[:, :3]
    root_quat_w = state[:, 3:7]
    device = root_pos_w.device
    dtype = root_pos_w.dtype
    num_envs = root_pos_w.shape[0]

    entrance_parent_pos = torch.tensor(cfg["entrance_parent_pos"], device=device, dtype=dtype).expand(num_envs, -1)
    entrance_parent_rpy = torch.tensor(cfg["entrance_parent_rpy"], device=device, dtype=dtype).expand(num_envs, -1)
    entrance_parent_quat = quat_from_euler_xyz(
        entrance_parent_rpy[:, 0], entrance_parent_rpy[:, 1], entrance_parent_rpy[:, 2]
    )
    entrance_local_pos = torch.tensor(cfg["entrance_local_pos"], device=device, dtype=dtype).expand(num_envs, -1)
    entrance_local_rpy = torch.tensor(cfg["entrance_local_rpy"], device=device, dtype=dtype).expand(num_envs, -1)
    entrance_local_quat = quat_from_euler_xyz(
        entrance_local_rpy[:, 0], entrance_local_rpy[:, 1], entrance_local_rpy[:, 2]
    )

    parent_pos_w, parent_quat_w = combine_frame_transforms(
        root_pos_w, root_quat_w, entrance_parent_pos, entrance_parent_quat
    )
    entrance_pos_w, entrance_quat_w = combine_frame_transforms(
        parent_pos_w, parent_quat_w, entrance_local_pos, entrance_local_quat
    )
    return entrance_pos_w, entrance_quat_w


def _desired_seated_tip_pose_w(
    env: "ManagerBasedRLEnv",
    target_key: str,
    depth: float = 0.0,
):
    """Desired seated tip pose in world frame.

    Depth is positive into the port and is expressed along local -Z of the port
    entrance frame.
    """
    port_pos_w, port_quat_w = _port_pose_w(env, target_key)
    num_envs = port_pos_w.shape[0]
    local_pos = torch.zeros((num_envs, 3), device=port_pos_w.device, dtype=port_pos_w.dtype)
    local_pos[:, 2] = -depth
    local_quat = _seated_tip_quat_port(num_envs, port_pos_w.device, port_pos_w.dtype)
    return combine_frame_transforms(port_pos_w, port_quat_w, local_pos, local_quat)


def _plug_tip_pose_w(env: "ManagerBasedRLEnv", target_key: str):
    """Plug tip pose in world frame.

    Use the same wrist-to-tip transform as the IK controller body_offset. The
    unified articulation also contains cable/plug bodies, but those are not the
    controller TCP in this task and can be far from the insertion frame.
    """
    if hasattr(env, "_kinematic_tip_pos_w") and hasattr(env, "_kinematic_tip_quat_w"):
        return env._kinematic_tip_pos_w, env._kinematic_tip_quat_w

    cfg = _TARGETS[target_key]
    robot: Articulation = env.scene["robot"]
    tip_cfg = _TIP_IN_WRIST[cfg["port_type"]]
    ee_idx = robot.data.body_names.index("wrist_3_link")
    wrist_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    wrist_quat_w = robot.data.body_quat_w[:, ee_idx, :]

    device = wrist_pos_w.device
    num_envs = wrist_pos_w.shape[0]
    offset = torch.tensor(tip_cfg["pos"], device=device, dtype=wrist_pos_w.dtype).expand(num_envs, -1)
    tip_quat_local = torch.tensor(tip_cfg["quat_wxyz"], device=device, dtype=wrist_pos_w.dtype).expand(num_envs, -1)
    return combine_frame_transforms(wrist_pos_w, wrist_quat_w, offset, tip_quat_local)


def _tip_in_port_frame(env: "ManagerBasedRLEnv", target_key: str):
    """Tip pose and pose error expressed in the true port entrance frame."""
    tip_pos_w, tip_quat_w = _plug_tip_pose_w(env, target_key)
    port_pos_w, port_quat_w = _port_pose_w(env, target_key)
    delta_port, rel_quat = subtract_frame_transforms(port_pos_w, port_quat_w, tip_pos_w, tip_quat_w)
    desired_tip_pos_w, desired_tip_quat_w = _desired_seated_tip_pose_w(env, target_key)
    _, rot_err_w = compute_pose_error(
        tip_pos_w, tip_quat_w, desired_tip_pos_w, desired_tip_quat_w, rot_error_type="axis_angle"
    )
    rot_err_port = quat_rotate_inverse(port_quat_w, rot_err_w)
    return delta_port, rel_quat, rot_err_port, tip_pos_w, tip_quat_w, port_pos_w, port_quat_w


# ---------------------------------------------------------------------------
# Camera-perception helpers (SC/SC2)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rot_np(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _triangulate_dlt(points_2d: list[tuple[float, float]], proj_mats: list[np.ndarray]) -> np.ndarray | None:
    if len(points_2d) < 2 or len(points_2d) != len(proj_mats):
        return None
    A = []
    for (u, v), P in zip(points_2d, proj_mats):
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    A = np.asarray(A)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X_h = vt[-1]
    if abs(X_h[3]) < 1e-8:
        return None
    return (X_h[:3] / X_h[3]).astype(np.float32)


def _detect_sc_centroid(bgr: np.ndarray) -> tuple[float, float] | None:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([90, 80, 60], dtype=np.uint8),
        np.array([130, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    best_idx, best_area = -1, -1
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 15 <= area <= 50000 and area > best_area:
            best_idx, best_area = i, area
    if best_idx < 0:
        return None
    cx, cy = centroids[best_idx]
    return float(cx), float(cy)


def _perceived_port_pose_w(env: "ManagerBasedRLEnv", target_key: str):
    """Port pose estimated from camera perception.

    For SC/SC2: triangulates SC centroid from left/center/right RGB.
    For SFP: falls back to scene pose (until YOLO keypoint integration is added).
    """
    port_p_gt, port_q = _port_pose_w(env, target_key)
    if target_key not in ("sc", "sc2"):
        return port_p_gt, port_q

    step_count = int(getattr(env, "common_step_counter", 0))
    cache_key = f"_sc_perception_cache_{target_key}"
    step_key = f"_sc_perception_step_{target_key}"
    memory_key = f"_sc_perception_memory_{target_key}"
    valid_key = f"_sc_perception_valid_{target_key}"

    if getattr(env, step_key, -1) == step_count and hasattr(env, cache_key):
        return getattr(env, cache_key), port_q

    device = port_p_gt.device
    num_envs = port_p_gt.shape[0]
    cam_names = ["left_camera", "center_camera", "right_camera"]
    est = port_p_gt.clone()
    valid = torch.zeros(num_envs, device=device, dtype=torch.bool)

    try:
        cams = [env.scene[name] for name in cam_names]
    except KeyError:
        valid = torch.ones(num_envs, device=device, dtype=torch.bool)
        setattr(env, cache_key, est)
        setattr(env, step_key, step_count)
        setattr(env, valid_key, valid)
        return est, port_q

    for env_id in range(num_envs):
        pts = []
        Ps = []
        for cam in cams:
            rgb = cam.data.output.get("rgb", None)
            if rgb is None:
                continue
            img = rgb[env_id].detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            c = _detect_sc_centroid(bgr)
            if c is None:
                continue
            K = cam.data.intrinsic_matrices[env_id].detach().cpu().numpy().astype(np.float64)
            pos = cam.data.pos_w[env_id].detach().cpu().numpy().astype(np.float64)
            # PerceptionInsert uses optical-frame TF; use ROS convention camera
            # orientation here for consistent projection geometry.
            quat_tensor = (
                cam.data.quat_w_ros[env_id]
                if hasattr(cam.data, "quat_w_ros")
                else cam.data.quat_w_world[env_id]
            )
            quat = quat_tensor.detach().cpu().numpy().astype(np.float64)
            R_wc = _quat_wxyz_to_rot_np(quat)
            R_cw = R_wc.T
            t_cw = -R_cw @ pos
            P = K @ np.concatenate([R_cw, t_cw[:, None]], axis=1)
            pts.append(c)
            Ps.append(P)
        X = _triangulate_dlt(pts, Ps)
        if X is None:
            continue
        est[env_id] = torch.tensor(X, device=device, dtype=est.dtype)
        valid[env_id] = True

    # Hold last valid perception per env to avoid jumps/dropouts.
    if not hasattr(env, memory_key):
        setattr(env, memory_key, port_p_gt.clone())
    memory = getattr(env, memory_key)
    memory[valid] = est[valid]
    setattr(env, memory_key, memory)
    est = memory.clone()

    setattr(env, cache_key, est)
    setattr(env, step_key, step_count)
    setattr(env, valid_key, valid)
    return est, port_q


def _tip_in_port_frame_perceived(env: "ManagerBasedRLEnv", target_key: str):
    tip_pos_w, tip_quat_w = _plug_tip_pose_w(env, target_key)
    port_pos_w, port_quat_w = _perceived_port_pose_w(env, target_key)
    delta_port, rel_quat = subtract_frame_transforms(port_pos_w, port_quat_w, tip_pos_w, tip_quat_w)
    num_envs = port_pos_w.shape[0]
    local_pos = torch.zeros((num_envs, 3), device=port_pos_w.device, dtype=port_pos_w.dtype)
    local_quat = _seated_tip_quat_port(num_envs, port_pos_w.device, port_pos_w.dtype)
    desired_tip_pos_w, desired_tip_quat_w = combine_frame_transforms(
        port_pos_w, port_quat_w, local_pos, local_quat
    )
    _, rot_err_w = compute_pose_error(
        tip_pos_w, tip_quat_w, desired_tip_pos_w, desired_tip_quat_w, rot_error_type="axis_angle"
    )
    rot_err_port = quat_rotate_inverse(port_quat_w, rot_err_w)
    return delta_port, rel_quat, rot_err_port, tip_pos_w, tip_quat_w, port_pos_w, port_quat_w


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def _arm_joint_ids(env: "ManagerBasedRLEnv"):
    robot: Articulation = env.scene["robot"]
    if not hasattr(env, "_aic_insert_arm_joint_ids"):
        arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
        env._aic_insert_arm_joint_ids = arm_ids
    return env._aic_insert_arm_joint_ids


def arm_joint_pos_rel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Six UR arm joint positions relative to their default reset positions."""
    robot: Articulation = env.scene["robot"]
    joint_ids = _arm_joint_ids(env)
    return robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]


def arm_joint_vel_rel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Six UR arm joint velocities."""
    robot: Articulation = env.scene["robot"]
    joint_ids = _arm_joint_ids(env)
    return robot.data.joint_vel[:, joint_ids]


def port_pose_obs(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Port pose in robot base frame, shape (N, 7) [xyz, wxyz].

    The robot base (env origin) is already the reference for env.scene.env_origins.
    We return world-frame pose minus env origin — that's base-relative because
    the robot is spawned at a fixed offset from the env origin.
    """
    port_p, port_q = _port_pose_w(env, target_key)
    port_p_env = port_p - env.scene.env_origins
    return torch.cat([port_p_env, port_q], dim=-1)


def tip_to_port_delta(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """6D privileged pose error in port frame: [xyz, axis_angle]."""
    delta_port, _, rot_err_port, *_ = _tip_in_port_frame(env, target_key)
    return torch.cat([delta_port, rot_err_port], dim=-1)


def tip_pose_error_port(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """6D pose error of the plug tip in the true port entrance frame."""
    return tip_to_port_delta(env, target_key)


def tip_axes_port(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Plug body axes expressed in port frame for robust rotation learning."""
    _, rel_quat, _, *_ = _tip_in_port_frame(env, target_key)
    device = rel_quat.device
    n_env = rel_quat.shape[0]
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=rel_quat.dtype).expand(n_env, -1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=rel_quat.dtype).expand(n_env, -1)
    plug_x_port = quat_rotate(rel_quat, x_axis)
    plug_z_port = quat_rotate(rel_quat, z_axis)
    return torch.cat([plug_x_port, plug_z_port], dim=-1)


def port_pose_obs_perceived(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Perception-derived port pose in base frame, shape (N,7)."""
    port_p, port_q = _perceived_port_pose_w(env, target_key)
    port_p_env = port_p - env.scene.env_origins
    return torch.cat([port_p_env, port_q], dim=-1)


def tip_to_port_delta_perceived(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """6D perceived tip-to-port delta in port frame: [xyz, axis_angle]."""
    delta_port, _, rot_err_port, *_ = _tip_in_port_frame_perceived(env, target_key)
    return torch.cat([delta_port, rot_err_port], dim=-1)


def perception_action_hint(
    env: "ManagerBasedRLEnv",
    target_key: str,
    xy_gain: float = 6.0,
    z_gain: float = 8.0,
    rot_gain: float = 3.0,
) -> torch.Tensor:
    """Camera-perception servo hint action in normalized action space.

    This is not directly applied as control; it is fed as an observation prior
    so PPO can condition its action on a perception-guided corrective hint.
    """
    delta_port, _, rot_err_port, _, _, _, port_q = _tip_in_port_frame_perceived(env, target_key)
    robot: Articulation = env.scene["robot"]
    rot_err_world = quat_rotate(port_q, rot_err_port)
    rot_err_root = quat_rotate_inverse(robot.data.root_quat_w, rot_err_world)
    hint = torch.zeros((delta_port.shape[0], 6), device=delta_port.device, dtype=delta_port.dtype)
    hint[:, 0] = torch.clamp(-xy_gain * delta_port[:, 0], -1.0, 1.0)
    hint[:, 1] = torch.clamp(-xy_gain * delta_port[:, 1], -1.0, 1.0)
    hint[:, 2] = torch.clamp(-z_gain * delta_port[:, 2], -1.0, 1.0)
    hint[:, 3:] = torch.clamp(rot_gain * rot_err_root, -1.0, 1.0)
    return hint


def scripted_insert_action_hint(
    env: "ManagerBasedRLEnv",
    target_key: str,
    xy_gate: float = 0.010,
    axis_gate: float = 0.45,
    twist_gate: float = 0.45,
    xy_step: float = 0.0002,
    z_step: float = 0.0002,
    z_above: float = 0.002,
    target_depth: float = 0.009,
    yaw_gain: float = 0.35,
    rot_step: tuple[float, float, float] = (0.010, 0.010, 0.015),
    action_scale: tuple[float, float, float, float, float, float] = (0.0015, 0.0015, 0.014, 0.05, 0.05, 0.08),
) -> torch.Tensor:
    """Stateful phased expert action for bootstrapping final insertion.

    The command is expressed in the raw action space expected by the relative
    IK action: world-frame translation deltas divided by action scale.
    """
    delta_port, _, rot_err_port, _, _, _, port_q = _tip_in_port_frame(env, target_key)
    robot: Articulation = env.scene["robot"]
    xy = torch.norm(delta_port[:, :2], dim=-1)
    axis = plug_port_axis_alignment(env, target_key)
    twist = plug_port_twist_alignment(env, target_key)
    n = delta_port.shape[0]

    phase_align = 0
    phase_center = 1
    phase_descend = 2
    phase_settle = 3
    if not hasattr(env, "_scripted_insert_phase"):
        env._scripted_insert_phase = torch.zeros(n, device=env.device, dtype=torch.int64)
    if not hasattr(env, "_scripted_insert_ready_count"):
        env._scripted_insert_ready_count = torch.zeros(n, device=env.device, dtype=torch.int64)
    if not hasattr(env, "_scripted_insert_settle_count"):
        env._scripted_insert_settle_count = torch.zeros(n, device=env.device, dtype=torch.int64)
    phase = env._scripted_insert_phase
    ready_count = env._scripted_insert_ready_count
    settle_count = env._scripted_insert_settle_count
    if phase.shape[0] != n:
        phase = env._scripted_insert_phase = torch.zeros(n, device=env.device, dtype=torch.int64)
        ready_count = env._scripted_insert_ready_count = torch.zeros(n, device=env.device, dtype=torch.int64)
        settle_count = env._scripted_insert_settle_count = torch.zeros(n, device=env.device, dtype=torch.int64)

    depth = torch.clamp(-delta_port[:, 2], min=0.0)
    rot_ready = (axis > axis_gate) & (twist > twist_gate)
    rot_lost = (axis < 0.75 * axis_gate) | (twist < 0.75 * twist_gate)
    xy_ready = xy < xy_gate
    xy_lost = xy > 1.25 * xy_gate
    depth_ready = depth >= target_depth
    stable_ready = rot_ready & xy_ready & (depth < target_depth)
    ready_count[:] = torch.where(stable_ready, ready_count + 1, torch.zeros_like(ready_count))

    phase[:] = torch.where((phase == phase_align) & rot_ready, phase_center, phase)
    phase[:] = torch.where((phase == phase_center) & rot_ready & xy_ready & (ready_count >= 5), phase_descend, phase)
    phase[:] = torch.where((phase == phase_descend) & depth_ready & rot_ready & xy_ready, phase_settle, phase)
    phase[:] = torch.where((phase >= phase_descend) & xy_lost, phase_center, phase)
    phase[:] = torch.where((phase >= phase_center) & rot_lost, phase_align, phase)
    settle_ok = (phase == phase_settle) & rot_ready & xy_ready & depth_ready
    settle_count[:] = torch.where(settle_ok, settle_count + 1, torch.zeros_like(settle_count))

    delta_cmd_port = torch.zeros((n, 3), device=env.device, dtype=delta_port.dtype)
    xy_norm = torch.clamp(xy, min=1.0e-6)
    xy_cmd_mag = torch.minimum(xy, torch.full_like(xy, xy_step))
    center_mask = (phase == phase_center) | (phase == phase_descend)
    delta_cmd_port[:, 0] = torch.where(center_mask, -delta_port[:, 0] / xy_norm * xy_cmd_mag, 0.0)
    delta_cmd_port[:, 1] = torch.where(center_mask, -delta_port[:, 1] / xy_norm * xy_cmd_mag, 0.0)
    seat_z = -target_depth - delta_port[:, 2]
    descend_mask = (phase == phase_descend) & rot_ready & xy_ready
    retract_mask = (phase != phase_descend) & (delta_port[:, 2] < -target_depth - z_above)
    delta_cmd_port[:, 2] = torch.where(descend_mask, torch.clamp(seat_z, -z_step, 0.0), 0.0)
    delta_cmd_port[:, 2] = torch.where(retract_mask, torch.clamp(seat_z, 0.0, z_step), delta_cmd_port[:, 2])

    delta_cmd_world = quat_rotate(port_q, delta_cmd_port)
    delta_cmd_root = quat_rotate_inverse(robot.data.root_quat_w, delta_cmd_world)
    rot_err_world = quat_rotate(port_q, rot_err_port)
    rot_err_root = quat_rotate_inverse(robot.data.root_quat_w, rot_err_world)
    scale = torch.tensor(action_scale, device=env.device, dtype=delta_port.dtype).unsqueeze(0)
    rot_step_t = torch.tensor(rot_step, device=env.device, dtype=delta_port.dtype).unsqueeze(0)
    hint = torch.zeros((n, 6), device=env.device, dtype=delta_port.dtype)
    hint[:, :3] = delta_cmd_root / scale[:, :3]
    rot_cmd_root = torch.clamp(yaw_gain * rot_err_root, min=-rot_step_t, max=rot_step_t)
    rot_cmd_root = torch.where(
        rot_ready.unsqueeze(-1),
        torch.zeros_like(rot_cmd_root),
        rot_cmd_root,
    )
    hint[:, 3:] = rot_cmd_root / scale[:, 3:]
    return torch.clamp(hint, -1.0, 1.0)


def perception_valid_flag(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Binary flag indicating if SC triangulation succeeded this step."""
    if target_key not in ("sc", "sc2"):
        return torch.ones((env.num_envs, 1), device=env.device, dtype=torch.float32)
    _perceived_port_pose_w(env, target_key)
    valid = getattr(
        env,
        f"_sc_perception_valid_{target_key}",
        torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
    )
    return valid.float().unsqueeze(-1)


def tcp_velocity_port(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """TCP linear/angular velocity expressed in port frame, shape (N, 6)."""
    robot: Articulation = env.scene["robot"]
    ee_idx = robot.data.body_names.index("wrist_3_link")
    lin_w = robot.data.body_lin_vel_w[:, ee_idx, :]
    ang_w = robot.data.body_ang_vel_w[:, ee_idx, :]
    _, port_q = _port_pose_w(env, target_key)
    lin_p = quat_rotate_inverse(port_q, lin_w)
    ang_p = quat_rotate_inverse(port_q, ang_w)
    return torch.cat([lin_p, ang_p], dim=-1)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def tip_to_port_xy_l2(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Negative-signed: XY distance in port frame (L2)."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    return torch.norm(delta_port[:, :2], dim=-1)


def tip_to_port_z(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Diagnostic signed Z error in port frame."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    return delta_port[:, 2]


def tip_to_port_xy_exp(env: "ManagerBasedRLEnv", target_key: str, sigma: float = 0.01) -> torch.Tensor:
    """Gaussian on XY distance — dense signal near target, flat far away."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    d2 = torch.sum(delta_port[:, :2] ** 2, dim=-1)
    return torch.exp(-d2 / (sigma ** 2))


def tip_to_port_xy_inv(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Dense far-range XY attraction reward."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    return 1.0 / (1.0 + xy)


def tip_to_port_z_exp(
    env: "ManagerBasedRLEnv",
    target_key: str,
    sigma_z: float = 0.03,
    xy_gate: float = 0.05,
) -> torch.Tensor:
    """Reward bringing the tip near the port entrance plane (z ~= 0).

    The reward is gated by XY proximity so the policy learns a staged behavior:
    first center over the port, then descend.
    """
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    z = delta_port[:, 2]
    z_reward = torch.exp(-(z ** 2) / (sigma_z ** 2))
    return z_reward * (xy < xy_gate).float()


def tip_to_port_z_progress(
    env: "ManagerBasedRLEnv",
    target_key: str,
    xy_gate: float = 0.25,
    clip: float = 0.03,
) -> torch.Tensor:
    """Reward progress in reducing |z error| once roughly centered in XY."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    abs_z = torch.abs(delta_port[:, 2])

    if not hasattr(env, "_prev_abs_z_err"):
        env._prev_abs_z_err = abs_z.detach().clone()
    prev_abs_z = env._prev_abs_z_err

    progress = torch.clamp(prev_abs_z - abs_z, min=0.0, max=clip)
    gate = (xy < xy_gate).float()
    env._prev_abs_z_err = abs_z.detach().clone()
    return progress * gate


def insertion_depth_raw(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Ungated normalized depth below the port entrance plane."""
    cfg = _TARGETS[target_key]
    full_depth = cfg["depth"]
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    depth = torch.clamp(-delta_port[:, 2], min=0.0, max=full_depth)
    return depth / full_depth


def insertion_depth_progress(
    env: "ManagerBasedRLEnv", target_key: str, clip: float = 0.05
) -> torch.Tensor:
    """Reward progress in moving below the entrance plane."""
    depth = insertion_depth_raw(env, target_key)
    if not hasattr(env, "_prev_insert_depth"):
        env._prev_insert_depth = depth.detach().clone()
    prev_depth = env._prev_insert_depth
    progress = torch.clamp(depth - prev_depth, min=0.0, max=clip)
    env._prev_insert_depth = depth.detach().clone()
    return progress


def centered_insertion_depth_reward(
    env: "ManagerBasedRLEnv",
    target_key: str,
    sigma_xy: float = 0.008,
) -> torch.Tensor:
    """Reward depth only when the tip is centered on the port axis."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    depth = insertion_depth_raw(env, target_key)
    centered = torch.exp(-((xy / sigma_xy) ** 2))
    return depth * centered


def inserted_xy_alignment(
    env: "ManagerBasedRLEnv",
    target_key: str,
    sigma_xy: float = 0.008,
    min_depth_fraction: float = 0.05,
) -> torch.Tensor:
    """Reward XY centering once the tip is below the entrance plane."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    depth = insertion_depth_raw(env, target_key)
    centered = torch.exp(-((xy / sigma_xy) ** 2))
    return centered * (depth > min_depth_fraction).float()


def offcenter_depth_penalty(
    env: "ManagerBasedRLEnv",
    target_key: str,
    free_xy: float = 0.006,
) -> torch.Tensor:
    """Penalty for gaining depth while away from the port centerline."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    depth = insertion_depth_raw(env, target_key)
    return depth * torch.relu(xy - free_xy)


def scripted_action_imitation_reward(
    env: "ManagerBasedRLEnv",
    target_key: str,
    sigma: float = 0.45,
) -> torch.Tensor:
    """Reward matching the scripted bootstrap action in raw action space."""
    action = env.action_manager.action
    hint = scripted_insert_action_hint(env, target_key)
    err = torch.mean((action - hint) ** 2, dim=-1)
    return torch.exp(-err / (sigma ** 2))


def scripted_action_error(
    env: "ManagerBasedRLEnv",
    target_key: str,
) -> torch.Tensor:
    """Diagnostic mean-squared raw action error to the scripted bootstrap action."""
    action = env.action_manager.action
    hint = scripted_insert_action_hint(env, target_key)
    return torch.mean((action - hint) ** 2, dim=-1)


def adaptive_centered_depth_curriculum(
    env: "ManagerBasedRLEnv",
    target_key: str,
    sigma_xy: float = 0.015,
    xy_thresh: float = 0.015,
    xy_ready: float = 0.35,
    ema_alpha: float = 0.02,
) -> torch.Tensor:
    """Batch-adaptive reward: learn XY centering first, then allow depth.

    The scalar depth phase is driven by an EMA of the current batch's XY gate.
    When almost no samples are centered, this term is mostly a centering reward.
    As the policy starts consistently satisfying the XY gate, the term becomes
    a centered-depth reward.
    """
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    depth = insertion_depth_raw(env, target_key)
    centered = torch.exp(-((xy / sigma_xy) ** 2))

    xy_rate = (xy < xy_thresh).float().mean().detach()
    if not hasattr(env, "_adaptive_xy_gate_ema"):
        env._adaptive_xy_gate_ema = torch.zeros((), device=env.device, dtype=xy.dtype)
    env._adaptive_xy_gate_ema = (
        (1.0 - ema_alpha) * env._adaptive_xy_gate_ema + ema_alpha * xy_rate
    )
    depth_phase = torch.clamp(env._adaptive_xy_gate_ema / xy_ready, 0.0, 1.0)
    env._adaptive_depth_phase = depth_phase.detach()

    return centered * ((1.0 - depth_phase) + depth_phase * depth)


def adaptive_xy_gate_ema(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Diagnostic scalar for the adaptive reward scheduler's XY EMA."""
    if not hasattr(env, "_adaptive_xy_gate_ema"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.ones(env.num_envs, device=env.device) * env._adaptive_xy_gate_ema


def adaptive_depth_phase(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Diagnostic scalar: 0 = XY phase, 1 = depth phase."""
    if not hasattr(env, "_adaptive_depth_phase"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.ones(env.num_envs, device=env.device) * env._adaptive_depth_phase


def crossed_port_plane(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Binary diagnostic/reward: tip is below the port entrance plane."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    return (delta_port[:, 2] < 0.0).float()


def insertion_depth_gate(
    env: "ManagerBasedRLEnv", target_key: str, depth_fraction: float = 0.35
) -> torch.Tensor:
    """Diagnostic gate: normalized insertion depth is above threshold."""
    return (insertion_depth_raw(env, target_key) > depth_fraction).float()


def insertion_xy_gate(
    env: "ManagerBasedRLEnv", target_key: str, xy_thresh: float = 0.015
) -> torch.Tensor:
    """Diagnostic gate: tip XY is close enough to the port centerline."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    return (xy < xy_thresh).float()


def insertion_axis_gate(
    env: "ManagerBasedRLEnv", target_key: str, axis_thresh: float = 0.45
) -> torch.Tensor:
    """Diagnostic gate: insertion axis alignment is above threshold."""
    return (plug_port_axis_alignment(env, target_key) > axis_thresh).float()


def insertion_twist_gate(
    env: "ManagerBasedRLEnv", target_key: str, twist_thresh: float = 0.45
) -> torch.Tensor:
    """Diagnostic gate: twist alignment is above threshold."""
    return (plug_port_twist_alignment(env, target_key) > twist_thresh).float()


def insertion_depth_xy_gate(
    env: "ManagerBasedRLEnv",
    target_key: str,
    depth_fraction: float = 0.35,
    xy_thresh: float = 0.015,
) -> torch.Tensor:
    """Diagnostic gate: depth and XY success gates are both satisfied."""
    return insertion_depth_gate(env, target_key, depth_fraction) * insertion_xy_gate(env, target_key, xy_thresh)


def tip_to_port_z_inv(
    env: "ManagerBasedRLEnv", target_key: str, xy_gate: float = 0.40
) -> torch.Tensor:
    """Dense z-alignment reward that does not saturate to exact zero."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    abs_z = torch.abs(delta_port[:, 2])
    z_reward = 1.0 / (1.0 + abs_z)
    return z_reward * (xy < xy_gate).float()


def tip_to_port_descend_reward(
    env: "ManagerBasedRLEnv",
    target_key: str,
    sigma_z: float = 0.08,
    xy_gate: float = 0.40,
) -> torch.Tensor:
    """Reward moving down toward/into the port along local -Z.

    If tip is above the entrance (z > 0), reward grows as z approaches 0.
    If already below (z <= 0), give full score to avoid discouraging insertion.
    """
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    z = delta_port[:, 2]
    above_reward = torch.exp(-(torch.relu(z) ** 2) / (sigma_z ** 2))
    descend_reward = torch.where(z <= 0.0, torch.ones_like(z), above_reward)
    return descend_reward * (xy < xy_gate).float()


def tip_to_port_xy_progress(
    env: "ManagerBasedRLEnv", target_key: str, clip: float = 0.05
) -> torch.Tensor:
    """Reward reduction in XY error from previous step.

    Positive when moving toward the port centerline, negative when drifting
    away. This gives a learnable directional signal before insertion depth
    starts to activate.
    """
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)

    if not hasattr(env, "_prev_xy_err"):
        env._prev_xy_err = xy.detach().clone()
    prev_xy = env._prev_xy_err

    # Reward only forward progress; avoid large negative updates from noisy
    # reset transitions that can dominate early learning.
    progress = torch.clamp(prev_xy - xy, min=0.0, max=clip)
    env._prev_xy_err = xy.detach().clone()
    return progress


def plug_port_axis_dot(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Dot product of plug Z-axis with desired seated Z-axis in world frame."""
    _, _, _, _, tip_q_w, _, _ = _tip_in_port_frame(env, target_key)
    _, desired_tip_q_w = _desired_seated_tip_pose_w(env, target_key)
    device = tip_q_w.device
    N = tip_q_w.shape[0]
    z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, -1)
    plug_z = quat_rotate(tip_q_w, z)
    desired_z = quat_rotate(desired_tip_q_w, z)
    return torch.sum(plug_z * desired_z, dim=-1)


def plug_port_axis_alignment(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Shifted insertion-axis alignment in [0, 1] against the seated pose."""
    dot = plug_port_axis_dot(env, target_key)
    return 0.5 * (dot + 1.0)


def plug_port_twist_alignment(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Alignment of plug twist around insertion axis in [0, 1].

    Axis alignment alone leaves one free DoF: rotation about insertion axis.
    This term resolves that ambiguity by aligning projected X-axes in the plane
    orthogonal to the insertion direction (-port Z).
    """
    _, _, _, _, tip_q_w, _, _ = _tip_in_port_frame(env, target_key)
    _, desired_tip_q_w = _desired_seated_tip_pose_w(env, target_key)
    device = tip_q_w.device
    n_env = tip_q_w.shape[0]
    z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n_env, -1)
    x = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n_env, -1)

    insertion_axis = quat_rotate(desired_tip_q_w, z)
    insertion_axis = insertion_axis / torch.clamp(
        torch.norm(insertion_axis, dim=-1, keepdim=True), min=1e-6
    )
    plug_x = quat_rotate(tip_q_w, x)
    desired_x = quat_rotate(desired_tip_q_w, x)

    plug_x_proj = plug_x - torch.sum(plug_x * insertion_axis, dim=-1, keepdim=True) * insertion_axis
    desired_x_proj = desired_x - torch.sum(desired_x * insertion_axis, dim=-1, keepdim=True) * insertion_axis
    plug_x_proj = plug_x_proj / torch.clamp(torch.norm(plug_x_proj, dim=-1, keepdim=True), min=1e-6)
    desired_x_proj = desired_x_proj / torch.clamp(torch.norm(desired_x_proj, dim=-1, keepdim=True), min=1e-6)

    dot = torch.sum(plug_x_proj * desired_x_proj, dim=-1)
    return 0.5 * (dot + 1.0)


def plug_port_twist_progress(
    env: "ManagerBasedRLEnv", target_key: str, clip: float = 0.05
) -> torch.Tensor:
    """Reward progress in reducing twist misalignment around insertion axis."""
    align = plug_port_twist_alignment(env, target_key)
    err = 1.0 - align
    if not hasattr(env, "_prev_twist_err"):
        env._prev_twist_err = err.detach().clone()
    prev_err = env._prev_twist_err
    progress = torch.clamp(prev_err - err, min=0.0, max=clip)
    env._prev_twist_err = err.detach().clone()
    return progress


def tip_rotation_l2(env: "ManagerBasedRLEnv", target_key: str) -> torch.Tensor:
    """Axis-angle magnitude between current tip pose and the port entrance frame."""
    _, _, rot_err_port, *_ = _tip_in_port_frame(env, target_key)
    return torch.norm(rot_err_port, dim=-1)


def tip_rotation_exp(env: "ManagerBasedRLEnv", target_key: str, sigma: float = 0.20) -> torch.Tensor:
    """Dense rotation reward around the exact tip frame."""
    rot_err = tip_rotation_l2(env, target_key)
    return torch.exp(-(rot_err ** 2) / (sigma ** 2))


def tip_rotation_progress(
    env: "ManagerBasedRLEnv", target_key: str, clip: float = 0.08
) -> torch.Tensor:
    """Reward progress in reducing full 3D rotational error."""
    rot_err = tip_rotation_l2(env, target_key)
    if not hasattr(env, "_prev_rot_err"):
        env._prev_rot_err = rot_err.detach().clone()
    prev_rot = env._prev_rot_err
    progress = torch.clamp(prev_rot - rot_err, min=0.0, max=clip)
    env._prev_rot_err = rot_err.detach().clone()
    return progress


def insertion_depth_reward(
    env: "ManagerBasedRLEnv",
    target_key: str,
    xy_gate: float = 0.012,
    axis_gate: float = 0.55,
    twist_gate: float = 0.55,
) -> torch.Tensor:
    """Dense insertion depth reward gated by insertion-axis and twist alignment."""
    cfg = _TARGETS[target_key]
    full_depth = cfg["depth"]
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    depth = torch.clamp(-delta_port[:, 2], min=0.0, max=full_depth)
    xy_gate_soft = torch.exp(-((xy / xy_gate) ** 2))
    axis = torch.clamp((plug_port_axis_alignment(env, target_key) - axis_gate) / (1.0 - axis_gate), 0.0, 1.0)
    twist = torch.clamp((plug_port_twist_alignment(env, target_key) - twist_gate) / (1.0 - twist_gate), 0.0, 1.0)
    align_gate = xy_gate_soft * axis * twist
    return (depth / full_depth) * align_gate


def insertion_success_bonus(
    env: "ManagerBasedRLEnv",
    target_key: str,
    xy_thresh: float = 0.005,
    axis_thresh: float = 0.75,
    twist_thresh: float = 0.70,
    depth_fraction: float = 0.75,
) -> torch.Tensor:
    """One-shot +1 when insertion is near-complete."""
    cfg = _TARGETS[target_key]
    full_depth = cfg["depth"]
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    depth_below = -delta_port[:, 2]
    axis = plug_port_axis_alignment(env, target_key)
    twist = plug_port_twist_alignment(env, target_key)
    return (
        (depth_below > depth_fraction * full_depth)
        & (xy < xy_thresh)
        & (axis > axis_thresh)
        & (twist > twist_thresh)
    ).float()


def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Small positive per-step reward to bias against immediate failures."""
    return torch.ones(env.num_envs, device=env.device)


def wrist_force_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Force magnitude on the wrist (N-scale, not squared).

    We intentionally avoid squaring here because large transient contacts make
    the squared penalty dominate early exploration and drown out insertion
    shaping terms.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    # body_incoming_wrench is 6D (force+torque) per body; take force part
    wrench = asset.root_physx_view.get_link_incoming_joint_force()[:, body_ids, :]
    # shape (N, B, 6). Force = first 3, torque = last 3
    force = wrench[..., :3].reshape(env.num_envs, -1)
    return torch.norm(force, dim=-1)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

def insertion_success(
    env: "ManagerBasedRLEnv",
    target_key: str,
    xy_thresh: float = 0.005,
    axis_thresh: float = 0.75,
    twist_thresh: float = 0.70,
    depth_fraction: float = 0.75,
) -> torch.Tensor:
    """Terminate with success when plug is seated with correct insertion-axis alignment."""
    cfg = _TARGETS[target_key]
    full_depth = cfg["depth"]
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    depth_below = -delta_port[:, 2]
    axis = plug_port_axis_alignment(env, target_key)
    twist = plug_port_twist_alignment(env, target_key)
    return (
        (depth_below > depth_fraction * full_depth)
        & (xy < xy_thresh)
        & (axis > axis_thresh)
        & (twist > twist_thresh)
    )


def drift_failure(env: "ManagerBasedRLEnv", target_key: str, xy_thresh: float = 0.10) -> torch.Tensor:
    """Terminate on failure when plug wanders far in XY (port frame)."""
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    return xy > xy_thresh


def offcenter_insert_failure(
    env: "ManagerBasedRLEnv",
    target_key: str,
    xy_thresh: float = 0.025,
    depth_fraction: float = 0.10,
    duration_steps: int = 3,
) -> torch.Tensor:
    """Fail fast when the plug goes below the entrance while clearly off-center."""
    cfg = _TARGETS[target_key]
    delta_port, *_ = _tip_in_port_frame(env, target_key)
    xy = torch.norm(delta_port[:, :2], dim=-1)
    depth = torch.clamp(-delta_port[:, 2], min=0.0) / cfg["depth"]
    bad = (depth > depth_fraction) & (xy > xy_thresh)

    if not hasattr(env, "_offcenter_insert_count"):
        env._offcenter_insert_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    counter = env._offcenter_insert_count
    counter[bad] += 1
    counter[~bad] = 0
    env._offcenter_insert_count = counter
    return counter > duration_steps


# Force failure needs stateful tracking (duration). We keep a per-env counter
# attached to the env object, initialized lazily.
def force_failure(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    force_thresh: float = 20.0,
    duration_steps: int = 30,
) -> torch.Tensor:
    """Fail if force exceeds threshold for `duration_steps` consecutive steps."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    wrench = asset.root_physx_view.get_link_incoming_joint_force()[:, body_ids, :]
    force = wrench[..., :3].reshape(env.num_envs, -1)
    force_mag = torch.norm(force, dim=-1)

    # Lazy counter init
    if not hasattr(env, "_force_over_count"):
        env._force_over_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    counter = env._force_over_count
    over = force_mag > force_thresh
    counter[over] += 1
    counter[~over] = 0
    env._force_over_count = counter
    return counter > duration_steps


def reset_to_preinsertion_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    target_key: str,
    curriculum_steps: int = 150000,
    joint_noise_start: float = 0.008,
    joint_noise_end: float = 0.003,
    xy_error_start: float = 0.008,
    xy_error_end: float = 0.0025,
    z_above_start: float = 0.020,
    z_above_end: float = 0.006,
    z_below_start: float = 0.0,
    z_below_end: float = 0.0,
    roll_pitch_start: float = 0.18,
    roll_pitch_end: float = 0.05,
    yaw_start: float = 0.30,
    yaw_end: float = 0.08,
    ik_gain: float = 0.45,
    ik_delta_limit: float = 0.05,
    ik_iters: int = 1,
) -> None:
    """Reset directly into the near-port handoff manifold seen in Gazebo logs."""
    robot: Articulation = env.scene["robot"]

    if not hasattr(env, "_insert_arm_joint_ids"):
        arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        arm_joint_ids, _ = robot.find_joints(arm_joint_names)
        env._insert_arm_joint_ids = arm_joint_ids
    arm_ids = env._insert_arm_joint_ids

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    base = torch.tensor(
        _PREINSERT_ARM_JOINTS[target_key], device=env.device, dtype=joint_pos.dtype
    ).unsqueeze(0)
    base = base.expand(len(env_ids), -1)

    step_count = float(getattr(env, "common_step_counter", 0))
    frac = min(1.0, step_count / float(curriculum_steps))
    noise_mag = joint_noise_start + frac * (joint_noise_end - joint_noise_start)
    noise = (torch.rand_like(base) * 2.0 - 1.0) * noise_mag

    joint_pos[:, arm_ids] = base + noise

    # Write initial warm-start before IK refinement.
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)

    # Sample a realistic near-port tip pose and refine the arm with a damped
    # TCP-offset IK loop. The reward TCP must match the controller body_offset;
    # solving the bare wrist pose leaves large tip errors when rotation differs.
    port_p, port_q = _port_pose_w(env, target_key)
    port_p = port_p[env_ids]
    port_q = port_q[env_ids]

    eef_idx = robot.data.body_names.index("wrist_3_link")
    jac_idx = eef_idx - 1  # fixed-base articulation convention

    num_envs = len(env_ids)
    xy_band = xy_error_start + frac * (xy_error_end - xy_error_start)
    z_band = z_above_start + frac * (z_above_end - z_above_start)
    z_below_band = z_below_start + frac * (z_below_end - z_below_start)
    roll_pitch_band = roll_pitch_start + frac * (roll_pitch_end - roll_pitch_start)
    yaw_band = yaw_start + frac * (yaw_end - yaw_start)

    desired_tip_local = torch.zeros((num_envs, 3), device=env.device, dtype=joint_pos.dtype)
    desired_tip_local[:, 0] = (torch.rand(num_envs, device=env.device, dtype=joint_pos.dtype) * 2.0 - 1.0) * xy_band
    desired_tip_local[:, 1] = (torch.rand(num_envs, device=env.device, dtype=joint_pos.dtype) * 2.0 - 1.0) * xy_band
    z_span = z_band + z_below_band
    desired_tip_local[:, 2] = torch.rand(num_envs, device=env.device, dtype=joint_pos.dtype) * z_span - z_below_band

    roll = (torch.rand(num_envs, device=env.device, dtype=joint_pos.dtype) * 2.0 - 1.0) * roll_pitch_band
    pitch = (torch.rand(num_envs, device=env.device, dtype=joint_pos.dtype) * 2.0 - 1.0) * roll_pitch_band
    yaw = (torch.rand(num_envs, device=env.device, dtype=joint_pos.dtype) * 2.0 - 1.0) * yaw_band
    seated_tip_local_quat = _seated_tip_quat_port(num_envs, env.device, joint_pos.dtype)
    desired_tip_local_quat = quat_mul(
        seated_tip_local_quat,
        quat_from_euler_xyz(roll, pitch, yaw),
    )
    desired_tip_pos_w, desired_tip_quat_w = combine_frame_transforms(
        port_p, port_q, desired_tip_local, desired_tip_local_quat
    )
    env._last_reset_desired_tip_local = desired_tip_local.detach().clone()
    env._last_reset_desired_tip_pos_w = desired_tip_pos_w.detach().clone()
    env._last_reset_desired_tip_quat_w = desired_tip_quat_w.detach().clone()

    tip_cfg = _TIP_IN_WRIST[_TARGETS[target_key]["port_type"]]
    tip_in_wrist_pos = torch.tensor(tip_cfg["pos"], device=env.device, dtype=joint_pos.dtype).expand(num_envs, -1)
    tip_in_wrist_quat = torch.tensor(
        tip_cfg["quat_wxyz"], device=env.device, dtype=joint_pos.dtype
    ).expand(num_envs, -1)
    wrist_in_tip_pos, wrist_in_tip_quat = subtract_frame_transforms(tip_in_wrist_pos, tip_in_wrist_quat)
    desired_wrist_pos_w, desired_wrist_quat_w = combine_frame_transforms(
        desired_tip_pos_w, desired_tip_quat_w, wrist_in_tip_pos, wrist_in_tip_quat
    )
    env._last_reset_desired_wrist_pos_w = desired_wrist_pos_w.detach().clone()
    env._last_reset_desired_wrist_quat_w = desired_wrist_quat_w.detach().clone()

    if getattr(env.cfg, "use_kinematic_tip", False):
        if not hasattr(env, "_kinematic_tip_pos_w"):
            env._kinematic_tip_pos_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=joint_pos.dtype)
            env._kinematic_tip_quat_w = torch.zeros((env.num_envs, 4), device=env.device, dtype=joint_pos.dtype)
            env._kinematic_tip_quat_w[:, 0] = 1.0
        env._kinematic_tip_pos_w[env_ids] = desired_tip_pos_w
        env._kinematic_tip_quat_w[env_ids] = desired_tip_quat_w

        # Keep the physical arm in a benign warm-start pose. The terminal
        # insertion MDP is expressed at the TCP handoff, so rewards and
        # observations use the kinematic tip above rather than forcing a
        # synthetic whole-arm/cable state to settle.
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)

        if hasattr(env, "_force_over_count"):
            env._force_over_count[env_ids] = 0
        if hasattr(env, "_offcenter_insert_count"):
            env._offcenter_insert_count[env_ids] = 0
        if hasattr(env, "_prev_xy_err"):
            env._prev_xy_err[env_ids] = 0.0
        if hasattr(env, "_prev_abs_z_err"):
            env._prev_abs_z_err[env_ids] = 0.0
        if hasattr(env, "_prev_twist_err"):
            env._prev_twist_err[env_ids] = 1.0
        if hasattr(env, "_prev_rot_err"):
            env._prev_rot_err[env_ids] = 0.0
        if hasattr(env, "_prev_insert_depth"):
            env._prev_insert_depth[env_ids] = 0.0
        if hasattr(env, "_scripted_insert_phase"):
            env._scripted_insert_phase[env_ids] = 0
        if hasattr(env, "_scripted_insert_ready_count"):
            env._scripted_insert_ready_count[env_ids] = 0
        if hasattr(env, "_scripted_insert_settle_count"):
            env._scripted_insert_settle_count[env_ids] = 0
        return

    action_term = None
    if hasattr(env, "action_manager"):
        try:
            action_term = env.action_manager.get_term("arm_action")
        except KeyError:
            action_term = None

    if action_term is not None and hasattr(action_term, "_compute_frame_pose"):
        root_pos_w = robot.data.root_pos_w[env_ids]
        root_quat_w = robot.data.root_quat_w[env_ids]
        desired_tip_pos_b, desired_tip_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, desired_tip_pos_w, desired_tip_quat_w
        )
        ee_pos_b, ee_quat_b = action_term._compute_frame_pose()
        env._last_reset_ik_initial_pos_err = (desired_tip_pos_b - ee_pos_b[env_ids]).detach().clone()
        for _ in range(max(1, int(ik_iters))):
            ee_pos_b, ee_quat_b = action_term._compute_frame_pose()
            jac = action_term._compute_frame_jacobian()
            ee_pos_b = ee_pos_b[env_ids]
            ee_quat_b = ee_quat_b[env_ids]
            jac_frame = jac[env_ids, :, :]
            pos_err = desired_tip_pos_b - ee_pos_b
            _, rot_err = compute_pose_error(
                ee_pos_b,
                ee_quat_b,
                desired_tip_pos_b,
                desired_tip_quat_b,
                rot_error_type="axis_angle",
            )
            pose_err = torch.cat([pos_err, 0.35 * rot_err], dim=-1)
            jac_pinv = torch.linalg.pinv(jac_frame)
            delta = torch.bmm(jac_pinv, pose_err.unsqueeze(-1)).squeeze(-1)
            delta = torch.clamp(ik_gain * delta, min=-ik_delta_limit, max=ik_delta_limit)
            joint_pos[:, arm_ids] = robot.data.joint_pos[env_ids][:, arm_ids] + delta
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
        ee_pos_b, ee_quat_b = action_term._compute_frame_pose()
        env._last_reset_ik_final_pos_err = (desired_tip_pos_b - ee_pos_b[env_ids]).detach().clone()
        _, final_rot_err = compute_pose_error(
            ee_pos_b[env_ids],
            ee_quat_b[env_ids],
            desired_tip_pos_b,
            desired_tip_quat_b,
            rot_error_type="axis_angle",
        )
        env._last_reset_ik_final_rot_err = final_rot_err.detach().clone()
        env._last_reset_ik_joint_ids = list(arm_ids)
    else:
        eef_pos = robot.data.body_pos_w[env_ids, eef_idx, :]
        eef_quat = robot.data.body_quat_w[env_ids, eef_idx, :]
        tip_pos_w = eef_pos + quat_rotate(eef_quat, tip_in_wrist_pos)
        env._last_reset_ik_initial_pos_err = (desired_tip_pos_w - tip_pos_w).detach().clone()
        for _ in range(max(1, int(ik_iters))):
            eef_pos = robot.data.body_pos_w[env_ids, eef_idx, :]
            eef_quat = robot.data.body_quat_w[env_ids, eef_idx, :]
            tip_offset_w = quat_rotate(eef_quat, tip_in_wrist_pos)
            tip_pos_w = eef_pos + tip_offset_w
            tip_quat_w = quat_mul(eef_quat, tip_in_wrist_quat)
            pos_err = desired_tip_pos_w - tip_pos_w
            wrist_jac = robot.root_physx_view.get_jacobians()[env_ids, jac_idx, :, :][:, :, arm_ids]
            jac_ang_t = wrist_jac[:, 3:, :].transpose(1, 2)
            tip_linear_from_rot = torch.cross(
                jac_ang_t,
                tip_offset_w.unsqueeze(1).expand_as(jac_ang_t),
                dim=-1,
            ).transpose(1, 2)
            jac = wrist_jac[:, :3, :] + tip_linear_from_rot
            jac_pinv = torch.linalg.pinv(jac)
            delta = torch.bmm(jac_pinv, pos_err.unsqueeze(-1)).squeeze(-1)
            delta = torch.clamp(ik_gain * delta, min=-ik_delta_limit, max=ik_delta_limit)
            joint_pos[:, arm_ids] = joint_pos[:, arm_ids] + delta
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
        eef_pos = robot.data.body_pos_w[env_ids, eef_idx, :]
        eef_quat = robot.data.body_quat_w[env_ids, eef_idx, :]
        tip_pos_w = eef_pos + quat_rotate(eef_quat, tip_in_wrist_pos)
        env._last_reset_ik_final_pos_err = (desired_tip_pos_w - tip_pos_w).detach().clone()
        env._last_reset_ik_joint_ids = list(arm_ids)

    # Reset force-failure counters for the envs being reset.
    if hasattr(env, "_force_over_count"):
        env._force_over_count[env_ids] = 0
    if hasattr(env, "_offcenter_insert_count"):
        env._offcenter_insert_count[env_ids] = 0
    if hasattr(env, "_prev_xy_err"):
        env._prev_xy_err[env_ids] = 0.0
    if hasattr(env, "_prev_abs_z_err"):
        env._prev_abs_z_err[env_ids] = 0.0
    if hasattr(env, "_prev_twist_err"):
        env._prev_twist_err[env_ids] = 1.0
    if hasattr(env, "_prev_rot_err"):
        env._prev_rot_err[env_ids] = 0.0
    if hasattr(env, "_prev_insert_depth"):
        env._prev_insert_depth[env_ids] = 0.0
    if hasattr(env, "_scripted_insert_phase"):
        env._scripted_insert_phase[env_ids] = 0
    if hasattr(env, "_scripted_insert_ready_count"):
        env._scripted_insert_ready_count[env_ids] = 0
    if hasattr(env, "_scripted_insert_settle_count"):
        env._scripted_insert_settle_count[env_ids] = 0
