"""Gymnasium env wrapping the REAL MuJoCo scene (`aic_utils/aic_mujoco/mjcf/scene.xml`).

Sim-side backend for last-inch insertion RL on the Gazebo->MuJoCo pipeline scene
(UR5e + Robotiq Hand-E + welded LC/SFP plug + elastic cable + task board + the
**real receptacle ports**: NIC card SFP cage `nic_card_mount_2` + `sc_port_0/1`).

This replaces the procedural box env (`RL/env.py`) with the actual robot/scene.

Design (v0.5, 2026-07-03 — port-frame last inch):
  * INSERT TARGET = the exported SFP entrance frame (`sfp_port_1_link_entrance`
    by default), not the NIC-card mount origin. The seated plug TIP is defined in
    that port frame, with a configurable full-depth offset along the cage inward
    axis.
  * LAST-INCH REVERSE CURRICULUM (relative to the port, NOT to robot home): level 0
    starts the plug inserted at the port; as level -> 1 it retracts up to
    `last_inch_m` (default 4 cm) along the insertion axis (+ lateral/yaw jitter).
    So training stays focused on the last inch even though the port is ~21 cm from
    the cable's home spawn (the robot transports it there in the real task).
  * IMAGE-SAC reward from `RL/reward.py:compute_reward` (image L1 + force + potential
    depth + xy + action smoothness + terminal). The image *observation* is 256x256x9
    (3 wrist cams); the image *distance reward* is rendered at native camera res
    (1152x1024, center cam) for last-inch pixel accuracy.
  * SUCCESS = plug tip near the seated pose, depth at the bottom of the last-inch
    envelope, and seating contact force > `success_force_n` (contact force = FT
    minus the ~10 N gripped-plug baseline, captured at reset).
  * ACTION = incremental 6-D UR5e joint residual on a bounded, gravity-
    compensated PD target.

Two scene subtleties handled here:
  1. Stable reset despite the tool<->plug weld: IK the arm so gripper_tcp is at the
     sampled pose, then rigidly move the straight cable so the welded plug starts at
     `tool . relpose` (zero weld violation) -> no QACC blow-up.
  2. Home MUST be the real AIC config (else the wrist buries in the enclosure and FT
     reads ~20 kN); with it FT reads the true ~10 N gripped-plug load.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces  # type: ignore

import mujoco

os.environ.setdefault("MUJOCO_GL", os.environ.get("AIC_MUJOCO_GL", "egl"))

try:
    from .reward import RewardConfig, TerminationConfig, compute_reward
except ImportError:  # run as a script (python RL/scene_env.py)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from RL.reward import RewardConfig, TerminationConfig, compute_reward

_DEFAULT_SCENE = str(
    Path(__file__).resolve().parents[1]
    / "aic_utils" / "aic_mujoco" / "mjcf" / "scene.xml"
)

ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
# Real AIC home (aic_engine/config/base_config.yaml home_joint_positions).
AIC_HOME = (-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110)
CAMERAS = ("center_camera", "left_camera", "right_camera")


@dataclass
class SceneEnvConfig:
    scene_path: str = _DEFAULT_SCENE
    home_qpos: tuple = AIC_HOME
    settle_steps: int = 120
    control_substeps: int = 5
    max_episode_steps: int = 200
    # base PD controller (torque motors) + residual action scaling
    kp: float = 200.0
    kd: float = 25.0
    action_joint_scale: float = 0.01   # rad per policy step at |action|=1
    action_joint_limit: float = 0.35   # rad envelope around the reset arm target
    gripper_ctrl: float = 0.0
    # --- insertion target: the REAL fixed SFP receptacle on the board ---
    insert_target_body: str = "sfp_port_1_link_entrance"
    insert_target_bodies: tuple = ()    # optional future randomisation hook
    randomize_target_body: bool = False
    plug_tip_body: str = "sfp_tip_link"
    plug_axis_tail_body: str = "sfp_module_link"
    # Keep the TCP offset reachable; the reset IK below aligns the actual plug axis.
    align_plug_axis_to_port: bool = False
    insert_goal_offset: tuple = (0.0, 0.0, 0.0)   # world offset for fine-tuning
    insertion_axis_world: Optional[tuple] = None   # default: target body local +Z
    seated_depth_m: float = 0.046       # full-depth tip target inside the SFP entrance frame
    # --- last-inch reverse curriculum (relative to the port) ---
    last_inch_m: float = 0.04            # curriculum spans the last 4 cm of insertion
    curriculum_level: float = 0.0        # 0 = seated, 1 = retracted last_inch_m
    jitter_xy_m: float = 0.003           # lateral jitter at level 1
    jitter_yaw_rad: float = 0.12         # yaw about insertion axis at level 1
    jitter_tilt_rad: float = 0.04        # small port-frame x/y tilt at level 1
    reset_max_attempts: int = 12
    reset_contact_abort_n: float = 35.0
    reset_max_plug_port_penetration_m: float = 0.004
    reset_ik_tip_tol_m: float = 0.006
    ik_iters: int = 200
    ik_tol: float = 1e-4
    ik_damping: float = 0.05
    ik_step_max: float = 0.15
    ik_tip_iters: int = 120
    ik_fd_eps: float = 1e-4
    ik_tip_damping: float = 1e-4
    ik_tip_step_max: float = 0.12
    ik_axis_weight_m: float = 0.06
    ik_axis_tol_rad: float = 0.06
    # --- observation ---
    include_images: bool = True
    cameras: tuple = CAMERAS
    image_h: int = 256
    image_w: int = 256
    # --- reward-distance image (native res, center cam) ---
    reward_image_res: int = 0            # 0 = native cam res (1152x1024); else HxW square
    # --- reward / termination ---
    reward: RewardConfig = field(default_factory=RewardConfig)
    term: TerminationConfig = field(default_factory=TerminationConfig)
    # --- success ---
    success_pos_tol_m: float = 0.006
    success_axial_tol_m: float = 0.006
    success_lateral_tol_m: float = 0.008
    success_depth_norm: float = 0.97
    success_axis_tol_rad: float = 0.05
    success_force_n: float = 2.0
    success_max_plug_port_penetration_excess_m: float = 0.001
    bad_collision_axis_rad: float = 0.20
    bad_collision_depth_gate: float = 0.20
    bad_collision_penetration_excess_m: float = 0.003
    force_abort_n: float = 60.0          # hard abort on |contact force| (safety)


class SceneInsertEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: SceneEnvConfig = SceneEnvConfig()):
        super().__init__()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.scene_path)
        self.data = mujoco.MjData(self.model)

        jid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        self._arm_qadr = np.array([self.model.jnt_qposadr[jid(j)] for j in ARM_JOINTS])
        self._arm_vadr = np.array([self.model.jnt_dofadr[jid(j)] for j in ARM_JOINTS])
        self._tool_id = bid("ati/tool_link")
        self._plug_id = bid("lc_plug_link")
        self._plug_tip_id = bid(cfg.plug_tip_body)
        self._plug_axis_tail_id = bid(cfg.plug_axis_tail_body)
        if self._plug_axis_tail_id < 0:
            self._plug_axis_tail_id = self._plug_id
        self._cend_id = bid("cable_end_0")
        self._cfree_adr = self.model.jnt_qposadr[jid("cable_end_0_free")]
        self._tcp_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
        target_names = tuple(cfg.insert_target_bodies) or (cfg.insert_target_body,)
        self._target_names = target_names
        self._target_bids = np.array([bid(n) for n in target_names], dtype=np.int32)
        missing = [n for n, i in zip(target_names, self._target_bids) if i < 0]
        if missing:
            raise ValueError(f"insert target body/bodies not in scene: {missing!r}")
        self._target_bid = int(self._target_bids[0])
        self._target_name = target_names[0]
        self._weld_id = [i for i in range(self.model.neq)
                         if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD][0]
        ed = self.model.eq_data[self._weld_id]
        self._rp_pos, self._rp_quat = ed[3:6].copy(), ed[6:10].copy()
        self._ft_force_adr = self._sensor_adr("AtiForceTorqueSensor_force")
        self._ft_torque_adr = self._sensor_adr("AtiForceTorqueSensor_torque")

        self._home = np.asarray(cfg.home_qpos, dtype=np.float64)
        self._curriculum_level = float(cfg.curriculum_level)
        self._level_file: Optional[Path] = None
        self._step_count = 0
        self._last_action = np.zeros(6, np.float32)
        self._arm_target = self._home.copy()
        self._prev_depth_norm = 0.0

        # obs renderer (3 wrist cams) + reward renderer (center cam, native res)
        self._renderer, self._cams = None, []
        self._reward_renderer = None
        if cfg.include_images:
            self._renderer = mujoco.Renderer(self.model, height=cfg.image_h, width=cfg.image_w)
            self._cams = [c for c in cfg.cameras
                          if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c) >= 0]
            cam0 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "center_camera")
            native = self.model.cam_resolution[cam0] if cam0 >= 0 else (1152, 1024)
            rw = int(cfg.reward_image_res) or int(native[0])
            rh = int(cfg.reward_image_res) or int(native[1])
            self._reward_renderer = mujoco.Renderer(self.model, height=rh, width=rw)

        # --- home & goal poses (FK at home, welded plug settled) ---
        self._rigid_home(self._home)
        for _ in range(cfg.settle_steps):      # settle so FT baseline is steady
            self.data.ctrl[:6] = self._base_torque(self._home)
            self.data.ctrl[6] = cfg.gripper_ctrl
            mujoco.mj_step(self.model, self.data)
        self._home_tcp = self.data.site_xpos[self._tcp_sid].copy()
        self._home_quat = self._site_quat()
        plug_tip_home = self.data.xpos[self._plug_tip_id].copy()
        self._tip_rel_pos_tcp = self._qrot(self._qinv(self._home_quat),
                                           plug_tip_home - self._home_tcp)
        self._plug_tip_to_tcp = self._home_tcp - plug_tip_home  # legacy/debug
        self._goal_quat = self._home_quat.copy()
        self._configure_port_frame(self._target_bid)
        self._goal_quat = self._aligned_goal_quat()
        self._configure_port_frame(self._target_bid)
        # FT baseline (gripped-plug weight in free space) = FT at home, no port contact
        self._ft_baseline = self._raw_ft()[:3].copy()

        # spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        obs_spaces = {
            "arm_qpos": spaces.Box(-np.pi, np.pi, (6,), np.float32),
            "arm_qvel": spaces.Box(-50.0, 50.0, (6,), np.float32),
            "tcp_pose": spaces.Box(-3.0, 3.0, (7,), np.float32),
            "ft": spaces.Box(-500.0, 500.0, (6,), np.float32),
            "last_action": spaces.Box(-1.0, 1.0, (6,), np.float32),
        }
        if self._renderer is not None and self._cams:
            obs_spaces["image"] = spaces.Box(
                0, 255, (cfg.image_h, cfg.image_w, 3 * len(self._cams)), np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        # goal images (seated pose): reward-res (center cam) + obs (3 cams)
        self._goal_image_reward = None
        self._goal_image_obs = None
        goal_reset_done = False
        if self._renderer is not None and self._cams:
            self._reset_to_level(0.0, jitter=False)
            goal_reset_done = True
            self._goal_image_reward = self._render_reward_image().astype(np.float32)
            self._goal_image_obs = self._render_obs_image().astype(np.float32)
        else:
            self._goal_image_reward = np.zeros((1, 1, 3), np.float32)
            self._goal_image_obs = np.zeros((1, 1, 3), np.float32)
        if not goal_reset_done:
            self._reset_to_level(0.0, jitter=False)
        goal_diag = self._geometry_diag()
        self._goal_plug_port_penetration_m = float(goal_diag["plug_port_penetration_m"])
        self._goal_axis_error_rad = float(goal_diag["plug_axis_error_rad"])

    # ------------------------------------------------------------------ #
    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid]) if sid >= 0 else None

    def _raw_ft(self):
        ft = np.zeros(6)
        if self._ft_force_adr is not None:
            ft[:3] = self.data.sensordata[self._ft_force_adr:self._ft_force_adr + 3]
        if self._ft_torque_adr is not None:
            ft[3:] = self.data.sensordata[self._ft_torque_adr:self._ft_torque_adr + 3]
        return ft

    def _site_quat(self):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self._tcp_sid].reshape(9))
        return q

    @staticmethod
    def _qmul(a, b):
        r = np.zeros(4); mujoco.mju_mulQuat(r, a, b); return r

    @staticmethod
    def _qinv(a):
        r = np.zeros(4); mujoco.mju_negQuat(r, a); return r

    @staticmethod
    def _qrot(q, v):
        r = np.zeros(3); mujoco.mju_rotVecQuat(r, v, q); return r

    @staticmethod
    def _axis_angle(axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(axis)
        if n < 1e-12 or abs(angle) < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = axis / n
        s = np.sin(angle / 2.0)
        return np.array([np.cos(angle / 2.0), *(axis * s)], dtype=np.float64)

    @staticmethod
    def _unit(v, fallback):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.asarray(fallback, dtype=np.float64)
        return v / n

    @classmethod
    def _quat_between(cls, a, b):
        """Quaternion rotating unit-ish vector `a` onto unit-ish vector `b`."""
        a = cls._unit(a, (1.0, 0.0, 0.0))
        b = cls._unit(b, (1.0, 0.0, 0.0))
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if dot > 1.0 - 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if dot < -1.0 + 1e-10:
            axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if np.linalg.norm(axis) < 1e-8:
                axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            return cls._axis_angle(axis, np.pi)
        axis = np.cross(a, b)
        q = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float64)
        return q / np.linalg.norm(q)

    def _plug_axis(self):
        """World direction from the SFP module body toward the insertion tip."""
        tail = self.data.xpos[self._plug_axis_tail_id]
        tip = self.data.xpos[self._plug_tip_id]
        return self._unit(tip - tail, self._insert_axis)

    def _aligned_goal_quat(self):
        if not self.cfg.align_plug_axis_to_port:
            return self._home_quat.copy()
        home_axis = self._plug_axis()
        q_align = self._quat_between(home_axis, self._insert_axis)
        return self._qmul(q_align, self._home_quat)

    # ------------------------------------------------------------------ #
    def _configure_port_frame(self, target_bid: int):
        self._target_bid = int(target_bid)
        self._target_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self._target_bid) or str(target_bid)
        xmat = self.data.xmat[self._target_bid].reshape(3, 3)
        if self.cfg.insertion_axis_world is None:
            insert_axis = xmat[:, 2]
        else:
            insert_axis = np.asarray(self.cfg.insertion_axis_world, dtype=np.float64)
        self._insert_axis = self._unit(insert_axis, (0.0, 0.0, -1.0))
        self._retract_dir = -self._insert_axis
        lat_x = xmat[:, 0] - np.dot(xmat[:, 0], self._insert_axis) * self._insert_axis
        self._lat_x = self._unit(lat_x, (1.0, 0.0, 0.0))
        self._lat_y = self._unit(np.cross(self._insert_axis, self._lat_x), (0.0, 1.0, 0.0))
        self._port_pos = (self.data.xpos[self._target_bid].copy()
                          + np.asarray(self.cfg.insert_goal_offset, dtype=np.float64))
        self._inserted_tip = self._port_pos + self.cfg.seated_depth_m * self._insert_axis
        self._goal_tcp = self._tcp_for_tip(self._inserted_tip, self._goal_quat)

    def _tcp_for_tip(self, tip_pos, tcp_quat):
        return np.asarray(tip_pos, dtype=np.float64) - self._qrot(tcp_quat, self._tip_rel_pos_tcp)

    def _select_target_body(self):
        if self.cfg.randomize_target_body and len(self._target_bids) > 1:
            bid = int(self.np_random.choice(self._target_bids))
        else:
            bid = int(self._target_bids[0])
        self._configure_port_frame(bid)

    def _rigid_home(self, home):
        """Arm to `home`; rigidly move straight cable so welded plug starts at tool.relpose."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qadr] = home
        mujoco.mj_forward(self.model, self.data)
        xpt, xqt = self.data.xpos[self._tool_id].copy(), self.data.xquat[self._tool_id].copy()
        des_q = self._qmul(xqt, self._rp_quat)
        des_p = xpt + self._qrot(xqt, self._rp_pos)
        xpp, xqp = self.data.xpos[self._plug_id].copy(), self.data.xquat[self._plug_id].copy()
        xpc, xqc = self.data.xpos[self._cend_id].copy(), self.data.xquat[self._cend_id].copy()
        Tq = self._qmul(des_q, self._qinv(xqp))
        Tp = des_p - self._qrot(Tq, xpp)
        self.data.qpos[self._cfree_adr:self._cfree_adr + 3] = Tp + self._qrot(Tq, xpc)
        self.data.qpos[self._cfree_adr + 3:self._cfree_adr + 7] = self._qmul(Tq, xqc)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _ik(self, target_pos, target_quat, q_init):
        """Damped-least-squares 6-DoF IK for gripper_tcp; returns arm qpos."""
        q = np.array(q_init, dtype=np.float64)
        Jp, Jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(self.cfg.ik_iters):
            self.data.qpos[self._arm_qadr] = q
            mujoco.mj_forward(self.model, self.data)
            perr = target_pos - self.data.site_xpos[self._tcp_sid]
            qerr = np.zeros(3)
            mujoco.mju_subQuat(qerr, target_quat, self._site_quat())
            err = np.concatenate([perr, qerr])
            if np.linalg.norm(err) < self.cfg.ik_tol:
                break
            mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self._tcp_sid)
            J = np.vstack([Jp[:, self._arm_vadr], Jr[:, self._arm_vadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + self.cfg.ik_damping * np.eye(6), err)
            n = np.linalg.norm(dq)
            if n > self.cfg.ik_step_max:
                dq *= self.cfg.ik_step_max / n
            q = q + dq
            for k, adr in enumerate(self._arm_qadr):
                jid = int(np.flatnonzero(self.model.jnt_qposadr == adr)[0])
                lo, hi = self.model.jnt_range[jid]
                q[k] = np.clip(q[k], lo, hi) if self.model.jnt_limited[jid] else q[k]
        return q

    def _ik_tcp_position(self, target_pos, q_init):
        """Position-dominant seed IK for the TCP.

        The SFP port frame is reachable in position, but strict wrist
        orientation can stall in the exported scene. This seed keeps the arm
        near the AIC posture; `_ik_tip_position` then corrects the welded tip.
        """
        q = np.array(q_init, dtype=np.float64)
        Jp, Jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(self.cfg.ik_iters):
            self.data.qpos[self._arm_qadr] = q
            mujoco.mj_forward(self.model, self.data)
            err = np.asarray(target_pos, dtype=np.float64) - self.data.site_xpos[self._tcp_sid]
            if np.linalg.norm(err) < self.cfg.ik_tol:
                break
            mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self._tcp_sid)
            J = Jp[:, self._arm_vadr]
            dq = J.T @ np.linalg.solve(J @ J.T + self.cfg.ik_damping * np.eye(3), err)
            n = np.linalg.norm(dq)
            if n > self.cfg.ik_step_max:
                dq *= self.cfg.ik_step_max / n
            q = self._clip_arm(q + dq)
        return q

    def _tip_at_arm(self, q_arm):
        self._rigid_home(q_arm)
        return self.data.xpos[self._plug_tip_id].copy()

    def _tip_axis_at_arm(self, q_arm):
        self._rigid_home(q_arm)
        return self.data.xpos[self._plug_tip_id].copy(), self._plug_axis()

    def _ik_tip_position(self, target_tip, q_init):
        """Finite-difference IK on the welded SFP tip position.

        MuJoCo's kinematic Jacobian does not directly expose the arm-to-plug
        weld after we rigidly reposition the cable free joint at reset, so this
        reset-only correction numerically differentiates the actual welded tip.
        """
        q = np.array(q_init, dtype=np.float64)
        target_tip = np.asarray(target_tip, dtype=np.float64)
        eps = float(self.cfg.ik_fd_eps)
        best_q = q.copy()
        best_err = float("inf")
        for _ in range(self.cfg.ik_tip_iters):
            tip = self._tip_at_arm(q)
            err = target_tip - tip
            err_norm = float(np.linalg.norm(err))
            if err_norm < best_err:
                best_q, best_err = q.copy(), err_norm
            if err_norm < self.cfg.ik_tol:
                break
            J = np.zeros((3, 6), dtype=np.float64)
            for k in range(6):
                q_eps = q.copy()
                q_eps[k] += eps
                J[:, k] = (self._tip_at_arm(q_eps) - tip) / eps
            dq = J.T @ np.linalg.solve(J @ J.T + self.cfg.ik_tip_damping * np.eye(3), err)
            n = np.linalg.norm(dq)
            if n > self.cfg.ik_tip_step_max:
                dq *= self.cfg.ik_tip_step_max / n
            q = self._clip_arm(q + dq)
        final_tip = self._tip_at_arm(best_q)
        return best_q, float(np.linalg.norm(target_tip - final_tip))

    def _ik_tip_axis(self, target_tip, target_axis, q_init):
        """Finite-difference reset IK on welded tip position + plug axis.

        The old reset matched the tip position but left the plug attitude mostly
        inherited from the arm home posture. For the SFP cage that can look like
        a diagonal pass through the port. This reset-only solve aligns the actual
        module->tip direction with the port inward axis.
        """
        q = np.array(q_init, dtype=np.float64)
        target_tip = np.asarray(target_tip, dtype=np.float64)
        target_axis = self._unit(target_axis, self._insert_axis)
        eps = float(self.cfg.ik_fd_eps)
        weight = float(self.cfg.ik_axis_weight_m)
        best_q = q.copy()
        best_score = float("inf")
        best_pos = float("inf")
        best_axis = float("inf")

        for _ in range(self.cfg.ik_tip_iters):
            tip, axis = self._tip_axis_at_arm(q)
            pos_err = target_tip - tip
            axis_delta = target_axis - axis
            axis_angle = float(np.arccos(np.clip(np.dot(axis, target_axis), -1.0, 1.0)))
            pos_norm = float(np.linalg.norm(pos_err))
            score = pos_norm + weight * axis_angle
            if score < best_score:
                best_q, best_score = q.copy(), score
                best_pos, best_axis = pos_norm, axis_angle
            if pos_norm < self.cfg.ik_tol and axis_angle < self.cfg.ik_axis_tol_rad:
                break

            err = np.concatenate([pos_err, weight * axis_delta])
            J = np.zeros((6, 6), dtype=np.float64)
            for k in range(6):
                q_eps = q.copy()
                q_eps[k] += eps
                tip_eps, axis_eps = self._tip_axis_at_arm(q_eps)
                J[:3, k] = (tip_eps - tip) / eps
                J[3:, k] = weight * (axis_eps - axis) / eps
            dq = J.T @ np.linalg.solve(J @ J.T + self.cfg.ik_tip_damping * np.eye(6), err)
            n = np.linalg.norm(dq)
            if n > self.cfg.ik_tip_step_max:
                dq *= self.cfg.ik_tip_step_max / n
            q = self._clip_arm(q + dq)

        tip, axis = self._tip_axis_at_arm(best_q)
        best_pos = float(np.linalg.norm(target_tip - tip))
        best_axis = float(np.arccos(np.clip(np.dot(axis, target_axis), -1.0, 1.0)))
        return best_q, best_pos, best_axis

    def _clip_arm(self, q):
        q = np.asarray(q, dtype=np.float64).copy()
        for k, adr in enumerate(self._arm_qadr):
            jid = int(np.flatnonzero(self.model.jnt_qposadr == adr)[0])
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                q[k] = np.clip(q[k], lo, hi)
        return q

    def _sample_start_tcp(self, level, jitter=True, rng=None):
        """Last-inch curriculum in the port frame."""
        rng = rng or np.random
        level = float(np.clip(level, 0.0, 1.0))
        max_retract = level * self.cfg.last_inch_m
        retract = rng.uniform(0.0, max_retract) if (jitter and max_retract > 0) else max_retract
        tip = self._inserted_tip + retract * self._retract_dir
        quat = self._goal_quat.copy()
        if jitter and level > 0:
            tip = tip + self._lat_x * rng.uniform(-1, 1) * self.cfg.jitter_xy_m * level
            tip = tip + self._lat_y * rng.uniform(-1, 1) * self.cfg.jitter_xy_m * level
            yaw = rng.uniform(-1, 1) * self.cfg.jitter_yaw_rad * level
            tilt_x = rng.uniform(-1, 1) * self.cfg.jitter_tilt_rad * level
            tilt_y = rng.uniform(-1, 1) * self.cfg.jitter_tilt_rad * level
            dq = self._qmul(self._axis_angle(self._insert_axis, yaw),
                            self._qmul(self._axis_angle(self._lat_x, tilt_x),
                                       self._axis_angle(self._lat_y, tilt_y)))
            quat = self._qmul(dq, quat)
        tcp = self._tcp_for_tip(tip, quat)
        return tcp, quat, tip

    def _reset_to_level(self, level, jitter=True, settle=None):
        self._select_target_body()
        attempts = max(1, int(self.cfg.reset_max_attempts))
        best = None
        n_settle = self.cfg.settle_steps if settle is None else settle
        for attempt in range(attempts):
            tcp, quat, target_tip = self._sample_start_tcp(level, jitter=jitter, rng=self.np_random)
            q_seed = self._ik(tcp, quat, self._home)
            q_seed = self._ik_tcp_position(tcp, q_seed)
            q_arm, ik_tip_err, ik_axis_err = self._ik_tip_axis(target_tip, self._insert_axis, q_seed)
            self._rigid_home(q_arm)
            for _ in range(n_settle):
                self.data.ctrl[:6] = self._base_torque(q_arm)
                self.data.ctrl[6] = self.cfg.gripper_ctrl
                mujoco.mj_step(self.model, self.data)
            diag = self._geometry_diag(target_tip=target_tip, ik_tip_err=ik_tip_err,
                                       ik_axis_err=ik_axis_err)
            score = (diag["tip_error_m"]
                     + 0.02 * diag["plug_axis_error_rad"]
                     + 2.0 * diag["plug_port_penetration_m"]
                     + 0.001 * diag["contact_force_norm"])
            if best is None or score < best[0]:
                best = (score, q_arm.copy(), diag)
            if (diag["tip_error_m"] <= self.cfg.reset_ik_tip_tol_m
                    and diag["plug_axis_error_rad"] <= self.cfg.ik_axis_tol_rad
                    and diag["plug_port_penetration_m"] <= self.cfg.reset_max_plug_port_penetration_m
                    and diag["contact_force_norm"] <= self.cfg.reset_contact_abort_n):
                self._last_reset_diag = diag
                break
        else:
            _, q_arm, diag = best
            self._rigid_home(q_arm)
            for _ in range(n_settle):
                self.data.ctrl[:6] = self._base_torque(q_arm)
                self.data.ctrl[6] = self.cfg.gripper_ctrl
                mujoco.mj_step(self.model, self.data)
            self._last_reset_diag = diag
        self._reset_arm_target = q_arm.copy()
        self._arm_target = q_arm.copy()

    def _base_torque(self, target):
        q = self.data.qpos[self._arm_qadr]; qd = self.data.qvel[self._arm_vadr]
        return self.cfg.kp * (target - q) - self.cfg.kd * qd + self.data.qfrc_bias[self._arm_vadr]

    def _clip_action_target(self, target):
        lo = self._reset_arm_target - self.cfg.action_joint_limit
        hi = self._reset_arm_target + self.cfg.action_joint_limit
        return self._clip_arm(np.clip(target, lo, hi))

    # ------------------------------------------------------------------ #
    def set_curriculum_level(self, level: float):
        self._curriculum_level = float(np.clip(level, 0.0, 1.0))

    def get_curriculum_level(self) -> float:
        return self._curriculum_level

    def set_reset_mode(self, mode: str):
        """'curriculum' | 'random' (level 1) | 'near_goal' (level 0). Stored for parity."""
        self._reset_mode = mode

    def set_level_file(self, path: Optional[str]):
        self._level_file = Path(path) if path else None

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if self._level_file is not None and self._level_file.exists():
            try:
                self._curriculum_level = float(self._level_file.read_text().strip())
            except Exception:
                pass
        lvl = getattr(self, "_reset_mode", "curriculum")
        level = 1.0 if lvl == "random" else 0.0 if lvl == "near_goal" else self._curriculum_level
        self._reset_to_level(level, jitter=True)
        self._step_count = 0
        self._last_action = np.zeros(6, np.float32)
        self._prev_depth_norm = self._depth_norm()
        self._f_ax_buf: list[float] = []
        return self._obs(), {
            "curriculum_level": self._curriculum_level,
            "reset_diag": getattr(self, "_last_reset_diag", None),
        }

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64).reshape(6), -1.0, 1.0)
        self._arm_target = self._clip_action_target(
            self._arm_target + action * self.cfg.action_joint_scale)
        for _ in range(self.cfg.control_substeps):
            self.data.ctrl[:6] = self._base_torque(self._arm_target)
            self.data.ctrl[6] = self.cfg.gripper_ctrl
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._obs()
        total, breakdown, term_status = self._reward_and_term(obs, action)
        self._last_action = action.astype(np.float32)
        terminated = term_status in ("success", "force_abort", "bad_collision", "off_limit")
        truncated = (term_status == "timeout") or (self._step_count >= self.cfg.max_episode_steps)
        info = {
            "term_status": term_status if term_status else ("timeout" if truncated else None),
            "breakdown": breakdown,
            "image_l1_norm": breakdown.image_l1_norm,
            "f_z": float(self._f_axial),
            "f_z_mean": float(np.mean(self._f_ax_buf)) if self._f_ax_buf else float("nan"),
            "f_z_max": float(np.max(np.abs(self._f_ax_buf))) if self._f_ax_buf else float("nan"),
            "depth_norm": float(self._prev_depth_norm),
            "axial_error_m": float(getattr(self, "_axial_error", float("nan"))),
            "lateral_error_m": float(getattr(self, "_lateral_error", float("nan"))),
            **{k: v for k, v in self._contact_summary().items()
               if not isinstance(v, str)},
            "plug_axis_error_rad": float(self._plug_axis_error()),
            "wallclock": float(self._step_count) * 0.05,
            "curriculum_level": float(self._curriculum_level),
        }
        return obs, total, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _contact_force(self):
        """World-frame contact force on the plug = FT minus the gripped-plug baseline."""
        return self._raw_ft()[:3] - self._ft_baseline

    def _body_name(self, bid: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{bid}"

    def _geom_name(self, gid: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"

    def _contact_summary(self):
        plug_bodies = {int(self._plug_id), int(self._plug_tip_id), int(self._plug_axis_tail_id)}
        target_tokens = ("nic_card", "sfp_port", "sfp", "mount")
        min_dist = float("inf")
        plug_port_min = float("inf")
        worst_pair = ""
        plug_port_worst = ""
        plug_port_contacts = 0
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            n1, n2 = self._body_name(b1), self._body_name(b2)
            ge1, ge2 = self._geom_name(g1), self._geom_name(g2)
            dist = float(c.dist)
            if dist < min_dist:
                min_dist = dist
                worst_pair = f"{n1}:{ge1} <-> {n2}:{ge2}"

            one_plug = (b1 in plug_bodies) ^ (b2 in plug_bodies)
            other_name = n2 if b1 in plug_bodies else n1
            is_portish = any(tok in other_name for tok in target_tokens)
            if one_plug and is_portish:
                plug_port_contacts += 1
                if dist < plug_port_min:
                    plug_port_min = dist
                    plug_port_worst = f"{n1}:{ge1} <-> {n2}:{ge2}"

        if not np.isfinite(min_dist):
            min_dist = 0.0
        if not np.isfinite(plug_port_min):
            plug_port_min = 0.0
        plug_port_pen = float(max(0.0, -plug_port_min))
        goal_pen = float(getattr(self, "_goal_plug_port_penetration_m", plug_port_pen))
        return {
            "ncon": int(self.data.ncon),
            "min_contact_dist_m": float(min_dist),
            "max_penetration_m": float(max(0.0, -min_dist)),
            "worst_contact_pair": worst_pair,
            "plug_port_contacts": int(plug_port_contacts),
            "plug_port_min_dist_m": float(plug_port_min),
            "plug_port_penetration_m": plug_port_pen,
            "plug_port_penetration_excess_m": float(max(0.0, plug_port_pen - goal_pen)),
            "plug_port_worst_pair": plug_port_worst,
        }

    def _depth_norm(self) -> float:
        """Insertion fraction in [0,1]: 1 seated, 0 retracted by last_inch_m."""
        tip = self.data.xpos[self._plug_tip_id]
        retract = float(np.dot(tip - self._inserted_tip, self._retract_dir))
        return float(np.clip(1.0 - retract / max(self.cfg.last_inch_m, 1e-6), 0.0, 1.0))

    def _tip_port_errors(self, tip=None):
        tip = self.data.xpos[self._plug_tip_id] if tip is None else np.asarray(tip)
        delta = tip - self._inserted_tip
        axial = float(np.dot(delta, self._retract_dir))
        lateral = np.array([np.dot(delta, self._lat_x), np.dot(delta, self._lat_y)],
                           dtype=np.float32)
        return axial, lateral

    def _plug_axis_error(self) -> float:
        return float(np.arccos(np.clip(np.dot(self._plug_axis(), self._insert_axis), -1.0, 1.0)))

    def _reward_and_term(self, obs, action):
        fc = self._contact_force()
        f_axial = float(np.dot(fc, self._insert_axis))
        f_lat = float(np.linalg.norm(fc - f_axial * self._insert_axis))
        self._f_axial = f_axial
        self._f_ax_buf.append(f_axial)
        tip = self.data.xpos[self._plug_tip_id]
        axial_err, lat_vec = self._tip_port_errors(tip)
        lat_err = float(np.linalg.norm(lat_vec))
        self._axial_error = axial_err
        self._lateral_error = lat_err
        depth_norm = self._depth_norm()
        contact_summary = self._contact_summary()
        axis_error = self._plug_axis_error()
        pen_excess = float(contact_summary["plug_port_penetration_excess_m"])

        # success = bottomed depth + seated pose + straight, collision-clean seating contact
        term_status = None
        if (depth_norm >= self.cfg.success_depth_norm
                and abs(axial_err) < self.cfg.success_axial_tol_m
                and lat_err < self.cfg.success_lateral_tol_m
                and axis_error <= self.cfg.success_axis_tol_rad
                and pen_excess <= self.cfg.success_max_plug_port_penetration_excess_m
                and abs(f_axial) > self.cfg.success_force_n):
            term_status = "success"
        elif (pen_excess > self.cfg.bad_collision_penetration_excess_m
              or (depth_norm >= self.cfg.bad_collision_depth_gate
                  and axis_error > self.cfg.bad_collision_axis_rad)):
            term_status = "bad_collision"
        elif (abs(f_axial) > self.cfg.force_abort_n
              or f_lat > self.cfg.force_abort_n
              or np.linalg.norm(fc) > self.cfg.force_abort_n):
            term_status = "force_abort"

        img_curr = self._render_reward_image()
        total, breakdown = compute_reward(
            image_curr=img_curr, image_goal=self._goal_image_reward,
            f_z=f_axial, f_xy=np.array([f_lat, 0.0], np.float32),
            tip_xy=lat_vec, port_xy=np.zeros(2, np.float32),
            a_t=action.astype(np.float32), a_prev=self._last_action,
            term_status=term_status, cfg=self.cfg.reward,
            bonus_eligible=(term_status == "success"),
            depth_norm=depth_norm, prev_depth_norm=self._prev_depth_norm,
            axis_error_rad=axis_error,
            penetration_excess_m=pen_excess,
        )
        self._prev_depth_norm = depth_norm
        return float(total), breakdown, term_status

    def _render_obs_image(self):
        frames = []
        for cam in self._cams:
            self._renderer.update_scene(self.data, camera=cam)
            frames.append(self._renderer.render())
        return np.concatenate(frames, axis=2)

    def _render_reward_image(self):
        if self._reward_renderer is None:
            return np.zeros((1, 1, 3), np.uint8)
        self._reward_renderer.update_scene(self.data, camera="center_camera")
        return self._reward_renderer.render()

    def _geometry_diag(self, target_tip=None, ik_tip_err=None, ik_axis_err=None):
        tip = self.data.xpos[self._plug_tip_id].copy()
        target = self._inserted_tip if target_tip is None else np.asarray(target_tip, dtype=np.float64)
        axial_err, lat_vec = self._tip_port_errors(tip)
        fc = self._contact_force()
        f_axial = float(np.dot(fc, self._insert_axis))
        f_lat = float(np.linalg.norm(fc - f_axial * self._insert_axis))
        axis_err = self._plug_axis_error()
        out = {
            "target_body": self._target_name,
            "entrance": self._port_pos.copy(),
            "seated_tip": self._inserted_tip.copy(),
            "insert_axis": self._insert_axis.copy(),
            "plug_axis": self._plug_axis().copy(),
            "tip": tip,
            "target_tip": target.copy(),
            "tip_error_m": float(np.linalg.norm(tip - target)),
            "ik_tip_error_m": float(ik_tip_err if ik_tip_err is not None else np.linalg.norm(tip - target)),
            "plug_axis_error_rad": axis_err,
            "plug_axis_error_deg": float(np.degrees(axis_err)),
            "ik_axis_error_rad": float(ik_axis_err if ik_axis_err is not None else axis_err),
            "axial_error_m": float(axial_err),
            "lateral_error_m": float(np.linalg.norm(lat_vec)),
            "depth_norm": float(self._depth_norm()),
            "contact_force_norm": float(np.linalg.norm(fc)),
            "f_axial": f_axial,
            "f_lateral": f_lat,
        }
        out.update(self._contact_summary())
        return out

    def _obs(self) -> dict:
        d = self.data
        tcp_pos = d.site_xpos[self._tcp_sid].copy()
        obs = {
            "arm_qpos": d.qpos[self._arm_qadr].astype(np.float32),
            "arm_qvel": d.qvel[self._arm_vadr].astype(np.float32),
            "tcp_pose": np.concatenate([tcp_pos, self._site_quat()]).astype(np.float32),
            "ft": self._raw_ft().astype(np.float32),
            "last_action": self._last_action.copy(),
        }
        if self._renderer is not None and self._cams:
            obs["image"] = self._render_obs_image().astype(np.uint8)
        return obs

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera=self._cams[0] if self._cams else -1)
        return self._renderer.render()[:, :, :3]

    def close(self):
        for r in (getattr(self, "_renderer", None), getattr(self, "_reward_renderer", None)):
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
        self._renderer = None
        self._reward_renderer = None


__all__ = ["SceneEnvConfig", "SceneInsertEnv"]


if __name__ == "__main__":
    cfg = SceneEnvConfig(include_images=bool(int(os.environ.get("AIC_IMAGES", "1"))),
                         reward_image_res=int(os.environ.get("AIC_REWARD_RES", "0")))
    env = SceneInsertEnv(cfg)
    print("obs keys:", list(env.observation_space.spaces.keys()))
    print("target:", env._target_name,
          "entrance:", np.round(env._port_pos, 5),
          "seated_tip:", np.round(env._inserted_tip, 5),
          "axis:", np.round(env._insert_axis, 5),
          "goal_plug_axis_err_deg:", round(float(np.degrees(env._plug_axis_error())), 2),
          "goal_tcp:", np.round(env._goal_tcp, 5),
          "ft_baseline:", np.round(env._ft_baseline, 2))
    last_obs = None
    for lvl in (0.0, 0.5, 1.0):
        env.set_curriculum_level(lvl)
        obs, info = env.reset(seed=0)
        diag = env._geometry_diag()
        print(f"  reset lvl={lvl}: tip_err={diag['tip_error_m']:.4f} "
              f"axial_err={diag['axial_error_m']:.4f} "
              f"lat_err={diag['lateral_error_m']:.4f} "
              f"depth={diag['depth_norm']:.2f} "
              f"axis_deg={diag['plug_axis_error_deg']:.1f} "
              f"pen={diag['plug_port_penetration_m']*1000:.1f}mm "
              f"excess={diag['plug_port_penetration_excess_m']*1000:.1f}mm "
              f"contact={diag['contact_force_norm']:.2f} "
              f"f_axial={diag['f_axial']:.2f} ncon={diag['ncon']}")
        obs_hold, r_hold, term_hold, trunc_hold, i_hold = env.step(np.zeros(6, np.float32))
        print(f"  hold  lvl={lvl}: f_z={i_hold['f_z']:.2f} "
              f"depth={i_hold['depth_norm']:.2f} "
              f"axial_err={i_hold['axial_error_m']:.4f} "
              f"lat_err={i_hold['lateral_error_m']:.4f} "
              f"axis_deg={np.degrees(i_hold['plug_axis_error_rad']):.1f} "
              f"pen={i_hold['plug_port_penetration_m']*1000:.1f}mm "
              f"excess={i_hold['plug_port_penetration_excess_m']*1000:.1f}mm "
              f"reward={r_hold:.3f} term={i_hold['term_status']}")
        obs, info = env.reset(seed=0)
        r = 0.0; term = trunc = False
        for _ in range(30):
            obs, r, term, trunc, i = env.step(env.action_space.sample() * 0.3)
            if term or trunc:
                break
        img = obs.get("image")
        print(f"  step  lvl={lvl}: tcp={np.round(obs['tcp_pose'][:3],3)} "
              f"f_z={i['f_z']:.2f} depth={i['depth_norm']:.2f} "
              f"axial_err={i['axial_error_m']:.4f} lat_err={i['lateral_error_m']:.4f} "
              f"axis_deg={np.degrees(i['plug_axis_error_rad']):.1f} "
              f"pen={i['plug_port_penetration_m']*1000:.1f}mm "
              f"excess={i['plug_port_penetration_excess_m']*1000:.1f}mm "
              f"reward={r:.3f} term={i['term_status']} "
              f"img={None if img is None else img.shape}")
        last_obs = obs
    if bool(int(os.environ.get("AIC_WRITE_TEST_IMAGES", "0"))) and last_obs is not None:
        out = Path(__file__).resolve().parent / "tests"
        out.mkdir(exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.imwrite(out / "new_center_camera.png", env._render_reward_image())
            if "image" in last_obs:
                imageio.imwrite(out / "new_wrist_cams.png", last_obs["image"][:, :, :3])
            print(f"wrote new_* verification images under {out}")
        except Exception as exc:
            print(f"WARN: could not write new_* verification images: {exc}")
    env.close()
    print("SCENE ENV (real ports) SMOKE OK")
