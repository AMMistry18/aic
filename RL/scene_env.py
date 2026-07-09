"""Gymnasium env wrapping the REAL MuJoCo scene (`aic_utils/aic_mujoco/mjcf/scene.xml`).

Sim-side backend for last-inch insertion RL on the Gazebo->MuJoCo pipeline scene
(UR5e + Robotiq Hand-E + welded LC/SFP plug + elastic cable + task board + the
**real receptacle ports**: NIC card SFP cage `nic_card_mount_2` + `sc_port_0/1`).

This is the maintained environment for the actual robot/scene.

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
  * Geometry-first reward from `RL/reward.py:compute_reward` (depth progress +
    sparse success, with small alignment/contact/action safety costs). The image
    observation is still available to the policy, but the image-distance reward
    is optional and defaults off.
  * SUCCESS = plug tip near the seated pose, depth at the bottom of the last-inch
    envelope, keyed roll aligned, and collision-clean. A seating force check is
    available via `success_force_n` but defaults off because the exported SFP cage
    can have valid clearance with zero active contact at the seated pose.
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
    # AIC runs its impedance controller at 500 Hz. Keep one policy action for
    # 50 ms (20 Hz), matching the final-insertion policy cadence and score
    # timing used by the AIC/Gazebo stack.
    policy_dt_s: float = 0.05
    control_substeps: int = 25
    max_episode_steps: int = 200
    # Joint impedance defaults from aic_bringup/config/aic_ros2_controllers.yaml.
    # MuJoCo's general actuators accept torque commands, so this is the same
    # K(q_des-q) + D(dq_des-dq) + gravity-compensation form as AIC.
    joint_stiffness: tuple = (100.0, 100.0, 100.0, 50.0, 50.0, 50.0)
    joint_damping: tuple = (40.0, 40.0, 40.0, 15.0, 15.0, 15.0)
    joint_torque_limit: tuple = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    action_joint_scale: float = 0.01   # rad per policy step at |action|=1
    action_joint_limit: float = 0.35   # rad envelope around the reset arm target
    gripper_ctrl: float = 0.0
    # --- insertion target: the REAL fixed SFP receptacle on the board ---
    insert_target_body: str = "sfp_port_1_link_entrance"
    insert_target_bodies: tuple = ()    # optional future randomisation hook
    randomize_target_body: bool = False
    plug_tip_body: str = "sfp_tip_link"
    plug_axis_tail_body: str = "sfp_module_link"
    # Keep the TCP offset reachable; the reset IK below aligns the actual plug axis
    # and keyed roll. SFP insertion is not defined by axis alone: a 90 deg roll
    # can still report a tiny axis error while the module is sideways in the cage.
    align_plug_axis_to_port: bool = False
    align_plug_roll_to_port: bool = True
    plug_roll_body: str = "sfp_tip_link"
    plug_roll_axis_index: int = 0        # local +X of the tip/module cross-section
    plug_roll_target_axis: str = "x"     # port-frame lateral axis: x, y, -x, -y
    insert_goal_offset: tuple = (0.0, 0.0, 0.0)   # world offset for fine-tuning
    insertion_axis_world: Optional[tuple] = None   # default: target body local +Z
    # Gazebo NIC Card Mount defines sfp_port_X_link_entrance at z=-45.8 mm
    # relative to sfp_port_X_link. The entrance is the mouth; sfp_port_X_link is
    # the fully seated target frame.
    seated_depth_m: float = 0.0458
    # --- last-inch reverse curriculum (relative to the port) ---
    # The retract span now gives a real free-space approach. With seated depth
    # ~45.8 mm and span 90 mm, level 0.8 starts the plug tip ~26 mm (~1 inch)
    # outside the entrance plane; level 1 is farther back. This better matches
    # the AIC last-inch task, where pose estimation is imperfect and the policy
    # must acquire the port from above/nearby rather than start already nicked
    # into the cage.
    last_inch_m: float = 0.090
    curriculum_level: float = 0.0        # 0 = seated, 1 = retracted last_inch_m
    curriculum_band: float = 0.25        # frontier band width (fraction of level)
    curriculum_easy_frac: float = 0.2    # prob of an easy replay start in [0, level]
    # two-phase jitter: tight while the tip is inside the cage, wide outside
    jitter_xy_inport_m: float = 0.0008   # lateral jitter while inside the port
    jitter_yaw_inport_rad: float = 0.03  # yaw jitter while inside the port
    jitter_tilt_inport_rad: float = 0.01 # tilt jitter while inside the port
    jitter_xy_m: float = 0.006           # lateral jitter at level 1, fully outside
    jitter_yaw_rad: float = 0.12         # yaw about insertion axis at level 1
    jitter_tilt_rad: float = 0.04        # port-frame x/y tilt at level 1, fully outside
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
    ik_roll_weight_m: float = 0.04
    ik_roll_tol_rad: float = 0.12
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
    success_axial_tol_m: float = 0.003
    success_lateral_tol_m: float = 0.005
    success_depth_norm: float = 0.99
    success_axis_tol_rad: float = 0.035
    success_roll_tol_rad: float = 0.15
    success_force_n: float = 0.0          # <= 0 disables optional seating-force gate
    success_max_plug_port_penetration_excess_m: float = 0.001
    success_max_overinsert_m: float = 0.001
    success_require_port_contact: bool = True
    # mid-episode kill bounds are RECOVERY bounds, not quality bounds (success
    # strictness is separate and unchanged). At 0.20 rad / 3 mm the exploring
    # policy died exactly AT the bounds after ~145-step drifts (metrics
    # 2026-07-04) instead of learning to re-align — the dense axis/collision
    # costs already penalize those states continuously.
    bad_collision_axis_rad: float = 0.35
    bad_collision_roll_rad: float = 0.35
    # axis-bend abort only applies once the tip is actually inside the cage.
    bad_collision_depth_gate: float = 0.45
    bad_collision_penetration_excess_m: float = 0.0015
    bad_collision_overinsert_m: float = 0.002
    # penetration baseline vs depth: the exported cage's soft-contact overlap is
    # depth-dependent (~5.4 mm mid-cage vs 3.4 mm seated), so a single seated
    # baseline reads ~2 mm phantom "excess" mid-insertion and the collision term
    # taxes the whole traversal. Calibrated at construction along the axis.
    pen_baseline_points: int = 9
    pen_baseline_margin_m: float = 0.0005
    # force abort: transient FT spikes (PD contact) were ending 43% of episodes
    # at level 0.1; abort only on SUSTAINED force, with a high instant bound.
    force_abort_n: float = 60.0          # sustained-force abort threshold
    force_abort_dwell_steps: int = 3     # consecutive steps over force_abort_n
    force_abort_hard_n: float = 120.0    # instant abort (true safety bound)
    # reject settled in-cage starts wedged beyond the port clearance (the
    # 6 mm reset IK tolerance otherwise seeds unrecoverable lateral jams)
    reset_inport_lateral_tol_m: float = 0.003
    # --- score-style diagnostics (local estimator of the published rubric) ---
    score_lateral_tol_m: float = 0.005
    score_axial_tol_m: float = 0.003
    score_full_depth_norm: float = 0.99
    score_axis_tol_rad: float = 0.035
    score_roll_tol_rad: float = 0.15
    score_max_penetration_excess_m: float = 0.001
    score_force_threshold_n: float = 20.0
    score_force_duration_s: float = 1.0
    score_wrong_port_penalty: float = -12.0
    score_off_limit_penalty: float = -24.0
    # --- action mode ------------------------------------------------------
    # 'joint_residual'     = original behavior (6-D joint residual -> joint PD)
    # 'cartesian_residual' = SAC action is a small Cartesian TCP residual in the
    #   PORT frame [dx, dy, dz(=deeper), droll(about lat_x), dpitch(about lat_y),
    #   dyaw(about insert axis)], tracked by a Jacobian-transpose Cartesian
    #   impedance controller that mirrors aic_controller's
    #   CartesianImpedanceAction (tau = J^T (K dx + D dv) + gravity comp).
    action_mode: str = "joint_residual"
    cart_action_dims: int = 6            # 6 = [dx,dy,dz,droll,dpitch,dyaw]; 5 drops dyaw
    cart_trans_scale_m: float = 0.001    # 1 mm per policy step at |action|=1
    cart_rot_scale_rad: float = 0.0175   # ~1 deg per policy step at |action|=1
    # hard safety envelope for the ACCUMULATED residual around the reset TCP
    # pose (per port-frame axis). z must cover the full last-inch travel.
    cart_pos_limit_m: float = 0.10
    cart_rot_limit_rad: float = 0.35
    # Cartesian impedance defaults from aic_ros2_controllers.yaml:
    # stiffness [75]*6, damping [35]*6, maximum_wrench [10]*6, and a
    # damping-only nullspace term of [10]*6.
    cart_kp_pos: float = 75.0             # N/m
    cart_kd_pos: float = 35.0             # N/(m/s)
    cart_kp_rot: float = 75.0             # Nm/rad
    cart_kd_rot: float = 35.0             # Nm/(rad/s)
    cart_max_wrench: tuple = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    cart_nullspace_damping: float = 10.0
    # the <general> actuators are force-unlimited, so clip here (UR5e efforts)
    cart_torque_limit: tuple = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    # --- optional scripted base insertion (residual-on-script) -------------
    # When enabled (cartesian_residual only), the BASE target pose advances
    # along +insert_axis by base_script_step_m per policy step (clamped at the
    # seated goal); the SAC residual rides on top with a TIGHTER envelope, so
    # the policy learns the correction, not the transport.
    base_script_enabled: bool = False
    base_script_step_m: float = 0.0005   # 0.5 mm per policy step
    base_script_residual_limit_m: float = 0.01
    base_script_residual_limit_rad: float = 0.10


class SceneInsertEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: SceneEnvConfig = SceneEnvConfig()):
        super().__init__()
        self.cfg = cfg
        if cfg.action_mode not in ("joint_residual", "cartesian_residual"):
            raise ValueError(f"unknown action_mode: {cfg.action_mode!r}")
        self._cart_mode = cfg.action_mode == "cartesian_residual"
        if self._cart_mode and int(cfg.cart_action_dims) not in (5, 6):
            raise ValueError(f"cart_action_dims must be 5 or 6, got {cfg.cart_action_dims}")
        if cfg.base_script_enabled and not self._cart_mode:
            raise ValueError("base_script_enabled requires action_mode='cartesian_residual'")
        self._action_dim = int(cfg.cart_action_dims) if self._cart_mode else 6
        self._cart_max_wrench = np.asarray(cfg.cart_max_wrench, dtype=np.float64)
        self._cart_tau_limit = np.asarray(cfg.cart_torque_limit, dtype=np.float64)
        self._joint_stiffness = np.asarray(cfg.joint_stiffness, dtype=np.float64)
        self._joint_damping = np.asarray(cfg.joint_damping, dtype=np.float64)
        self._joint_tau_limit = np.asarray(cfg.joint_torque_limit, dtype=np.float64)
        if any(v.shape != (6,) for v in (
                self._cart_max_wrench, self._cart_tau_limit,
                self._joint_stiffness, self._joint_damping, self._joint_tau_limit)):
            raise ValueError("all arm impedance vectors must contain six values")
        self.model = mujoco.MjModel.from_xml_path(cfg.scene_path)
        self.data = mujoco.MjData(self.model)
        if int(cfg.control_substeps) < 1:
            raise ValueError("control_substeps must be positive")
        self._physics_dt_s = float(self.model.opt.timestep)
        self._policy_dt_s = self._physics_dt_s * int(cfg.control_substeps)
        if not np.isclose(self._policy_dt_s, cfg.policy_dt_s, rtol=0.0,
                          atol=self._physics_dt_s * 0.5):
            raise ValueError(
                "policy_dt_s must equal scene timestep * control_substeps; "
                f"got {cfg.policy_dt_s:g} != {self._physics_dt_s:g} * "
                f"{cfg.control_substeps} ({self._policy_dt_s:g})"
            )

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
        self._plug_roll_id = bid(cfg.plug_roll_body)
        if self._plug_roll_id < 0:
            self._plug_roll_id = self._plug_tip_id
        self._cend_id = bid("cable_end_0")
        self._cfree_adr = self.model.jnt_qposadr[jid("cable_end_0_free")]
        self._plug_body_ids = self._body_subtree_ids(self._plug_id)
        self._manipulated_body_ids = self._find_manipulated_body_ids()
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
        self._last_action = np.zeros(self._action_dim, np.float32)
        self._arm_target = self._home.copy()
        self._prev_depth_norm = 0.0

        # obs renderer (3 wrist cams) + reward renderer (center cam, native res)
        self._renderer, self._cams = None, []
        self._reward_renderer = None
        self._image_reward_enabled = (
            abs(float(getattr(cfg.reward, "w_image", 0.0))) > 0.0
            or abs(float(getattr(cfg.reward, "beta_s", 0.0))) > 0.0
        )
        self._debug_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        if cfg.include_images:
            self._renderer = mujoco.Renderer(self.model, height=cfg.image_h, width=cfg.image_w)
            self._cams = [c for c in cfg.cameras
                          if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c) >= 0]
            if self._image_reward_enabled:
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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._action_dim,), dtype=np.float32)
        obs_spaces = {
            "arm_qpos": spaces.Box(-np.pi, np.pi, (6,), np.float32),
            "arm_qvel": spaces.Box(-50.0, 50.0, (6,), np.float32),
            "tcp_pose": spaces.Box(-3.0, 3.0, (7,), np.float32),
            "ft": spaces.Box(-500.0, 500.0, (6,), np.float32),
            "last_action": spaces.Box(-1.0, 1.0, (self._action_dim,), np.float32),
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
        self._calibrate_pen_baseline()

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

    def _project_to_insert_plane(self, v, fallback):
        v = np.asarray(v, dtype=np.float64)
        v = v - np.dot(v, self._insert_axis) * self._insert_axis
        return self._unit(v, fallback)

    def _target_roll_axis(self):
        name = str(self.cfg.plug_roll_target_axis).strip().lower()
        sign = -1.0 if name.startswith("-") else 1.0
        key = name[1:] if sign < 0 else name
        if key in ("y", "lat_y", "port_y"):
            axis = self._lat_y
        else:
            axis = self._lat_x
        return sign * axis

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

    def _plug_roll_axis(self):
        """Projected keyed cross-section axis for SFP roll/orientation checks."""
        idx = int(np.clip(int(self.cfg.plug_roll_axis_index), 0, 2))
        xmat = self.data.xmat[self._plug_roll_id].reshape(3, 3)
        return self._project_to_insert_plane(xmat[:, idx], self._target_roll_axis())

    def _plug_roll_error(self) -> float:
        target = self._target_roll_axis()
        roll = self._plug_roll_axis()
        return float(np.arccos(np.clip(np.dot(roll, target), -1.0, 1.0)))

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

    def _insertion_depth_m(self, tip=None) -> float:
        """Signed tip depth from the port mouth toward the seated frame."""
        tip = self.data.xpos[self._plug_tip_id] if tip is None else np.asarray(tip)
        return float(np.dot(tip - self._port_pos, self._insert_axis))

    def _insertion_depth_norm(self, tip=None) -> float:
        depth = self._insertion_depth_m(tip)
        return float(np.clip(depth / max(self.cfg.seated_depth_m, 1e-6), 0.0, 1.0))

    def _overinsert_m(self, tip=None) -> float:
        return float(max(0.0, self._insertion_depth_m(tip) - self.cfg.seated_depth_m))

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

    def _tip_axis_roll_at_arm(self, q_arm):
        self._rigid_home(q_arm)
        return (
            self.data.xpos[self._plug_tip_id].copy(),
            self._plug_axis(),
            self._plug_roll_axis(),
        )

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
        """Finite-difference reset IK on welded tip position + plug axis + roll.

        The old reset matched the tip position but left the plug attitude mostly
        inherited from the arm home posture. For the SFP cage that can look like
        a diagonal or 90-degree rolled pass through the port. This reset-only
        solve aligns the actual module->tip direction with the port inward axis
        and aligns the keyed plug cross-section in the port plane.
        """
        q = np.array(q_init, dtype=np.float64)
        target_tip = np.asarray(target_tip, dtype=np.float64)
        target_axis = self._unit(target_axis, self._insert_axis)
        target_roll = self._target_roll_axis()
        eps = float(self.cfg.ik_fd_eps)
        axis_weight = float(self.cfg.ik_axis_weight_m)
        roll_weight = (float(self.cfg.ik_roll_weight_m)
                       if self.cfg.align_plug_roll_to_port else 0.0)
        best_q = q.copy()
        best_score = float("inf")
        best_pos = float("inf")
        best_axis = float("inf")
        best_roll = float("inf")

        for _ in range(self.cfg.ik_tip_iters):
            tip, axis, roll_axis = self._tip_axis_roll_at_arm(q)
            pos_err = target_tip - tip
            axis_delta = target_axis - axis
            roll_delta = target_roll - roll_axis
            axis_angle = float(np.arccos(np.clip(np.dot(axis, target_axis), -1.0, 1.0)))
            roll_angle = float(np.arccos(np.clip(np.dot(roll_axis, target_roll), -1.0, 1.0)))
            pos_norm = float(np.linalg.norm(pos_err))
            score = pos_norm + axis_weight * axis_angle + roll_weight * roll_angle
            if score < best_score:
                best_q, best_score = q.copy(), score
                best_pos, best_axis, best_roll = pos_norm, axis_angle, roll_angle
            roll_ok = (not self.cfg.align_plug_roll_to_port
                       or roll_angle < self.cfg.ik_roll_tol_rad)
            if (pos_norm < self.cfg.ik_tol
                    and axis_angle < self.cfg.ik_axis_tol_rad
                    and roll_ok):
                break

            err = np.concatenate([
                pos_err,
                axis_weight * axis_delta,
                roll_weight * roll_delta,
            ])
            J = np.zeros((9, 6), dtype=np.float64)
            for k in range(6):
                q_eps = q.copy()
                q_eps[k] += eps
                tip_eps, axis_eps, roll_eps = self._tip_axis_roll_at_arm(q_eps)
                J[:3, k] = (tip_eps - tip) / eps
                J[3:6, k] = axis_weight * (axis_eps - axis) / eps
                J[6:, k] = roll_weight * (roll_eps - roll_axis) / eps
            dq = J.T @ np.linalg.solve(J @ J.T + self.cfg.ik_tip_damping * np.eye(9), err)
            n = np.linalg.norm(dq)
            if n > self.cfg.ik_tip_step_max:
                dq *= self.cfg.ik_tip_step_max / n
            q = self._clip_arm(q + dq)

        tip, axis, roll_axis = self._tip_axis_roll_at_arm(best_q)
        best_pos = float(np.linalg.norm(target_tip - tip))
        best_axis = float(np.arccos(np.clip(np.dot(axis, target_axis), -1.0, 1.0)))
        best_roll = float(np.arccos(np.clip(np.dot(roll_axis, target_roll), -1.0, 1.0)))
        return best_q, best_pos, best_axis, best_roll

    def _clip_arm(self, q):
        q = np.asarray(q, dtype=np.float64).copy()
        for k, adr in enumerate(self._arm_qadr):
            jid = int(np.flatnonzero(self.model.jnt_qposadr == adr)[0])
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                q[k] = np.clip(q[k], lo, hi)
        return q

    def _sample_start_tcp(self, level, jitter=True, rng=None):
        """Last-inch reverse curriculum in the port frame.

        Retraction depth: sampled from the FRONTIER BAND
        [(level - curriculum_band) * span, level * span] so the start
        distribution actually tracks the level (uniform-over-[0, level]
        sampling kept half the starts trivially easy at every level). A small
        `curriculum_easy_frac` of starts replay the full easy range to guard
        against forgetting.

        Lateral/orientation jitter: two-phase. While the sampled tip is still
        inside the cage (retract < seated_depth_m) the SFP clearance is
        sub-millimetre, so jitter stays at the `*_inport` values; once outside
        the entrance, jitter widens linearly with how far outside the tip is,
        up to the full level-1 values. This is what makes level growth read as
        "seated -> slides out of the port -> approach pose varies in x/y/z".
        """
        rng = rng or np.random
        level = float(np.clip(level, 0.0, 1.0))
        span = float(self.cfg.last_inch_m)
        hi = level * span
        if jitter and hi > 0:
            lo = max(0.0, (level - self.cfg.curriculum_band)) * span
            if rng.uniform() < self.cfg.curriculum_easy_frac:
                retract = rng.uniform(0.0, hi)
            else:
                retract = rng.uniform(lo, hi)
        else:
            retract = hi
        tip = self._inserted_tip + retract * self._retract_dir
        quat = self._goal_quat.copy()
        if jitter and level > 0:
            in_port_travel = min(self.cfg.seated_depth_m, span)
            out_frac = float(np.clip(
                (retract - in_port_travel) / max(span - in_port_travel, 1e-6), 0.0, 1.0))
            xy_amp = (self.cfg.jitter_xy_inport_m
                      + (self.cfg.jitter_xy_m - self.cfg.jitter_xy_inport_m) * out_frac)
            yaw_amp = (self.cfg.jitter_yaw_inport_rad
                       + (self.cfg.jitter_yaw_rad - self.cfg.jitter_yaw_inport_rad) * out_frac)
            tilt_amp = (self.cfg.jitter_tilt_inport_rad
                        + (self.cfg.jitter_tilt_rad - self.cfg.jitter_tilt_inport_rad) * out_frac)
            tip = tip + self._lat_x * rng.uniform(-1, 1) * xy_amp * level
            tip = tip + self._lat_y * rng.uniform(-1, 1) * xy_amp * level
            yaw = rng.uniform(-1, 1) * yaw_amp * level
            tilt_x = rng.uniform(-1, 1) * tilt_amp * level
            tilt_y = rng.uniform(-1, 1) * tilt_amp * level
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
            in_port = (float(np.dot(np.asarray(target_tip) - self._inserted_tip,
                                    self._retract_dir)) < self.cfg.seated_depth_m)
            q_seed = self._ik(tcp, quat, self._home)
            q_seed = self._ik_tcp_position(tcp, q_seed)
            q_arm, ik_tip_err, ik_axis_err, ik_roll_err = self._ik_tip_axis(
                target_tip, self._insert_axis, q_seed)
            self._rigid_home(q_arm)
            for _ in range(n_settle):
                self.data.ctrl[:6] = self._base_torque(q_arm)
                self.data.ctrl[6] = self.cfg.gripper_ctrl
                mujoco.mj_step(self.model, self.data)
            diag = self._geometry_diag(target_tip=target_tip, ik_tip_err=ik_tip_err,
                                       ik_axis_err=ik_axis_err, ik_roll_err=ik_roll_err)
            score = (diag["tip_error_m"]
                     + 0.02 * diag["plug_axis_error_rad"]
                     + 0.02 * diag["plug_roll_error_rad"]
                     + 2.0 * diag["plug_port_penetration_m"]
                     + 0.001 * diag["contact_force_norm"]
                     + (1.0 * diag["lateral_error_m"] if in_port else 0.0))
            if best is None or score < best[0]:
                best = (score, q_arm.copy(), diag)
            # in-cage starts additionally require the SETTLED lateral error to be
            # within the port clearance: the 6 mm IK tip tolerance otherwise seeds
            # wedged starts the +/-0.01 rad/step residual policy cannot recover
            lateral_ok = (not in_port
                          or diag["lateral_error_m"] <= self.cfg.reset_inport_lateral_tol_m)
            roll_ok = (not self.cfg.align_plug_roll_to_port
                       or diag["plug_roll_error_rad"] <= self.cfg.ik_roll_tol_rad)
            if (diag["tip_error_m"] <= self.cfg.reset_ik_tip_tol_m
                    and diag["plug_axis_error_rad"] <= self.cfg.ik_axis_tol_rad
                    and roll_ok
                    and diag["plug_port_penetration_m"] <= self.cfg.reset_max_plug_port_penetration_m
                    and diag["contact_force_norm"] <= self.cfg.reset_contact_abort_n
                    and lateral_ok):
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
        self._cart_init_from_state()

    def _calibrate_pen_baseline(self) -> None:
        """Measure resting plug-port penetration vs depth along the axis.

        The collision cost then charges only the EXCESS over this curve, so
        traversing the cage is free while genuine wedging still registers.
        """
        n = max(2, int(self.cfg.pen_baseline_points))
        depths, pens = [], []
        # use the SAME reset pipeline episodes use (3-stage IK + settle), so the
        # baseline is measured exactly at the states resets actually produce
        for lvl in np.linspace(0.0, 1.0, n):
            self._reset_to_level(float(lvl), jitter=False)
            depths.append(self._depth_norm())
            pens.append(self._resting_penetration())
        order = np.argsort(depths)
        self._pen_baseline_depths = np.asarray(depths, dtype=np.float64)[order]
        self._pen_baseline_vals = np.asarray(pens, dtype=np.float64)[order]

    def _resting_penetration(self) -> float:
        plug_bodies = set(getattr(
            self, "_plug_body_ids",
            {int(self._plug_id), int(self._plug_tip_id), int(self._plug_axis_tail_id)},
        ))
        target_tokens = ("nic_card", "sfp_port", "sfp", "mount")
        worst = 0.0
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            b1 = int(self.model.geom_bodyid[int(c.geom1)])
            b2 = int(self.model.geom_bodyid[int(c.geom2)])
            one_plug = (b1 in plug_bodies) ^ (b2 in plug_bodies)
            other = self._body_name(b2 if b1 in plug_bodies else b1).lower()
            if one_plug and any(t in other for t in target_tokens):
                worst = max(worst, -float(c.dist))
        return worst

    def _pen_baseline_at(self, depth_norm: float) -> float:
        depths = getattr(self, "_pen_baseline_depths", None)
        if depths is None:
            # pre-calibration (during construction): fall back to the seated value
            return float(getattr(self, "_goal_plug_port_penetration_m", 0.0))
        base = float(np.interp(depth_norm, depths, self._pen_baseline_vals))
        return base + float(self.cfg.pen_baseline_margin_m)

    def _base_torque(self, target):
        q = self.data.qpos[self._arm_qadr]; qd = self.data.qvel[self._arm_vadr]
        tau = (self._joint_stiffness * (target - q)
               - self._joint_damping * qd
               + self.data.qfrc_bias[self._arm_vadr])
        return np.clip(tau, -self._joint_tau_limit, self._joint_tau_limit)

    def _clip_action_target(self, target):
        lo = self._reset_arm_target - self.cfg.action_joint_limit
        hi = self._reset_arm_target + self.cfg.action_joint_limit
        return self._clip_arm(np.clip(target, lo, hi))

    # ---------------- cartesian_residual mode ------------------------- #
    def _cart_init_from_state(self):
        """Anchor the Cartesian base/residual targets at the settled reset pose."""
        self._cart_base_pos = self.data.site_xpos[self._tcp_sid].copy()
        self._cart_base_quat = self._site_quat()
        # port frame: columns = [lat_x, lat_y, insert_axis]; fixed per episode
        self._cart_frame_R = np.column_stack(
            [self._lat_x, self._lat_y, self._insert_axis])
        self._cart_resid_pos = np.zeros(3)     # accumulated, port frame
        self._cart_resid_rotvec = np.zeros(3)  # accumulated, port frame
        self._cart_clip_events = 0
        self._cart_wrench_sat_events = 0
        self._cart_tau_sat_events = 0
        self._cart_base_progress_m = 0.0
        self._cart_base_travel_max = float(max(
            0.0, np.dot(self._goal_tcp - self._cart_base_pos, self._insert_axis)))
        self._cart_diag = {}

    @staticmethod
    def _quat_to_rotvec(q):
        q = q if q[0] >= 0.0 else -q   # short way around
        s = float(np.linalg.norm(q[1:]))
        if s < 1e-12:
            return np.zeros(3)
        angle = 2.0 * float(np.arctan2(s, q[0]))
        return (q[1:] / s) * angle

    def _cart_desired_pose(self):
        R = self._cart_frame_R
        des_pos = self._cart_base_pos + R @ self._cart_resid_pos
        rot_world = R @ self._cart_resid_rotvec
        angle = float(np.linalg.norm(rot_world))
        dq = self._axis_angle(rot_world, angle)
        des_quat = self._qmul(dq, self._cart_base_quat)
        return des_pos, des_quat

    def _cart_impedance_torque(self, des_pos, des_quat):
        """Mirror of aic_controller CartesianImpedanceAction in MuJoCo:
        wrench = K*pose_err - D*vel (clamped), tau = J^T wrench + gravity comp.
        """
        cfg = self.cfg
        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self._tcp_sid)
        v = Jp @ self.data.qvel
        w = Jr @ self.data.qvel
        pos_err = np.asarray(des_pos) - self.data.site_xpos[self._tcp_sid]
        q_diff = self._qmul(des_quat, self._qinv(self._site_quat()))
        rot_err = self._quat_to_rotvec(q_diff)   # world frame (matches Jr)
        force = cfg.cart_kp_pos * pos_err - cfg.cart_kd_pos * v
        torque = cfg.cart_kp_rot * rot_err - cfg.cart_kd_rot * w
        wrench = np.concatenate([force, torque])
        wrench_sat = bool(np.any(np.abs(wrench) > self._cart_max_wrench))
        wrench = np.clip(wrench, -self._cart_max_wrench, self._cart_max_wrench)
        J = np.vstack((Jp[:, self._arm_vadr], Jr[:, self._arm_vadr]))
        qd_arm = self.data.qvel[self._arm_vadr]
        # AIC applies its damping-only nullspace controller after the Cartesian
        # wrench. Project it into the joint nullspace so it does not fight the
        # commanded TCP motion.
        nullspace = np.eye(6) - J.T @ np.linalg.pinv(J.T)
        tau = (J.T @ wrench
               + nullspace @ (-cfg.cart_nullspace_damping * qd_arm)
               + self.data.qfrc_bias[self._arm_vadr])
        tau_sat = bool(np.any(np.abs(tau) > self._cart_tau_limit))
        tau = np.clip(tau, -self._cart_tau_limit, self._cart_tau_limit)
        return tau, wrench, wrench_sat, tau_sat, pos_err, rot_err

    def _apply_cartesian_action(self, action):
        """Integrate the policy's Cartesian residual and track it for one
        policy step (control_substeps sim steps) with the impedance controller."""
        cfg = self.cfg
        dpos = action[:3] * cfg.cart_trans_scale_m
        drot = np.zeros(3)
        drot[:2] = action[3:5] * cfg.cart_rot_scale_rad
        if self._action_dim == 6:
            drot[2] = action[5] * cfg.cart_rot_scale_rad
        if cfg.base_script_enabled:
            adv = float(np.clip(cfg.base_script_step_m, 0.0,
                                self._cart_base_travel_max - self._cart_base_progress_m))
            self._cart_base_pos = self._cart_base_pos + adv * self._insert_axis
            self._cart_base_progress_m += adv
            pos_lim = cfg.base_script_residual_limit_m
            rot_lim = cfg.base_script_residual_limit_rad
        else:
            pos_lim = cfg.cart_pos_limit_m
            rot_lim = cfg.cart_rot_limit_rad
        raw_pos = self._cart_resid_pos + dpos
        raw_rot = self._cart_resid_rotvec + drot
        new_pos = np.clip(raw_pos, -pos_lim, pos_lim)
        new_rot = np.clip(raw_rot, -rot_lim, rot_lim)
        clipped = bool(np.any(new_pos != raw_pos) or np.any(new_rot != raw_rot))
        self._cart_clip_events += int(clipped)
        self._cart_resid_pos = new_pos
        self._cart_resid_rotvec = new_rot

        des_pos, des_quat = self._cart_desired_pose()
        tcp_before = self.data.site_xpos[self._tcp_sid].copy()
        wrench_f_max = wrench_t_max = tau_max = 0.0
        for _ in range(cfg.control_substeps):
            tau, wrench, wrench_sat, tau_sat, _, _ = (
                self._cart_impedance_torque(des_pos, des_quat))
            self.data.ctrl[:6] = tau
            self.data.ctrl[6] = cfg.gripper_ctrl
            mujoco.mj_step(self.model, self.data)
            wrench_f_max = max(wrench_f_max, float(np.linalg.norm(wrench[:3])))
            wrench_t_max = max(wrench_t_max, float(np.linalg.norm(wrench[3:])))
            tau_max = max(tau_max, float(np.max(np.abs(tau))))
            self._cart_wrench_sat_events += int(wrench_sat)
            self._cart_tau_sat_events += int(tau_sat)
        tcp_delta = self.data.site_xpos[self._tcp_sid] - tcp_before
        pos_err = np.asarray(des_pos) - self.data.site_xpos[self._tcp_sid]
        rot_err = self._quat_to_rotvec(
            self._qmul(des_quat, self._qinv(self._site_quat())))
        tcp_delta_port = self._cart_frame_R.T @ tcp_delta
        pos_err_port = self._cart_frame_R.T @ pos_err
        self._cart_diag = {
            "cart_cmd_dpos_mm": float(np.linalg.norm(dpos)) * 1e3,
            "cart_cmd_dx_mm": float(dpos[0]) * 1e3,
            "cart_cmd_dy_mm": float(dpos[1]) * 1e3,
            "cart_cmd_dz_mm": float(dpos[2]) * 1e3,
            "cart_cmd_drot_deg": float(np.degrees(np.linalg.norm(drot))),
            "cart_cmd_droll_deg": float(np.degrees(drot[0])),
            "cart_cmd_dpitch_deg": float(np.degrees(drot[1])),
            "cart_cmd_dyaw_deg": float(np.degrees(drot[2])),
            "cart_resid_x_mm": float(self._cart_resid_pos[0]) * 1e3,
            "cart_resid_y_mm": float(self._cart_resid_pos[1]) * 1e3,
            "cart_resid_z_mm": float(self._cart_resid_pos[2]) * 1e3,
            "cart_resid_roll_deg": float(np.degrees(self._cart_resid_rotvec[0])),
            "cart_resid_pitch_deg": float(np.degrees(self._cart_resid_rotvec[1])),
            "cart_resid_yaw_deg": float(np.degrees(self._cart_resid_rotvec[2])),
            "cart_tcp_dpos_mm": float(np.linalg.norm(tcp_delta)) * 1e3,
            "cart_tcp_dx_mm": float(tcp_delta_port[0]) * 1e3,
            "cart_tcp_dy_mm": float(tcp_delta_port[1]) * 1e3,
            "cart_tcp_dz_mm": float(tcp_delta_port[2]) * 1e3,
            "cart_pos_err_mm": float(np.linalg.norm(pos_err)) * 1e3,
            "cart_pos_err_x_mm": float(pos_err_port[0]) * 1e3,
            "cart_pos_err_y_mm": float(pos_err_port[1]) * 1e3,
            "cart_pos_err_z_mm": float(pos_err_port[2]) * 1e3,
            "cart_rot_err_deg": float(np.degrees(np.linalg.norm(rot_err))),
            "cart_wrench_force_n": wrench_f_max,
            "cart_wrench_torque_nm": wrench_t_max,
            "cart_tau_max_nm": tau_max,
            "cart_clip_events": int(self._cart_clip_events),
            "cart_wrench_sat_events": int(self._cart_wrench_sat_events),
            "cart_tau_sat_events": int(self._cart_tau_sat_events),
            "cart_base_progress_mm": float(self._cart_base_progress_m) * 1e3,
        }

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
        jitter = True
        if options:   # optional deterministic overrides (used by smoke tests)
            level = float(options.get("level", level))
            jitter = bool(options.get("jitter", jitter))
        self._reset_to_level(level, jitter=jitter)
        self._step_count = 0
        self._last_action = np.zeros(self._action_dim, np.float32)
        self._prev_depth_norm = self._depth_norm()
        self._f_ax_buf: list[float] = []
        self._off_limit_event_fired = False
        self._force_sustain_event_fired = False
        self._force_over_count = 0
        self._reset_score_state()
        return self._obs(), {
            "curriculum_level": self._curriculum_level,
            "reset_diag": getattr(self, "_last_reset_diag", None),
        }

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64).reshape(self._action_dim), -1.0, 1.0)
        if self._cart_mode:
            self._apply_cartesian_action(action)
        else:
            self._arm_target = self._clip_action_target(
                self._arm_target + action * self.cfg.action_joint_scale)
            for _ in range(self.cfg.control_substeps):
                self.data.ctrl[:6] = self._base_torque(self._arm_target)
                self.data.ctrl[6] = self.cfg.gripper_ctrl
                mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._update_score_path()
        obs = self._obs()
        total, breakdown, term_status = self._reward_and_term(obs, action)
        self._last_action = action.astype(np.float32)
        # The post-step observation consumed by the next policy call must carry
        # the action that produced it. Reward computation above still sees the
        # prior action for the action-delta penalty.
        obs["last_action"] = self._last_action.copy()
        terminated = term_status in ("success", "force_abort", "bad_collision", "off_limit")
        truncated = (term_status == "timeout") or (self._step_count >= self.cfg.max_episode_steps)
        score_info = self._score_diag(term_status)
        info = {
            "term_status": term_status if term_status else ("timeout" if truncated else None),
            "breakdown": breakdown,
            "image_l1_norm": breakdown.image_l1_norm,
            "f_z": float(self._f_axial),
            "f_z_mean": float(np.mean(self._f_ax_buf)) if self._f_ax_buf else float("nan"),
            "f_z_max": float(np.max(np.abs(self._f_ax_buf))) if self._f_ax_buf else float("nan"),
            "depth_norm": float(self._prev_depth_norm),
            "insertion_depth_m": float(self._insertion_depth_m()),
            "approach_gap_m": float(max(
                0.0,
                float(getattr(self, "_axial_error", 0.0)) - self.cfg.seated_depth_m,
            )),
            "overinsert_m": float(self._overinsert_m()),
            "axial_error_m": float(getattr(self, "_axial_error", float("nan"))),
            "lateral_error_m": float(getattr(self, "_lateral_error", float("nan"))),
            **{k: v for k, v in self._contact_summary().items()
               if not isinstance(v, str)},
            "plug_axis_error_rad": float(self._plug_axis_error()),
            "plug_roll_error_rad": float(self._plug_roll_error()),
            "wallclock": float(self._step_count) * self._policy_dt_s,
            "physics_dt_s": self._physics_dt_s,
            "policy_dt_s": self._policy_dt_s,
            "policy_hz": 1.0 / self._policy_dt_s,
            "curriculum_level": float(self._curriculum_level),
            "action_mode": self.cfg.action_mode,
            **score_info,
        }
        if self._cart_mode:
            info.update(self._cart_diag)
        return obs, total, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _contact_force(self):
        """World-frame contact force on the plug = FT minus the gripped-plug baseline."""
        return self._raw_ft()[:3] - self._ft_baseline

    def _body_name(self, bid: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{bid}"

    def _geom_name(self, gid: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"

    def _body_subtree_ids(self, root_bid: int) -> set[int]:
        if root_bid < 0:
            return set()
        root_bid = int(root_bid)
        ids: set[int] = set()
        stack = [root_bid]
        while stack:
            bid = int(stack.pop())
            ids.add(bid)
            children = np.flatnonzero(self.model.body_parentid == bid)
            stack.extend(int(child) for child in children if int(child) not in ids)
        return ids

    def _find_manipulated_body_ids(self) -> set[int]:
        ids = set(getattr(self, "_plug_body_ids", set()))
        ids.update(int(i) for i in (self._plug_id, self._plug_tip_id, self._plug_axis_tail_id)
                   if int(i) >= 0)
        # The full cable subtree also contains the far SC plug/cable, which is
        # already close to the enclosure in the exported scene. Off-limit events
        # here should describe the gripped plug assembly that the policy moves.
        tokens = ("lc_plug", "sfp_module", "sfp_tip")
        for bid in range(self.model.nbody):
            name = self._body_name(bid).lower()
            if any(tok in name for tok in tokens):
                ids.add(int(bid))
        return {i for i in ids if i >= 0}

    def _contact_summary(self):
        plug_bodies = set(getattr(
            self, "_plug_body_ids",
            {int(self._plug_id), int(self._plug_tip_id), int(self._plug_axis_tail_id)},
        ))
        target_tokens = ("nic_card", "sfp_port", "sfp", "mount")
        stop_tokens = ("backstop",)
        off_limit_tokens = ("enclosure", "task_board")
        manipulated_bodies = set(getattr(self, "_manipulated_body_ids", plug_bodies))
        min_dist = float("inf")
        plug_port_min = float("inf")
        port_stop_min = float("inf")
        off_limit_min = float("inf")
        worst_pair = ""
        plug_port_worst = ""
        plug_port_contacts = 0
        port_stop_contacts = 0
        off_limit_contacts = 0
        off_limit_worst = ""
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
            other_name = (n2 if b1 in plug_bodies else n1).lower()
            is_portish = any(tok in other_name for tok in target_tokens)
            if one_plug and is_portish:
                plug_port_contacts += 1
                if dist < plug_port_min:
                    plug_port_min = dist
                    plug_port_worst = f"{n1}:{ge1} <-> {n2}:{ge2}"
            other_geom_name = (ge2 if b1 in plug_bodies else ge1).lower()
            if one_plug and any(tok in other_geom_name for tok in stop_tokens):
                port_stop_contacts += 1
                if dist < port_stop_min:
                    port_stop_min = dist

            one_manipulated = (b1 in manipulated_bodies) ^ (b2 in manipulated_bodies)
            other_manip_name = (n2 if b1 in manipulated_bodies else n1).lower()
            if one_manipulated and any(tok in other_manip_name for tok in off_limit_tokens):
                off_limit_contacts += 1
                if dist < off_limit_min:
                    off_limit_min = dist
                    off_limit_worst = f"{n1}:{ge1} <-> {n2}:{ge2}"

        if not np.isfinite(min_dist):
            min_dist = 0.0
        if not np.isfinite(plug_port_min):
            plug_port_min = 0.0
        if not np.isfinite(port_stop_min):
            port_stop_min = 0.0
        plug_port_pen = float(max(0.0, -plug_port_min))
        port_stop_pen = float(max(0.0, -port_stop_min))
        goal_pen = float(self._pen_baseline_at(self._depth_norm()))
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
            "port_stop_contacts": int(port_stop_contacts),
            "port_stop_penetration_m": port_stop_pen,
            "off_limit_contacts": int(off_limit_contacts),
            "off_limit_worst_pair": off_limit_worst,
        }

    def _depth_norm(self) -> float:
        """Insertion fraction in [0,1]: 0 at mouth, 1 at the seated frame."""
        return self._insertion_depth_norm()

    def _reset_score_state(self) -> None:
        tcp = self.data.site_xpos[self._tcp_sid].copy()
        tip = self.data.xpos[self._plug_tip_id].copy()
        self._score_last_tcp = tcp
        self._score_tcp_hist = [tcp.copy()]
        self._score_path_length_m = 0.0
        self._score_force_over_s = 0.0
        self._score_initial_insert_distance_m = float(
            max(np.linalg.norm(tip - self._inserted_tip), 1e-3))
        self._score_initial_entrance_distance_m = float(
            max(np.linalg.norm(tip - self._port_pos), 1e-3))

    def _update_score_path(self) -> None:
        tcp = self.data.site_xpos[self._tcp_sid].copy()
        last = getattr(self, "_score_last_tcp", tcp)
        self._score_path_length_m = float(
            getattr(self, "_score_path_length_m", 0.0) + np.linalg.norm(tcp - last))
        self._score_last_tcp = tcp
        hist = getattr(self, "_score_tcp_hist", [])
        hist.append(tcp.copy())
        self._score_tcp_hist = hist[-64:]

    def _score_smoothness_points(self, dt: float) -> tuple[float, float]:
        hist = np.asarray(getattr(self, "_score_tcp_hist", []), dtype=np.float64)
        if hist.shape[0] < 5:
            return 6.0, 0.0
        vel = np.diff(hist, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        if jerk.size == 0:
            return 6.0, 0.0
        speed = np.linalg.norm(vel[2:], axis=1)
        moving = speed > 0.01
        jerk_mag = np.linalg.norm(jerk, axis=1)
        if np.any(moving):
            jerk_mean = float(np.mean(jerk_mag[moving]))
        else:
            jerk_mean = 0.0
        points = 6.0 * (1.0 - np.clip(jerk_mean / 50.0, 0.0, 1.0))
        return float(points), jerk_mean

    def _score_diag(self, term_status: Optional[str]) -> dict[str, float | int | bool]:
        """Local estimate of the published score rubric for diagnostics.

        This is intentionally stricter than the dense reward terminal flag. It
        approximates the official evaluator with geometry/contact signals we can
        observe inside this single-ended last-inch environment.
        """
        dt = self._policy_dt_s
        duration_s = float(self._step_count) * dt
        fc = self._contact_force()
        force_norm = float(np.linalg.norm(fc))
        if force_norm > self.cfg.score_force_threshold_n:
            self._score_force_over_s = float(getattr(self, "_score_force_over_s", 0.0) + dt)

        tip = self.data.xpos[self._plug_tip_id]
        axial_err, lat_vec = self._tip_port_errors(tip)
        lateral_err = float(np.linalg.norm(lat_vec))
        insertion_depth_m = self._insertion_depth_m(tip)
        depth_norm = self._depth_norm()
        overinsert_m = self._overinsert_m(tip)
        axis_err = self._plug_axis_error()
        roll_err = self._plug_roll_error()
        contact_summary = self._contact_summary()
        pen_excess = float(contact_summary["plug_port_penetration_excess_m"])
        off_limit_contacts = int(contact_summary.get("off_limit_contacts", 0))
        port_contact_ok = int(contact_summary.get("plug_port_contacts", 0)) > 0

        in_port_box = (
            -self.cfg.score_axial_tol_m <= insertion_depth_m <= (
                self.cfg.seated_depth_m + self.cfg.success_max_overinsert_m)
            and lateral_err <= self.cfg.score_lateral_tol_m
        )
        full_insert = (
            in_port_box
            and port_contact_ok
            and depth_norm >= self.cfg.score_full_depth_norm
            and abs(axial_err) <= self.cfg.score_axial_tol_m
            and overinsert_m <= self.cfg.success_max_overinsert_m
            and axis_err <= self.cfg.score_axis_tol_rad
            and roll_err <= self.cfg.score_roll_tol_rad
            and pen_excess <= self.cfg.score_max_penetration_excess_m
        )
        if full_insert:
            tier3 = 75.0
            partial = False
            proximity = 25.0
        elif in_port_box:
            partial = True
            tier3 = 38.0 + 12.0 * float(np.clip(depth_norm, 0.0, 1.0))
            proximity = 25.0
        else:
            partial = False
            dist_to_entrance = float(np.linalg.norm(tip - self._port_pos))
            max_dist = max(float(getattr(self, "_score_initial_entrance_distance_m", 1e-3)) * 0.5, 1e-3)
            proximity = 25.0 * (1.0 - np.clip(dist_to_entrance / max_dist, 0.0, 1.0))
            tier3 = float(proximity)

        eligible_tier2 = tier3 > 0.0
        if eligible_tier2:
            duration_points = 12.0 * (1.0 - np.clip((duration_s - 5.0) / 55.0, 0.0, 1.0))
            path_len = float(getattr(self, "_score_path_length_m", 0.0))
            initial = max(float(getattr(self, "_score_initial_insert_distance_m", 1e-3)), 1e-3)
            efficiency_points = 6.0 * (1.0 - np.clip((path_len - initial) / 1.0, 0.0, 1.0))
            smooth_points, jerk_mean = self._score_smoothness_points(dt)
        else:
            duration_points = 0.0
            efficiency_points = 0.0
            smooth_points, jerk_mean = 0.0, 0.0
            path_len = float(getattr(self, "_score_path_length_m", 0.0))

        force_penalty = -12.0 if (
            float(getattr(self, "_score_force_over_s", 0.0)) > self.cfg.score_force_duration_s
        ) else 0.0
        off_limit_penalty = self.cfg.score_off_limit_penalty if off_limit_contacts > 0 else 0.0
        tier2 = duration_points + efficiency_points + smooth_points + force_penalty + off_limit_penalty
        total = 1.0 + tier2 + tier3
        return {
            "score_success": bool(full_insert),
            "score_partial": bool(partial),
            "score_tier3": float(tier3),
            "score_tier2": float(tier2),
            "score_total": float(total),
            "score_duration_points": float(duration_points),
            "score_efficiency_points": float(efficiency_points),
            "score_smoothness_points": float(smooth_points),
            "score_force_penalty": float(force_penalty),
            "score_off_limit_penalty": float(off_limit_penalty),
            "score_proximity_points": float(proximity),
            "score_duration_s": float(duration_s),
            "score_path_length_m": float(path_len),
            "score_force_over_s": float(getattr(self, "_score_force_over_s", 0.0)),
            "score_jerk_mps3": float(jerk_mean),
            "score_in_port_box": bool(in_port_box),
            "score_force_over_limit": bool(force_penalty < 0.0),
            "score_off_limit_contact": bool(off_limit_contacts > 0),
            "score_term_success": bool(term_status == "success"),
            "score_roll_error_rad": float(roll_err),
        }

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
        insertion_depth_m = self._insertion_depth_m(tip)
        depth_norm = self._depth_norm()
        overinsert_m = self._overinsert_m(tip)
        contact_summary = self._contact_summary()
        axis_error = self._plug_axis_error()
        roll_error = self._plug_roll_error()
        pen_excess = float(contact_summary["plug_port_penetration_excess_m"])
        port_contact_ok = (
            not self.cfg.success_require_port_contact
            or int(contact_summary.get("plug_port_contacts", 0)) > 0
        )

        # success = bottomed depth + seated pose + straight, collision-clean seating contact
        term_status = None
        if (depth_norm >= self.cfg.success_depth_norm
                and insertion_depth_m >= self.cfg.seated_depth_m - self.cfg.success_axial_tol_m
                and abs(axial_err) < self.cfg.success_axial_tol_m
                and lat_err < self.cfg.success_lateral_tol_m
                and axis_error <= self.cfg.success_axis_tol_rad
                and roll_error <= self.cfg.success_roll_tol_rad
                and overinsert_m <= self.cfg.success_max_overinsert_m
                and pen_excess <= self.cfg.success_max_plug_port_penetration_excess_m
                and port_contact_ok
                and (self.cfg.success_force_n <= 0.0
                     or abs(f_axial) > self.cfg.success_force_n)):
            term_status = "success"
        elif (pen_excess > self.cfg.bad_collision_penetration_excess_m
              or overinsert_m > self.cfg.bad_collision_overinsert_m
              or (depth_norm >= self.cfg.bad_collision_depth_gate
                  and (axis_error > self.cfg.bad_collision_axis_rad
                       or roll_error > self.cfg.bad_collision_roll_rad))):
            term_status = "bad_collision"
        else:
            # force abort: sustained (dwell) over force_abort_n, or instant only
            # past the hard bound — single-step PD contact transients don't kill
            # the episode (they were 43% of level-0.1 terminations)
            f_norm = float(np.linalg.norm(fc))
            f_peak = max(abs(f_axial), f_lat, f_norm)
            if f_peak > self.cfg.force_abort_n:
                self._force_over_count = int(getattr(self, "_force_over_count", 0)) + 1
            else:
                self._force_over_count = 0
            if (f_peak > self.cfg.force_abort_hard_n
                    or self._force_over_count >= max(1, int(self.cfg.force_abort_dwell_steps))):
                term_status = "force_abort"
        if term_status is None and self._step_count >= self.cfg.max_episode_steps:
            term_status = "timeout"

        # scoring-rubric one-shot events (each fires at most once per episode,
        # mirroring the one-time -24 / -12 point deductions; neither is terminal)
        off_limit_event = False
        if (int(contact_summary.get("off_limit_contacts", 0)) > 0
                and not self._off_limit_event_fired):
            self._off_limit_event_fired = True
            off_limit_event = True
        force_sustain_event = False
        if (float(getattr(self, "_score_force_over_s", 0.0)) > self.cfg.score_force_duration_s
                and not self._force_sustain_event_fired):
            self._force_sustain_event_fired = True
            force_sustain_event = True

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
            roll_error_rad=roll_error,
            penetration_excess_m=max(pen_excess, overinsert_m),
            off_limit_event=off_limit_event,
            force_sustain_event=force_sustain_event,
            success_time_frac=float(self._step_count) / max(self.cfg.max_episode_steps, 1),
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

    def _geometry_diag(self, target_tip=None, ik_tip_err=None, ik_axis_err=None,
                       ik_roll_err=None):
        tip = self.data.xpos[self._plug_tip_id].copy()
        target = self._inserted_tip if target_tip is None else np.asarray(target_tip, dtype=np.float64)
        axial_err, lat_vec = self._tip_port_errors(tip)
        approach_gap = max(0.0, float(axial_err) - self.cfg.seated_depth_m)
        fc = self._contact_force()
        f_axial = float(np.dot(fc, self._insert_axis))
        f_lat = float(np.linalg.norm(fc - f_axial * self._insert_axis))
        axis_err = self._plug_axis_error()
        roll_err = self._plug_roll_error()
        out = {
            "target_body": self._target_name,
            "entrance": self._port_pos.copy(),
            "seated_tip": self._inserted_tip.copy(),
            "insert_axis": self._insert_axis.copy(),
            "plug_axis": self._plug_axis().copy(),
            "target_roll_axis": self._target_roll_axis().copy(),
            "plug_roll_axis": self._plug_roll_axis().copy(),
            "tip": tip,
            "target_tip": target.copy(),
            "tip_error_m": float(np.linalg.norm(tip - target)),
            "ik_tip_error_m": float(ik_tip_err if ik_tip_err is not None else np.linalg.norm(tip - target)),
            "plug_axis_error_rad": axis_err,
            "plug_axis_error_deg": float(np.degrees(axis_err)),
            "ik_axis_error_rad": float(ik_axis_err if ik_axis_err is not None else axis_err),
            "plug_roll_error_rad": roll_err,
            "plug_roll_error_deg": float(np.degrees(roll_err)),
            "ik_roll_error_rad": float(ik_roll_err if ik_roll_err is not None else roll_err),
            "axial_error_m": float(axial_err),
            "approach_gap_m": float(approach_gap),
            "lateral_error_m": float(np.linalg.norm(lat_vec)),
            "depth_norm": float(self._depth_norm()),
            "insertion_depth_m": float(self._insertion_depth_m(tip)),
            "overinsert_m": float(self._overinsert_m(tip)),
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

    def render_camera(self, camera: str = "center_camera",
                      height: Optional[int] = None,
                      width: Optional[int] = None):
        """Render a diagnostic camera without changing policy observation size."""
        h = int(height or self.cfg.image_h)
        w = int(width or self.cfg.image_w)
        h += h % 2
        w += w % 2
        if self._renderer is not None and h == self.cfg.image_h and w == self.cfg.image_w:
            renderer = self._renderer
        else:
            key = (h, w)
            renderer = self._debug_renderers.get(key)
            if renderer is None:
                renderer = mujoco.Renderer(self.model, height=h, width=w)
                self._debug_renderers[key] = renderer
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()[:, :, :3]

    def close(self):
        renderers = [getattr(self, "_renderer", None), getattr(self, "_reward_renderer", None)]
        renderers.extend(getattr(self, "_debug_renderers", {}).values())
        for r in renderers:
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
        self._renderer = None
        self._reward_renderer = None
        self._debug_renderers = {}


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
          "goal_plug_roll_err_deg:", round(float(np.degrees(env._plug_roll_error())), 2),
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
              f"roll_deg={diag['plug_roll_error_deg']:.1f} "
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
              f"roll_deg={np.degrees(i_hold['plug_roll_error_rad']):.1f} "
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
              f"roll_deg={np.degrees(i['plug_roll_error_rad']):.1f} "
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
