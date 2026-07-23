"""Contract-A deployable-STUDENT env: EXACT 69-dim flat-state obs (Phase-3).

This mirrors the REAL Intrinsic AIC deploy contract read from
``aic_example_policies/ros/PerceptionInsert.py`` (source of truth):

    * ``_build_final_insert_observation``  -> the 69-dim flat obs vector
    * ``_apply_final_insert_action``       -> ``bounded_tip_pose_delta`` w/ per-axis
                                              POS_SCALE=[1.5,1.5,3.5]mm,
                                              ROT_SCALE=[.08,.08,.12]rad
    * ``_validate_final_insert_policy_contract`` -> obs_len 69, 6-D action

Unlike ``student_env.py`` (image + small vec dict for a CNN student), this student
is STATE-ONLY: no camera, no renderer -> light CPU env (the deploy final-insert
policy is likewise a pure MLP over the 69-vector). The wrapper reads the sim's
real quantities and assembles them into the deploy's EXACT field order / signs /
scales.

=========================================================================
EXACT 69-DIM CONTRACT-A OBSERVATION LAYOUT  (index -> field -> source/scale)
=========================================================================
Port frame basis R_port = columns [lat_x, lat_y, insert_axis]; q_port = quat(R_port)
PERCEIVED (GT + per-episode perception bias) for the port position AND orientation.
tip = plug-tip body world pose (xpos/xquat); X = perceived port entrance position.

  idx      field                 source (all wxyz quats, port frame = perceived)
  ------   -------------------   ----------------------------------------------
  0:6      joint_pos             scene.data.qpos[arm] - AIC_HOME (near-home offset)
  6:12     joint_vel             scene.data.qvel[arm]                       [rad/s]
  12:15    gripper_xyz           scene tcp-site world position (m)
  15:19    q_gripper (wxyz)      scene tcp-site world quaternion
  19:22    tcp_vel linear        R_port_perc.T @ (Jp @ qvel)   [m/s, port frame]
  22:25    tcp_vel angular       R_port_perc.T @ (Jr @ qvel)   [rad/s, port frame]
  25:28    port X (perceived)    scene._port_pos + pos_bias   (perception noise)  (m)
  28:32    q_port (perceived)    quat(R_port) rotated by rot_bias (perception noise)
  32:35    delta_port            R_port_perc.T @ (tip_xyz - X_perc)  [m, port frame]
  35:38    rot_err_port          axisangle( q_port_perc^-1 * q_tip ) [rad]
  38:44    hint                  [clip(-6 dx,-6 dy,-8 dz, ..), clip(-3*rot_err,..)]
  44:50    scripted_hint         == hint (deploy sets scripted_hint = hint.copy())
  50:51    [1.0]                 constant bias term
  51:57    wrench                configurable; Gazebo-v1 default is six zeros
  57:63    last_action           previous 6-D action (DEPLOY convention, [-1,1])
  63:69    tip_axes_port         [R_rel @ x_hat, R_rel @ z_hat]  (R_rel = mat(q_rel))
                                                                         total = 69
=========================================================================

The student sees only the *noised* perceived port pose. The plug-tip observation
is reconstructed from robot TCP plus the calibrated fixed transform, with an
optional per-episode hidden grasp perturbation; MuJoCo GT is reserved for the
teacher label. delta_port / rot_err_port
/ hint / tcp_vel all use the PERCEIVED (noised) port frame, so the perception error
propagates through the whole obs exactly as it does on the real robot (where X and
q_port come from the same YOLO/perception estimate).

ACTION CONVENTION
-----------------
Deploy scales the network's tanh output per-axis: POS_SCALE=[1.5,1.5,3.5]mm,
ROT_SCALE=[.08,.08,.12]rad (``_apply_final_insert_action``). Our teacher was trained
in the sim ``cartesian_residual`` env at a uniform 1mm / 1deg scale. To keep the
STUDENT speaking EXACTLY the deploy convention while the underlying physics stays
identical to the teacher's env, the wrapper converts between conventions with a
PHYSICAL-MOTION-PRESERVING rescale (see sim<->deploy helpers). Because the deploy
per-axis scale is LARGER than the sim scale, deploy-convention targets shrink into
[-0.67, 0.67] (pos) / [-0.22, 0.22] (rot) and never clip. Pass
``action_convention="sim"`` to disable the rescale (raw passthrough == image student).
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import gymnasium as gym
import mujoco

try:
    from aic_model.rl_insert_contract import (
        DEPLOY_POS_SCALE,
        DEPLOY_ROT_SCALE,
        HOME_QPOS as DEPLOY_HOME,
        build_observation69,
        canonicalize_quaternion,
        sfp_tip_pose_from_tcp,
    )
except ImportError:  # source-tree import before the ROS package is rebuilt
    _contract_path = (
        Path(__file__).resolve().parents[2]
        / "aic_model" / "aic_model" / "rl_insert_contract.py"
    )
    _spec = importlib.util.spec_from_file_location(
        "_aic_rl_insert_contract_source", _contract_path
    )
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load deploy contract from {_contract_path}")
    _contract = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_contract)
    DEPLOY_POS_SCALE = _contract.DEPLOY_POS_SCALE
    DEPLOY_ROT_SCALE = _contract.DEPLOY_ROT_SCALE
    DEPLOY_HOME = _contract.HOME_QPOS
    build_observation69 = _contract.build_observation69
    canonicalize_quaternion = _contract.canonicalize_quaternion
    sfp_tip_pose_from_tcp = _contract.sfp_tip_pose_from_tcp

# --------------------------------------------------------------------------- #
# deploy contract constants (verified against PerceptionInsert.py, lines 69-72)
# --------------------------------------------------------------------------- #
FINAL_INSERT_OBS_LEN = 69
FINAL_INSERT_ACTION_DIM = 6
# sim cartesian_residual scale (matches teacher / residual_env.py)
SIM_TRANS_SCALE_M = 1.0e-3
SIM_ROT_SCALE_RAD = float(np.radians(1.0))

# --------------------------------------------------------------------------- #
# perception-noise model (base 1-sigma; scaled by `perception_noise`)
# ~1.5mm lateral / 3.5mm axial position, ~1 / 1 / 1.5 deg orientation.
# --------------------------------------------------------------------------- #
PERCEPTION_POS_SIGMA = np.array([1.5e-3, 1.5e-3, 3.5e-3], dtype=np.float64)          # m
PERCEPTION_ROT_SIGMA = np.array([1.745e-2, 1.745e-2, 2.618e-2], dtype=np.float64)   # rad
# Hidden grasp/calibration error observed between the nominal TCP->tip transform
# and Gazebo scoring GT.  Drawn once per episode in the TCP frame.
GRASP_POS_SIGMA = np.array([2.0e-3, 2.0e-3, 4.0e-3], dtype=np.float64)
GRASP_ROT_SIGMA = np.array([2.618e-2, 2.618e-2, 2.618e-2], dtype=np.float64)


# --------------------------------------------------------------------------- #
# action-convention conversion (physical-motion preserving)
# --------------------------------------------------------------------------- #
def deploy_to_sim_action(a_deploy: np.ndarray) -> np.ndarray:
    """deploy-convention [-1,1]^6 -> sim cartesian_residual action (clip [-1,1]).

    physical motion = a_deploy * DEPLOY_SCALE ; sim action = motion / SIM_SCALE.
    """
    a = np.asarray(a_deploy, dtype=np.float64).reshape(6)
    pos = a[:3] * (DEPLOY_POS_SCALE / SIM_TRANS_SCALE_M)
    rot = a[3:6] * (DEPLOY_ROT_SCALE / SIM_ROT_SCALE_RAD)
    return np.clip(np.concatenate([pos, rot]), -1.0, 1.0).astype(np.float32)


def sim_to_deploy_action(a_sim: np.ndarray) -> np.ndarray:
    """sim cartesian_residual action -> deploy-convention [-1,1]^6 (BC target)."""
    a = np.asarray(a_sim, dtype=np.float64).reshape(6)
    pos = a[:3] * (SIM_TRANS_SCALE_M / DEPLOY_POS_SCALE)
    rot = a[3:6] * (SIM_ROT_SCALE_RAD / DEPLOY_ROT_SCALE)
    return np.clip(np.concatenate([pos, rot]), -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# tiny quaternion helpers (mujoco-backed, wxyz like scene_env)
# --------------------------------------------------------------------------- #
def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64).reshape(4))
    return R.reshape(3, 3)


def _axisangle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    ang = float(np.linalg.norm(rotvec))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    if ang > 1e-9:
        mujoco.mju_axisAngle2Quat(q, rotvec / ang, ang)
    return q


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.zeros(4)
    mujoco.mju_mulQuat(r, a, b)
    return r


def _qconj(a: np.ndarray) -> np.ndarray:
    r = np.zeros(4)
    mujoco.mju_negQuat(r, np.asarray(a, dtype=np.float64))   # conjugate == inverse (unit q)
    return r


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q if q[0] >= 0.0 else -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(s, q[0]))
    return (q[1:] / s) * angle


# --------------------------------------------------------------------------- #
# wrapper
# --------------------------------------------------------------------------- #
class StudentObsWrapperA(gym.Wrapper):
    """Emit the EXACT 69-dim Contract-A obs; 6-D cartesian residual action."""

    def __init__(self, env: gym.Env, perception_noise: float = 1.0,
                 action_convention: str = "deploy", wrench_mode: str = "zero",
                 grasp_noise: float = 1.0,
                 seed: int | None = None):
        super().__init__(env)
        self.scene = env.unwrapped                 # SceneInsertEnv (GT, read-only)
        if action_convention not in ("deploy", "sim"):
            raise ValueError("action_convention must be 'deploy' or 'sim'")
        self.action_convention = action_convention
        if wrench_mode not in ("raw", "zero", "baseline"):
            raise ValueError("wrench_mode must be raw, zero, or baseline")
        self.wrench_mode = wrench_mode

        self.action_space = gym.spaces.Box(-1.0, 1.0, (FINAL_INSERT_ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (FINAL_INSERT_OBS_LEN,), np.float32)

        self.perception_noise = float(perception_noise)
        self.grasp_noise = float(grasp_noise)
        self._rng = np.random.default_rng(seed)
        self._pos_bias = np.zeros(3)
        self._rot_bias = np.zeros(3)
        self._grasp_pos_bias = np.zeros(3)
        self._grasp_rot_bias = np.zeros(3)
        self._last_deploy_action = np.zeros(FINAL_INSERT_ACTION_DIM, np.float32)
        self._wrench_baseline = np.zeros(6, dtype=np.float64)
        self._home = np.asarray(getattr(self.scene, "_home", DEPLOY_HOME),
                                dtype=np.float64)[:6]

        # jacobian scratch
        self._Jp = np.zeros((3, self.scene.model.nv))
        self._Jr = np.zeros((3, self.scene.model.nv))

    # -- perception --------------------------------------------------------- #
    def _draw_bias(self):
        s = self.perception_noise
        self._pos_bias = self._rng.normal(0.0, PERCEPTION_POS_SIGMA * s)
        self._rot_bias = self._rng.normal(0.0, PERCEPTION_ROT_SIGMA * s)
        g = self.grasp_noise
        self._grasp_pos_bias = self._rng.normal(0.0, GRASP_POS_SIGMA * g)
        self._grasp_rot_bias = self._rng.normal(0.0, GRASP_ROT_SIGMA * g)

    def _perceived_port(self):
        """Perceived (GT + per-episode bias) port entrance pos + orientation quat."""
        X = self.scene._port_pos.astype(np.float64) + self._pos_bias
        q_gt = _mat_to_quat(self.scene._cart_frame_R)
        q = _qmul(_axisangle_to_quat(self._rot_bias), q_gt)
        return X, canonicalize_quaternion(q)

    # -- 69-dim obs assembly ------------------------------------------------ #
    def _build_obs69(self, raw: dict) -> np.ndarray:
        d = self.scene.data
        model = self.scene.model

        # [0:6] joint pos minus home, [6:12] joint vel
        joint_vel = np.asarray(raw["arm_qvel"], dtype=np.float64)[:6]

        # [12:19] gripper (tcp) world pose
        tcp_pose = np.asarray(raw["tcp_pose"], dtype=np.float64)
        gripper_xyz = tcp_pose[:3]
        q_gripper = tcp_pose[3:7]

        # perceived port frame
        X, q_port = self._perceived_port()

        # [19:25] tcp velocity in perceived port frame (world twist -> port frame)
        mujoco.mj_jacSite(model, d, self._Jp, self._Jr, self.scene._tcp_sid)
        v_world = self._Jp @ d.qvel
        w_world = self._Jr @ d.qvel

        # Reconstruct the plug tip from the measurable TCP using the exact same
        # calibrated transform as Gazebo deployment.  GT tip pose remains
        # available only through priv_obs() for teacher labels.
        tip_xyz, R_tip = sfp_tip_pose_from_tcp(gripper_xyz, q_gripper)
        R_tcp = _quat_to_mat(q_gripper)
        tip_xyz = tip_xyz + R_tcp @ self._grasp_pos_bias
        R_tip = (
            R_tcp @ _quat_to_mat(_axisangle_to_quat(self._grasp_rot_bias))
            @ R_tcp.T @ R_tip
        )

        return build_observation69(
            joint_pos=np.asarray(raw["arm_qpos"], dtype=np.float64)[:6],
            joint_vel=joint_vel,
            tcp_pos=gripper_xyz,
            tcp_quat=q_gripper,
            tcp_linear_velocity_world=v_world,
            tcp_angular_velocity_world=w_world,
            port_pos=X,
            port_quat=q_port,
            tip_pos=tip_xyz,
            tip_rotation=R_tip,
            wrench=np.asarray(raw["ft"], dtype=np.float64)[:6],
            last_action=self._last_deploy_action,
            home_qpos=self._home,
            wrench_mode=self.wrench_mode,
            wrench_baseline=self._wrench_baseline,
        )

    # -- privileged teacher obs (distillation only; read-only GT) ----------- #
    def priv_obs(self) -> np.ndarray:
        """The 21-D PRIVILEGED flat-state vector the frozen teacher consumes."""
        from RL.student_teacher.teacher_contract import build_teacher_obs21
        return build_teacher_obs21(self.scene)

    def perceived_port_pose(self) -> np.ndarray:
        X, q = self._perceived_port()
        return np.concatenate([X, q]).astype(np.float32)

    # -- gym API ------------------------------------------------------------ #
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._draw_bias()
        self._last_deploy_action = np.zeros(FINAL_INSERT_ACTION_DIM, np.float32)
        self._wrench_baseline = np.asarray(obs["ft"], dtype=np.float64)[:6].copy()
        return self._build_obs69(obs), info

    def _to_sim(self, action):
        if self.action_convention == "deploy":
            return deploy_to_sim_action(action)
        return np.clip(np.asarray(action, np.float32).reshape(6), -1.0, 1.0)

    def step(self, action):
        """Step with a policy action in the wrapper's `action_convention`."""
        self._last_deploy_action = (
            np.clip(np.asarray(action, np.float32).reshape(6), -1.0, 1.0)
            if self.action_convention == "deploy"
            else sim_to_deploy_action(action))
        a_sim = self._to_sim(action)
        obs, r, term, trunc, info = self.env.step(a_sim)
        info = dict(info)
        info["perceived_port_pos_bias"] = self._pos_bias.copy()
        info["perceived_port_rot_bias"] = self._rot_bias.copy()
        info["perceived_grasp_pos_bias"] = self._grasp_pos_bias.copy()
        info["perceived_grasp_rot_bias"] = self._grasp_rot_bias.copy()
        return self._build_obs69(obs), r, term, trunc, info

    def step_sim(self, a_sim):
        """Drive the env with a SIM-convention action (dataset gen w/ teacher).

        Records the deploy-convention equivalent as last_action so the NEXT obs's
        last_action field matches what the deployed net would have output.
        """
        a_sim = np.clip(np.asarray(a_sim, np.float32).reshape(6), -1.0, 1.0)
        self._last_deploy_action = sim_to_deploy_action(a_sim)
        obs, r, term, trunc, info = self.env.step(a_sim)
        return self._build_obs69(obs), r, term, trunc, info


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def make_student_env_a(perception_noise: float = 1.0,
                       level: float = 1.0,
                       start_jitter_xy_m: float | None = None,
                       start_jitter_yaw_rad: float | None = None,
                       start_jitter_tilt_rad: float | None = None,
                       start_curriculum_band: float | None = None,
                       start_curriculum_easy_frac: float | None = None,
                       action_convention: str = "deploy",
                       wrench_mode: str = "zero",
                       grasp_noise: float = 1.0,
                       domain_randomization: bool = False,
                       max_episode_steps: int | None = None,
                       seed: int | None = None) -> StudentObsWrapperA:
    """Build the Contract-A state-student env, pinned at `level` (default 1.0).

    cartesian_residual, base_script OFF, NO images / NO renderer (state-only ->
    light CPU, no GL). Same dynamics / cart scaling as the teacher env so the
    imitated action means the same physical thing (1mm / 1deg, cart_pos_limit=0.20,
    cart_rot_limit=0.35). Heavy reward renderer disabled (w_image=0, beta_s=0).
    """
    from RL.env import EnvConfig
    from RL.scene_env import SceneInsertEnv, SceneEnvConfig

    env_cfg = EnvConfig()
    reward_cfg = dataclasses.replace(env_cfg.reward, w_image=0.0, beta_s=0.0)
    ms = int(max_episode_steps or env_cfg.term.max_steps)
    # These are opt-in start-distribution overrides.  Keeping ``None`` preserves
    # the SceneEnvConfig defaults used by the existing student/align builders.
    start_cfg = {}
    if start_jitter_xy_m is not None:
        start_cfg["jitter_xy_inport_m"] = float(start_jitter_xy_m)
    if start_jitter_yaw_rad is not None:
        start_cfg["jitter_yaw_inport_rad"] = float(start_jitter_yaw_rad)
    if start_jitter_tilt_rad is not None:
        start_cfg["jitter_tilt_inport_rad"] = float(start_jitter_tilt_rad)
    if start_curriculum_band is not None:
        start_cfg["curriculum_band"] = float(start_curriculum_band)
    if start_curriculum_easy_frac is not None:
        start_cfg["curriculum_easy_frac"] = float(start_curriculum_easy_frac)

    cfg = SceneEnvConfig(
        reward=reward_cfg,
        max_episode_steps=ms,
        action_mode="cartesian_residual",
        cart_action_dims=6,
        cart_trans_scale_m=SIM_TRANS_SCALE_M,
        cart_rot_scale_rad=SIM_ROT_SCALE_RAD,
        base_script_enabled=False,
        cart_pos_limit_m=0.20,
        cart_rot_limit_rad=0.35,
        include_images=False,       # state-only -> no renderer, no GL
        domain_randomization=bool(domain_randomization),
        compiled_variant_seed=(int(seed) if domain_randomization and seed is not None else None),
        **start_cfg,
    )
    scene = SceneInsertEnv(cfg)
    scene.set_reset_mode("curriculum")     # + fixed level, NO level file -> pinned
    scene.set_curriculum_level(float(level))
    wrapped = StudentObsWrapperA(scene, perception_noise=perception_noise,
                                 action_convention=action_convention,
                                 wrench_mode=wrench_mode,
                                 grasp_noise=grasp_noise, seed=seed)
    if seed is not None:
        wrapped.reset(seed=int(seed))
    return wrapped


__all__ = [
    "StudentObsWrapperA",
    "make_student_env_a",
    "deploy_to_sim_action",
    "sim_to_deploy_action",
    "FINAL_INSERT_OBS_LEN",
    "FINAL_INSERT_ACTION_DIM",
    "DEPLOY_POS_SCALE",
    "DEPLOY_ROT_SCALE",
    "PERCEPTION_POS_SIGMA",
    "PERCEPTION_ROT_SIGMA",
    "GRASP_POS_SIGMA",
    "GRASP_ROT_SIGMA",
]
