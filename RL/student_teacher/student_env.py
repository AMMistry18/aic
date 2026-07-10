"""Deployable-STUDENT observation wrapper + env builder (Phase-3 distillation).

`StudentObsWrapper` turns a `SceneInsertEnv` (pure ``cartesian_residual`` mode,
scripted base OFF, level pinned to 1.0) into an env whose observation is the
DEPLOYABLE obs dict the real robot uses -- mirroring `RL/observation.py`:

    image         (H, W, 3*n_cams) uint8   stacked wrist cams (low-res render)
    force         (3,)   float32           wrist F/T force xyz (N)
    tcp_pose      (7,)   float32           gripper/tcp pose (pos + quat wxyz)
    port_pose     (7,)   float32           PERCEIVED port pose (GT + per-episode bias)
    tcp_pose_err  (6,)   float32           tcp vs perceived-port error (port frame)
    last_action   (6,)   float32           previous cartesian residual command

The ONLY privileged leak we allow is the *noised* perceived port pose: the exact
GT port pose (`_port_pos` + the `_cart_frame_R` port basis) PLUS a per-episode-
CONSISTENT Gaussian perception bias (drawn once at reset, held fixed for the
episode). No GT port frame, no exact-pose-as-oracle, no penetration/contact
signals reach the student. `tcp_pose` stays the FK pose (measurable-ish) and the
perception error lives on `port_pose`/`tcp_pose_err`, exactly as at deploy.

The action space is the SAME 6-D cartesian residual the teacher's EFFECTIVE
action lives in (funnel base + learned residual, combined) and what deploy uses:
the student outputs the TOTAL cartesian command; there is no scripted base.

This module ADDS a wrapper only -- it never edits `scene_env.py`; it reads GT
accessors (`_port_pos`, `_cart_frame_R`, `_privileged_obs`) read-only to
synthesize the perceived pose and (for distillation) the teacher's priv obs.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import gymnasium as gym
import mujoco

from RL.observation import ObsConfig, build_obs_dict_from_arrays


# --------------------------------------------------------------------------- #
# perception-noise model (base 1-sigma; scaled by `perception_noise`)
# --------------------------------------------------------------------------- #
# Default ~1.5 mm lateral / 3.5 mm axial position, ~1 deg / 1 deg / 1.5 deg
# orientation. Position sigmas mirror `RL/observation.py` ObsConfig.pos_scale;
# the rotation sigmas are deliberately ~1-2 deg (NOT observation.py's larger
# rot_scale, which is an obs-normalisation scale, not a perception sigma).
PERCEPTION_POS_SIGMA = np.array([1.5e-3, 1.5e-3, 3.5e-3], dtype=np.float64)      # m
PERCEPTION_ROT_SIGMA = np.array([1.745e-2, 1.745e-2, 2.618e-2], dtype=np.float64)  # rad

# Flat student vector-obs layout (image handled separately). Order is fixed and
# shared with the dataset/trainer so the student net indexes consistently.
STUDENT_VEC_KEYS = ("force", "tcp_pose", "port_pose", "tcp_pose_err", "last_action")


def flatten_student_vec(obs: dict) -> np.ndarray:
    """Concatenate the non-image obs into one float32 vector (order = STUDENT_VEC_KEYS).

    force(3) + tcp_pose(7) + port_pose(7) + tcp_pose_err(6) + last_action(6) = 29.
    """
    return np.concatenate(
        [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in STUDENT_VEC_KEYS]
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# tiny quaternion helpers (mujoco-backed; wxyz convention like scene_env)
# --------------------------------------------------------------------------- #
def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


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


# --------------------------------------------------------------------------- #
# wrapper
# --------------------------------------------------------------------------- #
class StudentObsWrapper(gym.Wrapper):
    """Emit the deployable obs dict; keep 6-D cartesian-residual actions."""

    def __init__(self, env: gym.Env, perception_noise: float = 1.0,
                 obs_cfg: ObsConfig | None = None, seed: int | None = None):
        super().__init__(env)
        self.scene = env.unwrapped                 # SceneInsertEnv (GT, read-only)
        if getattr(self.scene, "_renderer", None) is None or not self.scene._cams:
            raise ValueError("StudentObsWrapper needs include_images=True "
                             "(the student obs carries the wrist-cam image).")

        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

        h = int(self.scene.cfg.image_h)
        w = int(self.scene.cfg.image_w)
        n_cams = len(self.scene._cams)
        self._obs_cfg = obs_cfg or ObsConfig(
            image_h=h, image_w=w, n_cams=n_cams, image_ch_per_cam=3)
        ch = n_cams * self._obs_cfg.image_ch_per_cam

        self.perception_noise = float(perception_noise)
        self._rng = np.random.default_rng(seed)
        self._pos_bias = np.zeros(3)
        self._rot_bias = np.zeros(3)

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (h, w, ch), np.uint8),
            "force": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "port_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "tcp_pose_err": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
            "last_action": gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
        })
        self._last_deploy = None

    # -- perception --------------------------------------------------------- #
    def _draw_bias(self):
        s = self.perception_noise
        self._pos_bias = self._rng.normal(0.0, PERCEPTION_POS_SIGMA * s)
        self._rot_bias = self._rng.normal(0.0, PERCEPTION_ROT_SIGMA * s)

    def _perceived_port(self):
        """GT port pose (pos + basis) with the per-episode perception bias applied."""
        port_xyz = self.scene._port_pos.astype(np.float64) + self._pos_bias
        q_gt = _mat_to_quat(self.scene._cart_frame_R)          # port basis -> quat
        q_noised = _qmul(_axisangle_to_quat(self._rot_bias), q_gt)
        n = np.linalg.norm(q_noised)
        if n > 1e-9:
            q_noised = q_noised / n
        return port_xyz.astype(np.float32), q_noised.astype(np.float32)

    def _deploy_obs(self, raw: dict) -> dict:
        tcp_pose = np.asarray(raw["tcp_pose"], dtype=np.float32)
        tcp_xyz, tcp_q = tcp_pose[:3], tcp_pose[3:7]
        force = np.asarray(raw["ft"], dtype=np.float32)[:3]
        image = raw["image"]
        port_xyz, port_q = self._perceived_port()
        return build_obs_dict_from_arrays(
            image=image, tcp_xyz=tcp_xyz, tcp_q_wxyz=tcp_q,
            port_xyz=port_xyz, port_q_wxyz=port_q, force_xyz=force,
            last_action=raw["last_action"], cfg=self._obs_cfg)

    # -- teacher priv obs (distillation only; read-only GT) ----------------- #
    def priv_obs(self) -> np.ndarray:
        """The 21-D PRIVILEGED flat-state vector the frozen teacher consumes."""
        from RL.student_teacher.teacher_contract import build_teacher_obs21
        return build_teacher_obs21(self.scene)

    def perceived_port_pose(self) -> np.ndarray:
        p, q = self._perceived_port()
        return np.concatenate([p, q]).astype(np.float32)

    # -- gym API ------------------------------------------------------------ #
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._draw_bias()
        d = self._deploy_obs(obs)
        self._last_deploy = d
        return d, info

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32).reshape(6), -1.0, 1.0)
        obs, r, term, trunc, info = self.env.step(action)
        d = self._deploy_obs(obs)
        self._last_deploy = d
        info = dict(info)
        info["perceived_port_pos_bias"] = self._pos_bias.copy()
        info["perceived_port_rot_bias"] = self._rot_bias.copy()
        return d, r, term, trunc, info


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def make_student_env(perception_noise: float = 1.0,
                     level: float = 1.0,
                     image_size: int = 84,
                     max_episode_steps: int | None = None,
                     seed: int | None = None) -> StudentObsWrapper:
    """Build the deployable-student env, pinned at `level` (default 1.0).

    cartesian_residual, base_script OFF, image obs ON at `image_size` (square),
    non-privileged dict obs. Mirrors the teacher's env dynamics / cart scaling so
    the imitated action means the same thing (cart_trans_scale_m=1mm,
    cart_rot_scale_rad=1deg, cart_pos_limit=0.20, cart_rot_limit=0.35). The heavy
    reward renderer is disabled (w_image=0, beta_s=0) -- only the low-res obs
    renderer runs.
    """
    from RL.env import EnvConfig
    from RL.scene_env import SceneInsertEnv, SceneEnvConfig

    env_cfg = EnvConfig()
    reward_cfg = dataclasses.replace(env_cfg.reward, w_image=0.0, beta_s=0.0)
    ms = int(max_episode_steps or env_cfg.term.max_steps)
    cfg = SceneEnvConfig(
        reward=reward_cfg,
        max_episode_steps=ms,
        action_mode="cartesian_residual",
        cart_action_dims=6,
        cart_trans_scale_m=1.0e-3,
        cart_rot_scale_rad=float(np.radians(1.0)),
        base_script_enabled=False,
        cart_pos_limit_m=0.20,
        cart_rot_limit_rad=0.35,
        include_images=True,
        image_h=int(image_size),
        image_w=int(image_size),
    )
    scene = SceneInsertEnv(cfg)
    scene.set_reset_mode("curriculum")     # + fixed level, NO level file -> pinned
    scene.set_curriculum_level(float(level))
    wrapped = StudentObsWrapper(scene, perception_noise=perception_noise, seed=seed)
    if seed is not None:
        wrapped.reset(seed=int(seed))
    return wrapped


__all__ = [
    "StudentObsWrapper",
    "make_student_env",
    "flatten_student_vec",
    "STUDENT_VEC_KEYS",
    "PERCEPTION_POS_SIGMA",
    "PERCEPTION_ROT_SIGMA",
]
