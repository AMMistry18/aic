"""Residual-on-teacher RL environment (Phase 1 / Fix A, option b).

A gym wrapper that puts a LEARNED residual on top of the scripted FUNNEL teacher.
The base env is SceneInsertEnv in pure ``cartesian_residual`` mode with the scripted
base motion OFF and PRIVILEGED flat-state obs ON, level pinned to 1.0. Each step:

    base       = funnel_teacher.act(scene_env, obs)          # the ~70% scripted policy
    effective  = clip(base + residual * RESIDUAL_SCALE, -1, 1)
    obs, r, .. = scene_env.step(effective)

The RL policy therefore learns a small CORRECTION on the 70% base (RESIDUAL_SCALE
is a fraction of the full action range), not a rewrite. At residual == 0 the wrapped
env reproduces the funnel teacher (~70%). The impedance controller (cart_kp_pos=100,
cart_max_wrench=10 N) is untouched; the funnel teacher's control law is the base.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import gymnasium as gym

RESIDUAL_SCALE_DEFAULT = 0.15   # residual perturbation as a fraction of the full range


class ResidualTeacherWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, residual_scale: float = RESIDUAL_SCALE_DEFAULT):
        super().__init__(env)
        from RL.student_teacher.scripted_teacher_funnel import ScriptedTeacher
        self._scene = env.unwrapped            # underlying SceneInsertEnv (GT access)
        adim = int(self._scene.action_space.shape[0])
        self._teacher = ScriptedTeacher(action_dim=adim)
        self.residual_scale = float(residual_scale)
        # policy action = residual in [-1, 1]^adim; obs passes through (privileged Box)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (adim,), np.float32)
        self.observation_space = env.observation_space
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._teacher.reset()
        self._last_obs = obs
        return obs, info

    def step(self, residual):
        base = self._teacher.act(self._scene, self._last_obs)          # scripted funnel base
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        effective = np.clip(base.astype(np.float64) + residual * self.residual_scale,
                            -1.0, 1.0).astype(np.float32)
        obs, r, term, trunc, info = self.env.step(effective)
        self._last_obs = obs
        info = dict(info)
        info["teacher_base_action"] = np.asarray(base, dtype=np.float32)
        return obs, r, term, trunc, info


def make_residual_teacher_env(residual_scale: float = RESIDUAL_SCALE_DEFAULT,
                              level: float = 1.0,
                              max_episode_steps: int | None = None,
                              seed: int | None = None) -> gym.Env:
    """Build the residual-on-funnel-teacher env, pinned at `level` (default 1.0).

    Pure cartesian_residual, base_script OFF, privileged flat-state obs, no images.
    """
    from RL.env import EnvConfig
    from RL.scene_env import SceneInsertEnv, SceneEnvConfig
    from gymnasium.wrappers import TimeLimit

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
        cart_pos_limit_m=0.20,        # matches the funnel-teacher eval envelope
        cart_rot_limit_rad=0.35,
        include_images=False,         # privileged state obs -> no renderer (fast, no GL)
        privileged_obs=True,
    )
    scene = SceneInsertEnv(cfg)
    scene.set_reset_mode("curriculum")     # + fixed level, NO level file -> pinned
    scene.set_curriculum_level(float(level))
    wrapped = ResidualTeacherWrapper(scene, residual_scale=residual_scale)
    wrapped = TimeLimit(wrapped, max_episode_steps=ms)
    if seed is not None:
        wrapped.reset(seed=int(seed))
    return wrapped
