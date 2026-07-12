"""Student-v3 contact environment: guided base plus deployable residual history.

The actor receives only an eight-frame deployable history.  The critic receives
the same history plus privileged simulator state.  The privileged channel is
never needed to compute an action and is excluded from actor export.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from RL.student_teacher.parity.evaluate_guided_controller import guided_action
from RL.student_teacher.student_env_a import (
    DEPLOY_POS_SCALE,
    DEPLOY_ROT_SCALE,
    make_student_env_a,
)
from RL.student_teacher.teacher_contract import build_teacher_obs21


HISTORY = 8
ACTOR_FRAME_DIM = 48
ACTOR_OBS_DIM = HISTORY * ACTOR_FRAME_DIM
PRIVILEGED_DIM = 32


@dataclass(frozen=True)
class StudentV3Stage:
    name: str
    level: float
    perception_noise: float
    grasp_noise: float
    max_steps: int


STAGES = {
    "contact": StudentV3Stage("contact", 0.52, 0.25, 0.25, 400),
    "small": StudentV3Stage("small", 0.58, 0.50, 0.50, 400),
    "medium": StudentV3Stage("medium", 0.72, 0.75, 0.75, 500),
    "full": StudentV3Stage("full", 1.00, 1.00, 1.00, 600),
    "robust": StudentV3Stage("robust", 1.00, 1.00, 1.00, 600),
}


class StudentV3Env(gym.Wrapper):
    """Residual policy wrapper around the exact 69-value deployment contract."""

    # Physical residual increment per normalized action and accumulated cage.
    _STEP_POS = np.array([0.0005, 0.0005, 0.00075], dtype=np.float64)
    _STEP_ROT = np.radians(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    _ACCUM_POS = np.array([0.0025, 0.0025, 0.0060], dtype=np.float64)
    _ACCUM_ROT = np.radians(np.array([2.0, 2.0, 2.0], dtype=np.float64))

    def __init__(self, env: gym.Env, history: int = HISTORY):
        super().__init__(env)
        self.contract_env = env
        self.scene = env.unwrapped
        self.history = int(history)
        if self.history != HISTORY:
            raise ValueError(f"student-v3 contract requires history={HISTORY}")
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(
                -np.inf, np.inf, (HISTORY, ACTOR_FRAME_DIM), np.float32),
            "privileged": gym.spaces.Box(
                -np.inf, np.inf, (PRIVILEGED_DIM,), np.float32),
        })
        self._frames: deque[np.ndarray] = deque(maxlen=HISTORY)
        self._depth_window: deque[float] = deque(maxlen=8)
        self._wrench_ema = np.zeros(6, dtype=np.float64)
        self._prev_wrench = np.zeros(6, dtype=np.float64)
        self._prev_rel = np.zeros(6, dtype=np.float64)
        self._cmd_error = np.zeros(6, dtype=np.float64)
        self._prev_residual_action = np.zeros(6, dtype=np.float64)
        self._residual_pos = np.zeros(3, dtype=np.float64)
        self._residual_rot = np.zeros(3, dtype=np.float64)
        self._contact_engaged = False
        self._stalled = False
        self._recovery_pending = False
        self._stall_lateral = np.inf
        self._last_dt = 0.05
        self._last_info: dict = {}
        self._current_obs69 = np.zeros(69, dtype=np.float32)

    @staticmethod
    def _phase(rel: np.ndarray, stalled: bool, engaged: bool) -> np.ndarray:
        out = np.zeros(3, dtype=np.float64)
        out[2 if stalled else 1 if engaged else 0] = 1.0
        return out

    @staticmethod
    def _engagement_gate(obs69: np.ndarray) -> bool:
        rel = np.asarray(obs69[32:38], dtype=np.float64)
        return bool(
            rel[2] >= -0.002
            and np.linalg.norm(rel[:2]) <= 0.002
            and np.linalg.norm(rel[3:]) <= np.radians(2.0)
        )

    def _privileged(self) -> np.ndarray:
        teacher = np.asarray(build_teacher_obs21(self.scene), dtype=np.float64)
        info = self._last_info
        randomization = info.get("domain_randomization", {}) or {}
        extra = np.array([
            float(info.get("contact_force_norm", 0.0)),
            float(info.get("f_z", 0.0)),
            float(info.get("f_lateral", 0.0)),
            float(info.get("insertion_depth_m", self.scene._insertion_depth_m())),
            float(info.get("plug_port_contacts", 0.0)),
            float(randomization.get("friction_scale", 1.0)),
            float(randomization.get("controller_scale", 1.0)),
            float(randomization.get("wrench_limit_scale", 1.0)),
            float(info.get("policy_dt_s", self._last_dt)),
            float(randomization.get("action_delay_steps", 0.0)),
            float(randomization.get("pair_contact_margin_m", 0.0)),
        ], dtype=np.float64)
        out = np.concatenate([teacher, extra])
        if out.shape != (PRIVILEGED_DIM,):
            raise RuntimeError(f"privileged shape drifted: {out.shape}")
        return out.astype(np.float32)

    def _frame(self, obs69: np.ndarray) -> np.ndarray:
        obs69 = np.asarray(obs69, dtype=np.float64).reshape(69)
        rel = obs69[32:38].copy()
        velocity = obs69[19:25].copy()
        wrench = obs69[51:57].copy()
        dt = max(float(self._last_dt), 1e-4)
        self._wrench_ema = 0.8 * self._wrench_ema + 0.2 * wrench
        wrench_derivative = np.clip((wrench - self._prev_wrench) / dt, -500.0, 500.0)
        one_step_progress = float(rel[2] - self._prev_rel[2])
        short_progress = (
            float(rel[2] - self._depth_window[0]) if self._depth_window else 0.0)
        phase = self._phase(rel, self._stalled, self._contact_engaged)
        frame = np.concatenate([
            rel, velocity,
            wrench, self._wrench_ema, wrench_derivative,
            self._prev_residual_action,
            self._cmd_error,
            np.array([one_step_progress, short_progress, dt]),
            phase,
        ])
        if frame.shape != (ACTOR_FRAME_DIM,):
            raise RuntimeError(f"actor frame shape drifted: {frame.shape}")
        self._prev_wrench = wrench
        self._prev_rel = rel
        self._depth_window.append(float(rel[2]))
        return frame.astype(np.float32)

    def _observation(self, obs69: np.ndarray, *, append: bool = True) -> dict:
        if append:
            self._frames.append(self._frame(obs69))
        actor = np.stack(tuple(self._frames), axis=0).astype(np.float32)
        return {"actor": actor, "privileged": self._privileged()}

    def reset(self, **kwargs):
        obs69, info = self.env.reset(**kwargs)
        self._current_obs69 = np.asarray(obs69, dtype=np.float32).copy()
        self._last_info = dict(info)
        self._last_dt = float(getattr(self.scene, "_policy_dt_s", 0.05))
        self._wrench_ema[:] = 0.0
        self._prev_wrench[:] = np.asarray(obs69[51:57], dtype=np.float64)
        self._prev_rel[:] = np.asarray(obs69[32:38], dtype=np.float64)
        self._cmd_error[:] = 0.0
        self._prev_residual_action[:] = 0.0
        self._residual_pos[:] = 0.0
        self._residual_rot[:] = 0.0
        self._contact_engaged = False
        self._stalled = False
        self._recovery_pending = False
        self._stall_lateral = np.inf
        self._depth_window.clear()
        first = self._frame(obs69)
        self._frames = deque(
            (first.copy() for _ in range(HISTORY)), maxlen=HISTORY)
        return self._observation(obs69, append=False), info

    def _compose_action(self, obs69: np.ndarray, residual_action: np.ndarray):
        residual_action = np.clip(
            np.asarray(residual_action, dtype=np.float64).reshape(6), -1.0, 1.0)
        if not self._contact_engaged and self._engagement_gate(obs69):
            self._contact_engaged = True

        guided = np.asarray(guided_action(obs69), dtype=np.float64)
        if not self._contact_engaged:
            residual_action = np.zeros(6, dtype=np.float64)

        old_pos, old_rot = self._residual_pos.copy(), self._residual_rot.copy()
        self._residual_pos = np.clip(
            self._residual_pos + residual_action[:3] * self._STEP_POS,
            -self._ACCUM_POS, self._ACCUM_POS)
        self._residual_rot = np.clip(
            self._residual_rot + residual_action[3:] * self._STEP_ROT,
            -self._ACCUM_ROT, self._ACCUM_ROT)
        delta_pos = self._residual_pos - old_pos
        delta_rot = self._residual_rot - old_rot
        residual_deploy = np.concatenate([
            delta_pos / DEPLOY_POS_SCALE,
            delta_rot / DEPLOY_ROT_SCALE,
        ])
        combined = np.clip(guided + residual_deploy, -1.0, 1.0).astype(np.float32)
        return combined, residual_action, np.concatenate([delta_pos, delta_rot])

    def step(self, action):
        before_rel = self._prev_rel.copy()
        combined, residual_action, residual_physical = self._compose_action(
            self._current_obs69, action)
        obs69, reward, terminated, truncated, info = self.env.step(combined)
        self._current_obs69 = np.asarray(obs69, dtype=np.float32).copy()
        info = dict(info)
        self._last_info = info
        self._last_dt = float(info.get("policy_dt_s", self._last_dt))

        rel = np.asarray(obs69[32:38], dtype=np.float64)
        achieved = rel - before_rel
        commanded = np.concatenate([
            combined[:3] * DEPLOY_POS_SCALE,
            combined[3:] * DEPLOY_ROT_SCALE,
        ])
        self._cmd_error = commanded - achieved
        depth_progress = float(rel[2] - before_rel[2])
        force = float(info.get("contact_force_norm", 0.0))
        lateral = float(info.get("lateral_error_m", np.linalg.norm(rel[:2])))
        was_stalled = self._stalled
        self._stalled = bool(
            self._contact_engaged and force >= 8.0 and abs(depth_progress) < 0.0002)

        shaping = 0.0
        if self._contact_engaged and int(info.get("plug_port_contacts", 0)) > 0:
            breakdown = info.get("breakdown")
            shaping -= 0.7 * float(getattr(breakdown, "xy", 0.0))
        if self._stalled:
            shaping -= 2.0
            self._recovery_pending = True
            self._stall_lateral = min(self._stall_lateral, lateral)
        if was_stalled and depth_progress < -0.0002:
            shaping += 0.5
        if (self._recovery_pending and depth_progress > 0.0002
                and lateral < self._stall_lateral):
            shaping += 1.0
            self._recovery_pending = False
            self._stall_lateral = np.inf
        shaping -= 0.015 * min(force, 100.0)
        shaping -= 0.05 * float(np.linalg.norm(residual_action))
        shaping -= 0.10 * float(np.linalg.norm(
            residual_action - self._prev_residual_action))
        if depth_progress < -0.0015:
            shaping -= 1.0
        reward = float(reward) + shaping

        self._prev_residual_action = residual_action.copy()
        out = self._observation(obs69)
        info.update({
            "v3_shaping": shaping,
            "v3_contact_engaged": self._contact_engaged,
            "v3_stalled": self._stalled,
            "v3_residual_pos_m": self._residual_pos.copy(),
            "v3_residual_rot_rad": self._residual_rot.copy(),
            "v3_residual_delta": residual_physical.copy(),
            "v3_combined_action": combined.copy(),
        })
        return out, reward, terminated, truncated, info


def make_student_v3_env(
    *,
    seed: Optional[int] = None,
    stage: str = "contact",
    domain_randomization: bool = True,
) -> StudentV3Env:
    if stage not in STAGES:
        raise ValueError(f"unknown Student-v3 stage: {stage!r}")
    cfg = STAGES[stage]
    base = make_student_env_a(
        perception_noise=cfg.perception_noise,
        grasp_noise=cfg.grasp_noise,
        level=cfg.level,
        action_convention="deploy",
        wrench_mode="baseline",
        domain_randomization=domain_randomization,
        max_episode_steps=cfg.max_steps,
        seed=seed,
    )
    return StudentV3Env(base)


__all__ = [
    "StudentV3Env", "StudentV3Stage", "STAGES", "make_student_v3_env",
    "HISTORY", "ACTOR_FRAME_DIM", "ACTOR_OBS_DIM", "PRIVILEGED_DIM",
]
