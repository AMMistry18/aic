"""Align-first alignment environment (single-RL, align-only).

Historical stage-2 alignment-RL environment; current deployment status is in
`docs/INSERTION_HANDOFF.md`.
Unlike `student_v3_env.StudentV3Env` (a residual-at-contact policy rewarded on
DEPTH progress), this env's ONLY job is to square the plug over the perceived
port. Depth is deliberately NOT rewarded here -- in deployment, once this RL
reports "aligned" the BASE SCRIPT takes over the slow force-limited descent.

Reward = per-axis alignment error reduction (x, y, roll, pitch, yaw), with z
held at a standoff plane (penalize drift + penalize descending-while-misaligned).
No depth term. Episode terminates the instant alignment tolerance is met
(term_status="aligned", +success bonus). Base-env failure terminations
(bad_collision / force_abort / off_limit / timeout) are still honored.

obs69[32:38] is the relative pose error in the PORT frame:
  rel[0], rel[1] = lateral x, y error (m)
  rel[2]         = z standoff along the insertion axis (m); negative = above port
  rel[3:6]       = roll, pitch, yaw error (rad)

The RL commands a full 6-DoF delta (deploy convention, same physical scaling as
the guided controller): normalized action -> DEPLOY_POS_SCALE / DEPLOY_ROT_SCALE.
The RL IS the controller here (no guided base, no residual accumulator).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from RL.student_teacher.student_env_a import (
    DEPLOY_POS_SCALE,
    DEPLOY_ROT_SCALE,
    make_student_env_a,
)
from RL.student_teacher.teacher_contract import build_teacher_obs21


HISTORY = 8
# frame = rel(6) + velocity(6) + wrench(6) + wrench_ema(6) + prev_action(6)
#         + [lat_err, rot_err, z_off, dt](4) = 34
ACTOR_FRAME_DIM = 34
ACTOR_OBS_DIM = HISTORY * ACTOR_FRAME_DIM
PRIVILEGED_DIM = 32


# --- alignment tolerances (the success condition == hand-off trigger) ---------
ALIGN_LAT_TOL_M = 0.001            # lateral within 1 mm
ALIGN_ROT_TOL_RAD = np.radians(1.5)  # rotation within 1.5 deg
# Standoff plane the plug should HOLD during alignment (m along insertion axis,
# rel[2] is negative above the port). ~4 mm above the mouth.
Z_STANDOFF_M = -0.004


# --- reward weights -----------------------------------------------------------
# Progress terms (paid for REDUCING error, not for raw position -> avoids the
# "freeze to stop making it worse" degenerate policy).
W_LAT_PROGRESS = 40.0     # per meter of lateral error removed this step
W_ROT_PROGRESS = 8.0      # per radian of rotation error removed this step
# Gentle raw-magnitude penalty so it does not idle at a mediocre error.
K_LAT_HOLD = 2.0          # per meter of remaining lateral error
K_ROT_HOLD = 0.5          # per radian of remaining rotation error
# z is a HELD standoff, not a drive-to-zero. Penalize drift off the plane and
# (harder) descending while still misaligned.
K_Z_HOLD = 5.0            # per meter off the standoff plane
DESCEND_WHILE_MISALIGNED_PENALTY = 1.0  # flat, per step of downward drift
DESCEND_EPS_M = 0.0005    # z must move inward by > this to count as descending
# Effort / smoothness.
K_ACTION = 0.02
K_ACTION_RATE = 0.05
# Terminal.
ALIGN_SUCCESS_BONUS = 50.0
FAIL_PENALTY = 20.0       # bad_collision / force_abort / off_limit from base env
# Timeout penalty scales with how FAR from aligned we ran out of budget:
# fully aligned (would not time out) -> ~0; totally unaligned -> ~TIMEOUT_PENALTY_MAX.
# proximity = clip(lat_err/tol, 0..cap) + clip(rot_err/tol, 0..cap), normalized.
TIMEOUT_PENALTY_MAX = 20.0
TIMEOUT_PROXIMITY_CAP = 5.0  # errors >5x tolerance count as "fully unaligned"


@dataclass(frozen=True)
class AlignStage:
    name: str
    level: float
    perception_noise: float
    grasp_noise: float
    max_steps: int


STAGES = {
    "align": AlignStage("align", 0.52, 0.25, 0.25, 300),
    "small": AlignStage("small", 0.58, 0.50, 0.50, 300),
    "medium": AlignStage("medium", 0.72, 0.75, 0.75, 350),
    "full": AlignStage("full", 1.00, 1.00, 1.00, 400),
    "robust": AlignStage("robust", 1.00, 1.00, 1.00, 400),
}


def _lat_err(rel: np.ndarray) -> float:
    return float(np.linalg.norm(rel[:2]))


def _rot_err(rel: np.ndarray) -> float:
    return float(np.linalg.norm(rel[3:6]))


def is_aligned(rel: np.ndarray) -> bool:
    """The success condition AND the deployment hand-off trigger to base script."""
    return bool(_lat_err(rel) <= ALIGN_LAT_TOL_M and _rot_err(rel) <= ALIGN_ROT_TOL_RAD)


class AlignEnv(gym.Wrapper):
    """Align-only wrapper around the 69-value deployment contract.

    The RL directly commands 6-DoF deltas (deploy convention). Reward is pure
    alignment shaping (no depth). Terminates on alignment or on any base-env
    failure. This env is intended to be trained first; if a genuinely-aligned
    plug still jams in sim, add a bounded contact RL as a later stage.
    """

    def __init__(self, env: gym.Env, history: int = HISTORY):
        super().__init__(env)
        self.contract_env = env
        self.scene = env.unwrapped
        self.history = int(history)
        if self.history != HISTORY:
            raise ValueError(f"align env requires history={HISTORY}")
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(
                -np.inf, np.inf, (HISTORY, ACTOR_FRAME_DIM), np.float32),
            "privileged": gym.spaces.Box(
                -np.inf, np.inf, (PRIVILEGED_DIM,), np.float32),
        })
        self._frames: deque[np.ndarray] = deque(maxlen=HISTORY)
        self._wrench_ema = np.zeros(6, dtype=np.float64)
        self._prev_wrench = np.zeros(6, dtype=np.float64)
        self._prev_rel = np.zeros(6, dtype=np.float64)
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._last_dt = 0.05
        self._last_info: dict = {}
        self._current_obs69 = np.zeros(69, dtype=np.float32)

    # -- observation ----------------------------------------------------------
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
        lat_err = _lat_err(rel)
        rot_err = _rot_err(rel)
        z_off = float(rel[2] - Z_STANDOFF_M)
        frame = np.concatenate([
            rel, velocity, wrench, self._wrench_ema,
            self._prev_action,
            np.array([lat_err, rot_err, z_off, dt]),
        ])
        if frame.shape != (ACTOR_FRAME_DIM,):
            raise RuntimeError(f"actor frame shape drifted: {frame.shape}")
        self._prev_wrench = wrench
        self._prev_rel = rel
        return frame.astype(np.float32)

    def _observation(self, obs69: np.ndarray, *, append: bool = True) -> dict:
        if append:
            self._frames.append(self._frame(obs69))
        actor = np.stack(tuple(self._frames), axis=0).astype(np.float32)
        return {"actor": actor, "privileged": self._privileged()}

    # -- lifecycle ------------------------------------------------------------
    def reset(self, **kwargs):
        obs69, info = self.env.reset(**kwargs)
        self._current_obs69 = np.asarray(obs69, dtype=np.float32).copy()
        self._last_info = dict(info)
        self._last_dt = float(getattr(self.scene, "_policy_dt_s", 0.05))
        self._wrench_ema[:] = 0.0
        self._prev_wrench[:] = np.asarray(obs69[51:57], dtype=np.float64)
        self._prev_rel[:] = np.asarray(obs69[32:38], dtype=np.float64)
        self._prev_action[:] = 0.0
        first = self._frame(obs69)
        self._frames = deque(
            (first.copy() for _ in range(HISTORY)), maxlen=HISTORY)
        return self._observation(obs69, append=False), info

    def _compose_action(self, action: np.ndarray) -> np.ndarray:
        """RL directly commands the 6-DoF deploy-convention delta.

        No guided base, no residual accumulator: the alignment RL is the whole
        controller. It is free to command z, but the reward strongly discourages
        descending while misaligned, so it learns to hold the standoff and only
        square x/y/rot -- then the base script (outside this env) descends.
        """
        return np.clip(
            np.asarray(action, dtype=np.float64).reshape(6), -1.0, 1.0
        ).astype(np.float32)

    def step(self, action):
        before_rel = self._prev_rel.copy()
        commanded = self._compose_action(action)
        obs69, _base_reward, _base_term, base_trunc, info = self.env.step(commanded)
        self._current_obs69 = np.asarray(obs69, dtype=np.float32).copy()
        info = dict(info)
        self._last_info = info
        self._last_dt = float(info.get("policy_dt_s", self._last_dt))

        rel = np.asarray(obs69[32:38], dtype=np.float64)
        reward, aligned = self._align_reward(before_rel, rel, commanded, info)

        # --- termination: our alignment gate OR base-env hard failures --------
        base_status = info.get("term_status")
        term_status = None
        terminated = False
        if aligned:
            term_status = "aligned"
            terminated = True
        elif base_status in ("bad_collision", "force_abort", "off_limit"):
            term_status = base_status
            terminated = True
            reward -= FAIL_PENALTY
        truncated = bool(base_trunc) or bool(base_status == "timeout")
        # Timeout is not a hard failure, but running out of budget while far from
        # aligned should still hurt -- scaled by how far off we were.
        if truncated and not terminated:
            reward -= self._timeout_penalty(rel)

        self._prev_action = commanded.astype(np.float64)
        out = self._observation(obs69)
        info.update({
            "align_term_status": term_status if term_status else (
                "timeout" if truncated else None),
            "align_aligned": aligned,
            "align_lat_err_m": _lat_err(rel),
            "align_rot_err_rad": _rot_err(rel),
            "align_z_off_m": float(rel[2] - Z_STANDOFF_M),
        })
        return out, float(reward), terminated, truncated, info

    # -- reward ---------------------------------------------------------------
    @staticmethod
    def _timeout_penalty(rel: np.ndarray) -> float:
        """Penalty for timing out, scaled by distance from aligned.

        0 when already at tolerance, up to TIMEOUT_PENALTY_MAX when both lateral
        and rotation errors are >= TIMEOUT_PROXIMITY_CAP x their tolerances.
        """
        lat_ratio = min(_lat_err(rel) / ALIGN_LAT_TOL_M, TIMEOUT_PROXIMITY_CAP)
        rot_ratio = min(_rot_err(rel) / ALIGN_ROT_TOL_RAD, TIMEOUT_PROXIMITY_CAP)
        proximity = 0.5 * (lat_ratio + rot_ratio) / TIMEOUT_PROXIMITY_CAP  # 0..1
        return float(TIMEOUT_PENALTY_MAX * proximity)

    def _align_reward(self, before_rel, rel, commanded, info):
        """Pure alignment reward. NO depth term.

        Paid for REDUCING lateral/rotation error (progress), plus a gentle raw
        penalty so it does not idle. z is held at the standoff plane and
        descending-while-misaligned is penalized. Returns (reward, aligned).
        """
        lat_before, lat_now = _lat_err(before_rel), _lat_err(rel)
        rot_before, rot_now = _rot_err(before_rel), _rot_err(rel)
        aligned = is_aligned(rel)

        # progress toward aligned (positive when error shrinks)
        r = W_LAT_PROGRESS * (lat_before - lat_now)
        r += W_ROT_PROGRESS * (rot_before - rot_now)
        # gentle pressure on remaining error
        r -= K_LAT_HOLD * lat_now
        r -= K_ROT_HOLD * rot_now
        # hold z at standoff (drift off plane is bad in either direction)
        r -= K_Z_HOLD * abs(float(rel[2]) - Z_STANDOFF_M)
        # descending while misaligned is the jam we are eliminating: penalize it.
        # rel[2] more positive == deeper (toward seated); guard on that motion.
        if not aligned:
            z_progress_inward = float(rel[2] - before_rel[2])
            if z_progress_inward > DESCEND_EPS_M:
                r -= DESCEND_WHILE_MISALIGNED_PENALTY
        # effort / smoothness
        act = np.asarray(commanded, dtype=np.float64)
        r -= K_ACTION * float(np.linalg.norm(act))
        r -= K_ACTION_RATE * float(np.linalg.norm(act - self._prev_action))
        # success
        if aligned:
            r += ALIGN_SUCCESS_BONUS
        return float(r), aligned


def make_align_env(stage: str = "full", *, seed: int | None = None,
                   domain_randomization: bool = True) -> AlignEnv:
    """Build the align-only env at a named curriculum stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown align stage: {stage!r} (have {list(STAGES)})")
    s = STAGES[stage]
    base = make_student_env_a(
        perception_noise=s.perception_noise,
        grasp_noise=s.grasp_noise,
        level=s.level,
        action_convention="deploy",
        wrench_mode="baseline",
        domain_randomization=domain_randomization,
        max_episode_steps=s.max_steps,
        seed=seed,
    )
    return AlignEnv(base)


__all__ = [
    "AlignEnv",
    "make_align_env",
    "is_aligned",
    "STAGES",
    "ALIGN_LAT_TOL_M",
    "ALIGN_ROT_TOL_RAD",
    "Z_STANDOFF_M",
]
