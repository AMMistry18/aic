"""Force-reactive seat environment, with the script hand-off pinned at the chamfer."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from RL.student_teacher.student_env_a import make_student_env_a
from RL.student_teacher.teacher_contract import build_teacher_obs21


HISTORY = 8
ACTOR_FRAME_DIM = 34
PRIVILEGED_DIM = 32

# Reverse-curriculum retraction is measured FROM FULLY SEATED, not from the
# mouth.  Retracting 40.3 mm from the 45.8 mm seated pose places reset insertion
# depth at ~5.5 mm: the mouth/ridge chamfer where the script actually stalls.
SEAT_RETRACTION_FROM_SEATED_M = 0.0403
SEAT_LAST_INCH_M = 0.090
SEAT_LEVEL = SEAT_RETRACTION_FROM_SEATED_M / SEAT_LAST_INCH_M

# Contact-scale action: saturated deploy commands become at most 0.30 mm x/y,
# 0.70 mm axial, and 0.92/0.92/1.38 deg rotation per policy step.  This retains
# useful chamfer-relief wiggles without exposing the seat policy to gross motion.
SEAT_ACTION_GAIN = 0.20

# Reward weights.  Progress uses physical meters, forces use newtons.
# The 1200 weight still produced 8/8 crooked-deep bad collisions in the first
# squareness re-smoke.  At 800, a saturated 0.7 mm depth step is worth ~0.56,
# well below the -1.47 gate-depth cost of a 0.3 rad axis error.
W_DEPTH = 800.0
W_LAT_PROGRESS = 400.0
K_LAT_HOLD = 20.0
# Match SceneEnvConfig's crooked-at-depth bad-collision gate.  Squareness is
# free at the mouth and becomes important only as the plug approaches the gate.
W_SQUARE_AXIS = 4.0
W_SQUARE_ROLL = 4.0
SQUARE_AXIS_REF_RAD = 0.35
SQUARE_ROLL_REF_RAD = 0.35
SQUARE_DEPTH_ONSET = 0.15
SQUARE_DEPTH_GATE = 0.45
W_FORCE_DIRECTION = 0.50
FORCE_RELIEF_DEPTH_REF_M = 0.5e-3
FORCE_LATERAL_RELIEF_REF_N = 5.0
K_FORCE_LATERAL = 0.03
FORCE_SOFT_START_N = 15.0
FORCE_SOFT_RANGE_N = 5.0
W_FORCE_SOFT = 3.0
K_ACTION = 0.02
K_ACTION_RATE = 0.05
SEATED_SUCCESS_BONUS = 50.0
FAIL_PENALTY = 20.0
FAIL_STATUSES = frozenset(("force_abort", "bad_collision", "off_limit"))


@dataclass(frozen=True)
class SeatStage:
    name: str
    lateral_radius_m: float
    rotation_radius_rad: float
    perception_noise: float
    grasp_noise: float
    max_steps: int


STAGES = {
    "tight": SeatStage("tight", 0.7e-3, np.radians(0.6), 0.0, 0.0, 300),
    "band": SeatStage("band", 1.5e-3, np.radians(1.5), 0.0, 0.0, 350),
    # Hidden grasp perturbation is intentionally introduced only at full.
    "full": SeatStage("full", 1.5e-3, np.radians(1.5), 0.0, 0.10, 400),
}


def _lat_err(rel: np.ndarray) -> float:
    return float(np.linalg.norm(rel[:2]))


def _rot_err(rel: np.ndarray) -> float:
    return float(np.linalg.norm(rel[3:6]))


class SeatEnv(gym.Wrapper):
    """Stage-A wrapper retaining AlignEnv's 8x34 actor and 32-D critic ABI."""

    def __init__(self, env: gym.Env, history: int = HISTORY):
        super().__init__(env)
        self.contract_env = env
        self.scene = env.unwrapped
        self.history = int(history)
        if self.history != HISTORY:
            raise ValueError(f"seat env requires history={HISTORY}")
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(-np.inf, np.inf, (HISTORY, ACTOR_FRAME_DIM), np.float32),
            "privileged": gym.spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), np.float32),
        })
        self._frames: deque[np.ndarray] = deque(maxlen=HISTORY)
        self._wrench_ema = np.zeros(6, dtype=np.float64)
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._last_dt = 0.05
        self._last_info: dict = {}
        self._current_obs69 = np.zeros(69, dtype=np.float32)
        self._prev_f_lateral = 0.0
        self._last_reward_terms: dict[str, float] = {}

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
        wrench = obs69[51:57].copy()
        self._wrench_ema = 0.8 * self._wrench_ema + 0.2 * wrench
        frame = np.concatenate([
            rel, obs69[19:25], wrench, self._wrench_ema, self._prev_action,
            np.array([_lat_err(rel), _rot_err(rel), rel[2], max(self._last_dt, 1e-4)]),
        ])
        if frame.shape != (ACTOR_FRAME_DIM,):
            raise RuntimeError(f"actor frame shape drifted: {frame.shape}")
        return frame.astype(np.float32)

    def _observation(self, obs69: np.ndarray, *, append: bool = True) -> dict:
        if append:
            self._frames.append(self._frame(obs69))
        return {"actor": np.stack(tuple(self._frames), axis=0).astype(np.float32),
                "privileged": self._privileged()}

    def reset(self, **kwargs):
        obs69, info = self.env.reset(**kwargs)
        self._current_obs69 = np.asarray(obs69, dtype=np.float32).copy()
        self._last_info = dict(info)
        self._last_dt = float(getattr(self.scene, "_policy_dt_s", 0.05))
        self._wrench_ema[:] = 0.0
        self._prev_action[:] = 0.0
        self._prev_f_lateral = 0.0
        self._last_reward_terms = {}
        first = self._frame(obs69)
        self._frames = deque((first.copy() for _ in range(HISTORY)), maxlen=HISTORY)
        return self._observation(obs69, append=False), info

    def step(self, action):
        before_rel = np.asarray(self._current_obs69[32:38], dtype=np.float64).copy()
        commanded = (SEAT_ACTION_GAIN * np.clip(
            np.asarray(action, dtype=np.float64).reshape(6), -1.0, 1.0))
        obs69, _base_reward, terminated, truncated, info = self.env.step(commanded.astype(np.float32))
        self._current_obs69 = np.asarray(obs69, dtype=np.float32).copy()
        self._last_info = dict(info)
        self._last_dt = float(info.get("policy_dt_s", self._last_dt))
        rel = np.asarray(obs69[32:38], dtype=np.float64)
        reward = self._seat_reward(before_rel, rel, commanded, info)
        self._prev_action = commanded.copy()
        info = dict(info)
        info.update({
            "seat_lat_err_m": _lat_err(rel),
            "seat_rot_err_rad": _rot_err(rel),
            "seat_commanded_action": commanded.astype(np.float32),
            "seat_reward_terms": dict(self._last_reward_terms),
        })
        return self._observation(obs69), reward, terminated, truncated, info

    @staticmethod
    def _finite(value, default: float = 0.0) -> float:
        value = float(value)
        return value if np.isfinite(value) else float(default)

    def _seat_reward(self, before_rel, rel, commanded, info) -> float:
        """FORGE-style contact reward: make progress while keeping force gentle.

        Force direction is rewarded only while moving inward.  Its score is the
        axial share of contact force plus a bonus when lateral force drops from
        the previous step.  A separate lateral-force magnitude cost means that
        depth-up + side-load-down is strictly better than shoving with the same
        depth gain.  Absolute axial force is never rewarded without progress.

        Lateral pose shaping is explicit: reducing ``norm(rel[:2])`` is positive
        and remaining error is a small cost.  We intentionally do not reuse
        ``breakdown.xy``; that term is already non-positive in ``RL.reward`` and
        subtracting it would reward larger lateral error.

        Squareness shaping uses the same axis, roll, and depth scales as the
        environment's crooked-at-depth safety gate.  A quadratic depth ramp is
        zero near the mouth, then makes normalized axis/roll error increasingly
        costly as depth approaches the gate.  It shapes behavior only; the base
        environment remains responsible for termination.
        """
        before = np.asarray(before_rel, dtype=np.float64).reshape(6)
        after = np.asarray(rel, dtype=np.float64).reshape(6)
        act = np.asarray(commanded, dtype=np.float64).reshape(6)

        depth_progress = self._finite(after[2] - before[2])
        depth_term = W_DEPTH * depth_progress

        lat_before = _lat_err(before)
        lat_now = _lat_err(after)
        lateral_term = (W_LAT_PROGRESS * (lat_before - lat_now)
                        - K_LAT_HOLD * lat_now)

        depth_norm = max(0.0, self._finite(info.get("depth_norm", 0.0)))
        depth_ramp = float(np.clip(
            (depth_norm - SQUARE_DEPTH_ONSET)
            / (SQUARE_DEPTH_GATE - SQUARE_DEPTH_ONSET), 0.0, 1.0)) ** 2
        axis_error = abs(self._finite(info.get("plug_axis_error_rad", 0.0)))
        roll_error = abs(self._finite(info.get("plug_roll_error_rad", 0.0)))
        axis_ratio = float(np.clip(axis_error / SQUARE_AXIS_REF_RAD, 0.0, 1.0))
        roll_ratio = float(np.clip(roll_error / SQUARE_ROLL_REF_RAD, 0.0, 1.0))
        squareness_term = -depth_ramp * (
            W_SQUARE_AXIS * axis_ratio * axis_ratio
            + W_SQUARE_ROLL * roll_ratio * roll_ratio)

        f_z = abs(self._finite(info.get("f_z", 0.0)))
        f_lateral = max(0.0, self._finite(info.get("f_lateral", 0.0)))
        f_norm = max(0.0, self._finite(info.get("contact_force_norm", 0.0)))
        progress_gate = float(np.clip(
            max(depth_progress, 0.0) / FORCE_RELIEF_DEPTH_REF_M, 0.0, 1.0))
        force_sum = f_z + f_lateral
        axial_share = f_z / force_sum if force_sum > 1e-9 else 0.0
        lateral_drop = max(0.0, self._prev_f_lateral - f_lateral)
        relief = float(np.clip(
            lateral_drop / FORCE_LATERAL_RELIEF_REF_N, 0.0, 1.0))
        force_direction_term = (
            W_FORCE_DIRECTION * progress_gate * (axial_share + relief))
        lateral_force_term = -K_FORCE_LATERAL * f_lateral

        force_excess = max(0.0, f_norm - FORCE_SOFT_START_N)
        force_ratio = force_excess / FORCE_SOFT_RANGE_N
        force_soft_term = -W_FORCE_SOFT * force_ratio * force_ratio

        effort_term = -K_ACTION * float(np.linalg.norm(act))
        action_rate_term = -K_ACTION_RATE * float(np.linalg.norm(
            act - self._prev_action))

        term_status = info.get("term_status")
        success_term = SEATED_SUCCESS_BONUS if term_status == "success" else 0.0
        fail_term = -FAIL_PENALTY if term_status in FAIL_STATUSES else 0.0

        terms = {
            "depth": depth_term,
            "lateral": lateral_term,
            "squareness": squareness_term,
            "force_direction": force_direction_term,
            "lateral_force": lateral_force_term,
            "force_soft_cap": force_soft_term,
            "effort": effort_term,
            "action_rate": action_rate_term,
            "success": success_term,
            "failure": fail_term,
        }
        reward = float(sum(terms.values()))
        self._prev_f_lateral = f_lateral
        self._last_reward_terms = {name: float(value) for name, value in terms.items()}
        if not np.isfinite(reward):
            raise FloatingPointError(f"non-finite seat reward: {self._last_reward_terms}")
        return reward


def _sampler_amplitudes(stage: SeatStage) -> tuple[float, float, float]:
    """Translate total radial bands to SceneEnvConfig's per-axis amplitudes.

    The scene sampler independently applies x/y and yaw/tilt-x/tilt-y, then
    multiplies every in-port amplitude by level.  Dividing by both level and
    sqrt(dimensions) makes the configured total bounds physical bands.
    """
    return (stage.lateral_radius_m / (SEAT_LEVEL * np.sqrt(2.0)),
            stage.rotation_radius_rad / (SEAT_LEVEL * np.sqrt(3.0)),
            stage.rotation_radius_rad / (SEAT_LEVEL * np.sqrt(3.0)))


def make_seat_env(stage: str = "full", *, seed: int | None = None,
                  domain_randomization: bool = True) -> SeatEnv:
    """Build a seat-only environment at the script's chamfer hand-off pose."""
    if stage not in STAGES:
        raise ValueError(f"unknown seat stage: {stage!r} (have {list(STAGES)})")
    s = STAGES[stage]
    xy, yaw, tilt = _sampler_amplitudes(s)
    base = make_student_env_a(
        perception_noise=s.perception_noise,
        grasp_noise=s.grasp_noise,
        level=SEAT_LEVEL,
        start_jitter_xy_m=xy,
        start_jitter_yaw_rad=yaw,
        start_jitter_tilt_rad=tilt,
        start_curriculum_band=0.0,
        start_curriculum_easy_frac=0.0,
        action_convention="deploy",
        wrench_mode="baseline",
        domain_randomization=domain_randomization,
        max_episode_steps=s.max_steps,
        seed=seed,
    )
    return SeatEnv(base)


__all__ = [
    "ACTOR_FRAME_DIM", "HISTORY", "PRIVILEGED_DIM", "SEAT_LEVEL",
    "SEAT_ACTION_GAIN", "SEAT_RETRACTION_FROM_SEATED_M", "STAGES",
    "SeatEnv", "make_seat_env",
]
