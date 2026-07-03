"""
Reward function for residual SAC on the last-inch cable insertion task.

Implements the spec in REWARD_SPEC.md §2-§6. All components are designed to
be additive; the caller combines them with the weights specified in the
top-level RewardConfig (see §3).

The reward class is intentionally framework-agnostic: it consumes plain
numpy arrays / floats, so it works for SB3, RLlib, skrl, CleanRL, or hand-rolled
SAC. No PyTorch / Gym dependency at import time.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """All reward hyperparameters in one place. Defaults from REWARD_SPEC §8."""

    # §2.1 image
    alpha: float = 1.0           # image L1 coefficient
    beta_s: float = 50.0         # success bonus
    eps: float = 0.02            # image-L1 success threshold

    # §2.2 force piecewise log
    f_low: float = 2.0
    f_optimal: float = 10.0
    f_max: float = 20.0
    beta_low: float = 0.5
    gamma: float = 0.5

    # §2.3 tip–port xy progress
    delta_xy: float = 0.5

    # §2.4 action smoothness
    delta_a: float = 0.1

    # §2.5 lateral force penalty
    delta_lat: float = 0.02

    # §3 component weights in total
    w_image: float = 1.0
    w_force: float = 0.05
    w_xy: float = 0.05
    w_action: float = 0.10
    w_lateral: float = 0.10


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    """Termination thresholds for env wrapper (env.py). Defaults from §6."""

    f_abort_n: float = 24.0      # instant force abort
    f_linger_n: float = 16.0     # sustained-force threshold
    f_linger_sec: float = 1.0    # sustained-force dwell time
    max_steps: int = 600         # 30 s at 20 Hz


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #


def r_image(
    image_curr: np.ndarray,
    image_goal: np.ndarray,
    cfg: RewardConfig,
    bonus_eligible: bool = True,
) -> tuple[float, np.ndarray]:
    """Sparse image reward. Returns (scalar, l1_normalised) so the caller can
    re-use the L1 for termination checks.

    The `+β_s` success bonus is only added when `bonus_eligible` is True.
    The env should set this to False except at the moment the geometric
    success criterion is met (avoids flooding the reward when the image
    is *accidentally* close to the goal — e.g. when the plug is below
    the port and the camera sees only the empty cage).
    """
    if image_curr.shape != image_goal.shape:
        raise ValueError(
            f"image_curr {image_curr.shape} != image_goal {image_goal.shape}"
        )
    # cast to float to avoid uint8 overflow on subtraction
    a = image_curr.astype(np.float32)
    b = image_goal.astype(np.float32)
    l1 = float(np.sum(np.abs(a - b)))
    # normalise by total number of elements
    norm = l1 / max(a.size, 1)
    bonus = cfg.beta_s if (bonus_eligible and norm < cfg.eps) else 0.0
    scalar = -cfg.alpha * norm + bonus
    return scalar, np.array(norm, dtype=np.float32)


def r_force(f_z: float, cfg: RewardConfig) -> float:
    """Piecewise-logarithmic force reward from §2.2. f_z is signed; we use its
    absolute value because pressing in either direction of the insertion axis
    counts as contact (sign convention is gripper-down positive in AIC)."""
    if not math.isfinite(f_z):
        return 0.0
    f = abs(float(f_z))
    if f < 0.0:  # never, defensive
        return 0.0

    log_low = cfg.beta_low * math.log1p(cfg.f_low)            # plateau height
    peak = log_low + cfg.gamma * (cfg.f_optimal - cfg.f_low)   # value at f_optimal

    if f < cfg.f_low:
        return cfg.beta_low * math.log1p(f)
    if f <= cfg.f_optimal:
        return log_low + cfg.gamma * (f - cfg.f_low)
    if f < cfg.f_max:
        return peak - cfg.gamma * (f - cfg.f_optimal)
    # f >= f_max — log-square penalty so it stays bounded near the AIC limit
    delta = f - cfg.f_max
    penalty = math.log1p(delta) ** 2
    return peak - cfg.gamma * (cfg.f_max - cfg.f_optimal) - penalty


def r_xy(tip_xy: np.ndarray, port_xy: np.ndarray, cfg: RewardConfig) -> float:
    """Tip XY minus port entrance XY. Closer = less penalty."""
    diff = np.asarray(tip_xy, dtype=np.float32) - np.asarray(port_xy, dtype=np.float32)
    return float(-cfg.delta_xy * float(np.linalg.norm(diff)))


def r_action(
    a_t: np.ndarray,
    a_prev: np.ndarray,
    cfg: RewardConfig,
) -> float:
    """Smoothness penalty on residual action difference."""
    if a_prev is None:
        return 0.0
    diff = np.asarray(a_t, dtype=np.float32) - np.asarray(a_prev, dtype=np.float32)
    return float(-cfg.delta_a * float(np.linalg.norm(diff)))


def r_lateral(f_xy: np.ndarray, cfg: RewardConfig) -> float:
    """Linear penalty on lateral force magnitude."""
    if f_xy is None or len(f_xy) == 0:
        return 0.0
    mag = float(np.linalg.norm(np.asarray(f_xy, dtype=np.float32)))
    return float(-cfg.delta_lat * mag)


def r_done(
    term_status: Optional[str],
) -> float:
    """One-shot termination bonus/penalty. See §2.6.

    term_status is one of: "success", "force_abort", "off_limit", "timeout", None.
    """
    if term_status == "success":
        return 50.0
    if term_status == "force_abort":
        return -30.0
    if term_status == "off_limit":
        return -10.0
    if term_status == "timeout":
        return 0.0
    return 0.0


# --------------------------------------------------------------------------- #
# total
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class RewardBreakdown:
    """Per-component reward for the step. Sum into reward_total yourself so
    the caller can apply the weights and inspect each term independently."""

    image: float = 0.0
    force: float = 0.0
    xy: float = 0.0
    action: float = 0.0
    lateral: float = 0.0
    done: float = 0.0
    image_l1_norm: float = 0.0


def compute_reward(
    *,
    image_curr: np.ndarray,
    image_goal: np.ndarray,
    f_z: float,
    f_xy: np.ndarray,
    tip_xy: np.ndarray,
    port_xy: np.ndarray,
    a_t: np.ndarray,
    a_prev: Optional[np.ndarray],
    term_status: Optional[str],
    cfg: RewardConfig = RewardConfig(),
    bonus_eligible: bool = True,
) -> tuple[float, RewardBreakdown]:
    """One-shot helper: compute every component and return the weighted total
    and the per-term breakdown.

    `bonus_eligible` gates the +β_s success bonus in `r_image`. Set
    False except at the step the env's geometric success check passes.
    """
    b = RewardBreakdown()
    b.image, b.image_l1_norm = r_image(image_curr, image_goal, cfg,
                                       bonus_eligible=bonus_eligible)
    b.force = r_force(f_z, cfg)
    b.xy = r_xy(tip_xy, port_xy, cfg)
    b.action = r_action(a_t, a_prev, cfg)
    b.lateral = r_lateral(f_xy, cfg)
    b.done = r_done(term_status)

    total = (
        cfg.w_image * b.image
        + cfg.w_force * b.force
        + cfg.w_xy * b.xy
        + cfg.w_action * b.action
        + cfg.w_lateral * b.lateral
        + b.done  # r_done has no weight multiplier (it IS the big one-shot)
    )
    return total, b


# --------------------------------------------------------------------------- #
# termination check
# --------------------------------------------------------------------------- #


def check_termination(
    *,
    image_l1_norm: float,
    f_z: float,
    off_limit_contact: bool,
    step: int,
    cfg_rew: RewardConfig,
    cfg_term: TerminationConfig,
    f_linger_dwell_s: float = 0.0,
    bypass_image_success: bool = False,
) -> Optional[str]:
    """§6 termination rules. Returns one of the status strings or None.

    Pass `f_linger_dwell_s` from your env (rolling how long f_z has been
    above `cfg_term.f_linger_n`). The dwell timer is owned by the env, not
    the reward.

    `bypass_image_success=True` skips the image-L1 success check. Use this
    when the env has a more reliable success criterion (e.g. the
    seated-and-in-contact check for the last-inch env) — without it, an
    "empty" scene can match the goal image and falsely terminate as
    success.
    """
    # success — image goal match (unless bypassed)
    if not bypass_image_success and image_l1_norm < cfg_rew.eps:
        return "success"
    # safety — force instant abort (AIC scoring threshold + buffer)
    if abs(f_z) > cfg_term.f_abort_n:
        return "force_abort"
    # safety — off-limit contact
    if off_limit_contact:
        return "off_limit"
    # safety — sustained over-force
    if abs(f_z) > cfg_term.f_linger_n and f_linger_dwell_s >= cfg_term.f_linger_sec:
        return "force_abort"
    # horizon
    if step >= cfg_term.max_steps:
        return "timeout"
    return None


__all__ = [
    "RewardConfig",
    "TerminationConfig",
    "RewardBreakdown",
    "r_image",
    "r_force",
    "r_xy",
    "r_action",
    "r_lateral",
    "r_done",
    "compute_reward",
    "check_termination",
]
