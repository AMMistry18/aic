"""Redesigned reward for residual last-inch insertion (2026-07).

Structure (replaces the legacy depth/xy/axis/boundary stack, see
reward_legacy.py):

  1. Potential-based aligned-approach shaping: Phi(s) = w_progress * P(s) * A(s)
     where P in [0,1] grows as the tip nears the seated pose and A in [0,1] is a
     Gaussian alignment kernel over lateral / axis / roll errors. The per-step
     reward is Phi(s') - Phi(s), so the return over any path is bounded by the
     potential range and cannot be farmed by hovering (Ng et al. 1999 shaping).
     Approaching while aligned pays; approaching misaligned pays ~nothing;
     losing alignment at fixed depth costs.

  2. Entry-alignment potential: a small depth-independent potential on A(s).
     This gives the residual policy a direct signal to line up before the
     scripted base advance pushes into the cage; hovering aligned still pays zero.

  3. Predicted-miss penalty (exponential): while the tip is outside the port,
     ray-cast the plug axis to the entrance plane. If going straight would miss
     the opening, penalize exponentially in the predicted lateral miss. This is
     the "will collide if it goes straight" geometry term.

  4. Heavy contact-force penalty above 20 N (task limit), quadratic in the
     excess. There is no cost below the limit, so useful seating contact is not
     discouraged; the tail ramps quickly through the 20-25 N guard band before
     termination in scene_env.

  5. Over-insertion penalty past the seated depth, and a small wedge cost on
     interpenetration excess over the calibrated free-sliding baseline.

  6. Action regularization: first-difference action-rate penalty (legged_gym
     convention, weight ~1% of the per-step task scale), a small second-
     difference (acceleration) term, and a small magnitude bias toward slow
     residuals.

  7. Sparse terminal outcomes: success bonus; side-hit / force / bad-collision /
     off-limit aborts; mild timeout cost.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """Knobs for the residual-insertion reward."""

    # (1) potential-based aligned approach
    w_progress: float = 30.0          # full clean approach earns ~w_progress
    w_entry_align: float = 6.0        # bounded pre-entry alignment potential
    approach_span_m: float = 0.075    # nominal start-to-seated distance
    align_sigma_lat_m: float = 0.004
    align_sigma_axis_rad: float = 0.0873     # ~5 deg
    align_sigma_roll_rad: float = 0.0873     # ~5 deg

    # (2) exponential predicted-miss penalty (outside the port only)
    w_miss: float = 1.0               # per-step cost at full miss
    miss_free_m: float = 0.0025       # inside opening clearance: no penalty
    miss_ref_m: float = 0.010         # miss_free + miss_ref => full penalty
    miss_exp_k: float = 3.0           # exponent sharpness over the ramp

    # (3) contact-force penalty above the 20 N task budget
    force_penalty_start_n: float = 20.0
    force_penalty_ref_n: float = 5.0      # excess for ratio 1.0 (i.e. 25 N)
    force_penalty_cap: float = 1.0        # cap at the 25 N abort guard band
    w_force: float = 1.5

    # (4) over-insertion + wedge
    w_overinsert: float = 2.0
    overinsert_ref_m: float = 0.002       # full weight at the abort bound
    w_wedge: float = 0.5
    wedge_ref_m: float = 0.0015           # full weight at the abort bound

    # (5) action regularization (actions are [-1,1] residuals)
    w_action_rate: float = 0.02           # * sum((a_t - a_{t-1})^2)
    w_action_accel: float = 0.005         # * sum((a_t - 2a_{t-1} + a_{t-2})^2)
    w_action_mag: float = 0.01            # * sum(a_t^2)  (bias toward slow)

    # (6) terminal outcomes
    success_bonus: float = 50.0
    side_hit_penalty: float = 20.0
    force_abort_penalty: float = 10.0
    bad_collision_penalty: float = 10.0
    off_limit_penalty: float = 10.0
    timeout_penalty: float = 2.0


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #


def alignment_score(
    lateral_error_m: Optional[float],
    axis_error_rad: Optional[float],
    roll_error_rad: Optional[float],
    cfg: RewardConfig,
) -> float:
    """Gaussian alignment kernel in [0,1]; 1 = centered, straight, keyed."""
    q = 0.0
    if lateral_error_m is not None and math.isfinite(float(lateral_error_m)):
        q += (float(lateral_error_m) / max(cfg.align_sigma_lat_m, 1e-9)) ** 2
    if axis_error_rad is not None and math.isfinite(float(axis_error_rad)):
        q += (float(axis_error_rad) / max(cfg.align_sigma_axis_rad, 1e-9)) ** 2
    if roll_error_rad is not None and math.isfinite(float(roll_error_rad)):
        q += (float(roll_error_rad) / max(cfg.align_sigma_roll_rad, 1e-9)) ** 2
    return float(np.exp(-q))


def approach_potential(
    dist_to_seated_m: Optional[float],
    lateral_error_m: Optional[float],
    axis_error_rad: Optional[float],
    roll_error_rad: Optional[float],
    cfg: RewardConfig,
) -> float:
    """Phi(s) = w_progress * proximity * alignment, in [0, w_progress]."""
    if dist_to_seated_m is None or not math.isfinite(float(dist_to_seated_m)):
        return 0.0
    prox = 1.0 - float(np.clip(
        float(dist_to_seated_m) / max(cfg.approach_span_m, 1e-6), 0.0, 1.0))
    align = alignment_score(lateral_error_m, axis_error_rad, roll_error_rad, cfg)
    return cfg.w_progress * prox * align


def entry_align_potential(
    lateral_error_m: Optional[float],
    axis_error_rad: Optional[float],
    roll_error_rad: Optional[float],
    cfg: RewardConfig,
) -> float:
    """Depth-independent potential that pays only for becoming aligned."""
    return cfg.w_entry_align * alignment_score(
        lateral_error_m, axis_error_rad, roll_error_rad, cfg)


def r_miss(predicted_miss_m: Optional[float], outside_port: bool,
           cfg: RewardConfig) -> float:
    """Exponential penalty when a straight-line insertion would miss the port."""
    if not outside_port or predicted_miss_m is None:
        return 0.0
    miss = float(predicted_miss_m)
    if not math.isfinite(miss):
        miss = cfg.miss_free_m + cfg.miss_ref_m   # axis points away: full miss
    frac = float(np.clip(
        (miss - cfg.miss_free_m) / max(cfg.miss_ref_m, 1e-9), 0.0, 1.0))
    if frac <= 0.0:
        return 0.0
    k = max(cfg.miss_exp_k, 1e-6)
    shape = float(np.expm1(k * frac) / np.expm1(k))   # (0,1], exponential ramp
    return -cfg.w_miss * shape


def r_force(f_peak_n: Optional[float], cfg: RewardConfig) -> float:
    """Heavy quadratic cost on contact force above the 20 N task budget."""
    if f_peak_n is None or not math.isfinite(float(f_peak_n)):
        return 0.0
    excess = max(0.0, abs(float(f_peak_n)) - cfg.force_penalty_start_n)
    ratio = float(np.clip(excess / max(cfg.force_penalty_ref_n, 1e-9),
                          0.0, cfg.force_penalty_cap))
    return -cfg.w_force * ratio * ratio


def r_overinsert(overinsert_m: Optional[float], cfg: RewardConfig) -> float:
    """Quadratic cost for pushing past the seated depth."""
    if overinsert_m is None or not math.isfinite(float(overinsert_m)):
        return 0.0
    ratio = float(np.clip(
        max(0.0, float(overinsert_m)) / max(cfg.overinsert_ref_m, 1e-9), 0.0, 1.0))
    return -cfg.w_overinsert * ratio * ratio


def r_wedge(pen_excess_m: Optional[float], cfg: RewardConfig) -> float:
    """Small cost for interpenetration beyond the calibrated sliding baseline."""
    if pen_excess_m is None or not math.isfinite(float(pen_excess_m)):
        return 0.0
    ratio = float(np.clip(
        max(0.0, float(pen_excess_m)) / max(cfg.wedge_ref_m, 1e-9), 0.0, 1.0))
    return -cfg.w_wedge * ratio * ratio


def r_action_smooth(
    a_t: np.ndarray,
    a_prev: Optional[np.ndarray],
    a_prev2: Optional[np.ndarray],
    cfg: RewardConfig,
) -> tuple[float, float, float]:
    """(rate, accel, magnitude) penalties on the residual action sequence."""
    a = np.asarray(a_t, dtype=np.float64)
    mag = -cfg.w_action_mag * float(np.sum(a * a))
    if a_prev is None:
        return 0.0, 0.0, mag
    p = np.asarray(a_prev, dtype=np.float64)
    d1 = a - p
    rate = -cfg.w_action_rate * float(np.sum(d1 * d1))
    if a_prev2 is None:
        return rate, 0.0, mag
    p2 = np.asarray(a_prev2, dtype=np.float64)
    d2 = a - 2.0 * p + p2
    accel = -cfg.w_action_accel * float(np.sum(d2 * d2))
    return rate, accel, mag


def r_done(term_status: Optional[str], cfg: RewardConfig) -> float:
    """Sparse terminal outcome."""
    if term_status == "success":
        return cfg.success_bonus
    if term_status == "side_hit":
        return -cfg.side_hit_penalty
    if term_status == "force_abort":
        return -cfg.force_abort_penalty
    if term_status == "bad_collision":
        return -cfg.bad_collision_penalty
    if term_status == "off_limit":
        return -cfg.off_limit_penalty
    if term_status == "timeout":
        return -cfg.timeout_penalty
    return 0.0


# --------------------------------------------------------------------------- #
# total
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class RewardBreakdown:
    """Weighted per-component contributions."""

    progress: float = 0.0
    entry_align: float = 0.0
    miss: float = 0.0
    force: float = 0.0
    overinsert: float = 0.0
    wedge: float = 0.0
    action_rate: float = 0.0
    action_accel: float = 0.0
    action_mag: float = 0.0
    done: float = 0.0
    potential: float = 0.0        # approach Phi(s') diagnostic, not summed
    entry_align_potential: float = 0.0  # entry-align Phi(s') diagnostic
    alignment: float = 0.0        # A(s') diagnostic, not summed


def compute_reward(
    *,
    dist_to_seated_m: Optional[float],
    lateral_error_m: Optional[float],
    axis_error_rad: Optional[float],
    roll_error_rad: Optional[float],
    predicted_miss_m: Optional[float],
    outside_port: bool,
    f_peak_n: Optional[float],
    overinsert_m: Optional[float],
    pen_excess_m: Optional[float],
    a_t: np.ndarray,
    a_prev: Optional[np.ndarray],
    a_prev2: Optional[np.ndarray],
    prev_potential: Optional[float],
    prev_entry_align_potential: Optional[float],
    term_status: Optional[str],
    cfg: RewardConfig = RewardConfig(),
) -> tuple[float, RewardBreakdown]:
    """Compute the residual-insertion reward and a weighted breakdown.

    `prev_potential` must be Phi at the previous step's state (or at reset for
    the first step); the caller stores `breakdown.potential` for the next call.
    """
    b = RewardBreakdown()
    b.potential = approach_potential(
        dist_to_seated_m, lateral_error_m, axis_error_rad, roll_error_rad, cfg)
    b.alignment = alignment_score(
        lateral_error_m, axis_error_rad, roll_error_rad, cfg)
    b.entry_align_potential = entry_align_potential(
        lateral_error_m, axis_error_rad, roll_error_rad, cfg)
    prev = float(prev_potential) if prev_potential is not None else b.potential
    b.progress = b.potential - prev
    prev_entry = (
        float(prev_entry_align_potential)
        if prev_entry_align_potential is not None
        else b.entry_align_potential
    )
    b.entry_align = b.entry_align_potential - prev_entry

    b.miss = r_miss(predicted_miss_m, outside_port, cfg)
    b.force = r_force(f_peak_n, cfg)
    b.overinsert = r_overinsert(overinsert_m, cfg)
    b.wedge = r_wedge(pen_excess_m, cfg)
    b.action_rate, b.action_accel, b.action_mag = r_action_smooth(
        a_t, a_prev, a_prev2, cfg)
    b.done = r_done(term_status, cfg)

    total = (b.progress + b.entry_align + b.miss + b.force + b.overinsert + b.wedge
             + b.action_rate + b.action_accel + b.action_mag + b.done)
    return float(total), b


__all__ = [
    "RewardConfig",
    "RewardBreakdown",
    "alignment_score",
    "approach_potential",
    "entry_align_potential",
    "r_miss",
    "r_force",
    "r_overinsert",
    "r_wedge",
    "r_action_smooth",
    "r_done",
    "compute_reward",
]
