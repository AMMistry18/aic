"""LEGACY reward for last-inch insertion (kept verbatim for RL/env.py and
RL/recorded_env.py). The scene env now uses the redesigned RL/reward.py.

The training reward is intentionally small now:

    depth progress + sparse success
    - light pose/contact/action safety costs

At the hard end of the reverse curriculum, the reward adds a small,
level-conditioned progress bonus for safe, aligned motion through the final
depth band. This keeps the global reward scale close to the original run while
giving SAC a denser signal at the 0.85+ frontier.

Image distance is optional and defaults off. The scene env can still render and
log score-style diagnostics, but SAC no longer optimizes a long hand-built
rubric proxy.
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
    """Reward knobs kept deliberately few and interpretable."""

    # Optional image distance. Keep default at zero for geometry-only reward.
    alpha: float = 1.0
    beta_s: float = 0.0
    eps: float = 0.02
    w_image: float = 0.0

    # Main dense signal: positive only for inserting deeper, negative for backing out.
    w_depth: float = 24.0

    # Small per-step shaping/safety costs. These should guide, not dominate.
    xy_ref: float = 0.006
    w_xy: float = 0.08

    axis_free_rad: float = 0.05
    axis_ref_rad: float = 0.15
    w_axis: float = 0.08

    force_free_n: float = 12.0
    force_ref_n: float = 20.0
    w_force: float = 0.04

    lateral_free_n: float = 3.0
    lateral_ref_n: float = 10.0
    w_lateral: float = 0.08

    collision_ref_m: float = 0.002
    w_collision: float = 3.0

    w_action: float = 0.01

    # Extra frontier shaping. This is gated by level, alignment, and contact
    # health, then paid only on positive progress through the final depth band.
    boundary_level: float = 0.85
    boundary_depth_start: float = 0.85
    boundary_progress_bonus: float = 20.0
    boundary_lateral_tol_m: float = 0.005
    boundary_axis_tol_rad: float = math.radians(5.0)
    boundary_penetration_excess_tol_m: float = 0.0015
    boundary_lateral_force_max_n: float = 10.0

    # One-shot terms.
    success_bonus: float = 50.0
    high_level_success_bonus: float = 25.0
    top_level_success_bonus: float = 50.0
    top_level_success_start: float = 0.90
    bad_collision_penalty: float = 12.0
    force_abort_penalty: float = 10.0
    off_limit_penalty: float = 5.0
    timeout_penalty: float = 3.0
    off_limit_event_penalty: float = 5.0
    force_sustain_event_penalty: float = 5.0


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    """Legacy toy-env termination thresholds; scene_env uses its own richer checks."""

    f_abort_n: float = 15.0
    f_linger_n: float = 10.0
    f_linger_sec: float = 0.5
    max_steps: int = 600


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #


def _cap01(x: float, cap: float = 3.0) -> float:
    if not math.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, cap))


def r_image(
    image_curr: np.ndarray,
    image_goal: np.ndarray,
    cfg: RewardConfig,
    bonus_eligible: bool = True,
) -> tuple[float, np.ndarray]:
    """Optional L1 image term. Normally weighted by zero."""
    if image_curr.shape != image_goal.shape:
        raise ValueError(
            f"image_curr {image_curr.shape} != image_goal {image_goal.shape}"
        )
    a = image_curr.astype(np.float32)
    b = image_goal.astype(np.float32)
    norm = float(np.mean(np.abs(a - b)) / 255.0) if a.size else 0.0
    bonus = cfg.beta_s if (bonus_eligible and norm < cfg.eps) else 0.0
    return -cfg.alpha * norm + bonus, np.array(norm, dtype=np.float32)


def r_force(f_z: float, cfg: RewardConfig, depth_norm: float = 1.0) -> float:
    """Axial force is only a safety cost; no force farming bonus."""
    if not math.isfinite(f_z):
        return 0.0
    excess = max(0.0, abs(float(f_z)) - cfg.force_free_n)
    return -_cap01(excess / max(cfg.force_ref_n, 1e-6))


def r_depth(
    depth_norm: Optional[float],
    prev_depth_norm: Optional[float],
    cfg: RewardConfig,
) -> float:
    """Dense insertion progress. Full seated progress is worth about w_depth."""
    if depth_norm is None or prev_depth_norm is None:
        return 0.0
    cur = float(np.clip(depth_norm, 0.0, 1.0))
    prev = float(np.clip(prev_depth_norm, 0.0, 1.0))
    return cfg.w_depth * (cur - prev)


def r_xy(tip_xy: np.ndarray, port_xy: np.ndarray, cfg: RewardConfig) -> float:
    """Small centering cost in the port entrance plane."""
    diff = np.asarray(tip_xy, dtype=np.float32) - np.asarray(port_xy, dtype=np.float32)
    dist = float(np.linalg.norm(diff))
    return -_cap01(dist / max(cfg.xy_ref, 1e-6), cap=6.0)


def r_action(a_t: np.ndarray, a_prev: Optional[np.ndarray], cfg: RewardConfig) -> float:
    """Small smoothness cost on residual action changes."""
    if a_prev is None:
        return 0.0
    diff = np.asarray(a_t, dtype=np.float32) - np.asarray(a_prev, dtype=np.float32)
    return -_cap01(float(np.linalg.norm(diff)), cap=4.0)


def r_lateral(f_xy: np.ndarray, cfg: RewardConfig) -> float:
    """Small side-load cost."""
    if f_xy is None or len(f_xy) == 0:
        return 0.0
    mag = float(np.linalg.norm(np.asarray(f_xy, dtype=np.float32)))
    excess = max(0.0, mag - cfg.lateral_free_n)
    return -_cap01(excess / max(cfg.lateral_ref_n, 1e-6))


def r_axis(axis_error_rad: Optional[float], cfg: RewardConfig) -> float:
    """Small cost for a bent plug axis."""
    if axis_error_rad is None or not math.isfinite(axis_error_rad):
        return 0.0
    excess = max(0.0, float(axis_error_rad) - cfg.axis_free_rad)
    return -_cap01(excess / max(cfg.axis_ref_rad, 1e-6), cap=2.0)


def r_collision(penetration_excess_m: Optional[float], cfg: RewardConfig) -> float:
    """Sharper cost for plug-port interpenetration beyond the calibrated baseline."""
    if penetration_excess_m is None or not math.isfinite(penetration_excess_m):
        return 0.0
    excess = max(0.0, float(penetration_excess_m))
    return -_cap01(excess / max(cfg.collision_ref_m, 1e-6), cap=4.0)


def r_boundary_progress(
    *,
    depth_norm: Optional[float],
    prev_depth_norm: Optional[float],
    curriculum_level: Optional[float],
    lateral_error_m: Optional[float],
    align_error_rad: Optional[float],
    penetration_excess_m: Optional[float],
    lateral_force_n: Optional[float],
    cfg: RewardConfig,
) -> float:
    """Safe, level-conditioned progress bonus for the 0.85+ frontier.

    The potential is normalized over the final depth band, so the maximum extra
    return for moving cleanly from boundary_depth_start to seated is roughly
    boundary_progress_bonus. Staying still receives zero; backing out is left to
    the base depth term rather than double-punished.
    """
    if curriculum_level is None or float(curriculum_level) < cfg.boundary_level:
        return 0.0
    if depth_norm is None or prev_depth_norm is None:
        return 0.0

    lateral = float("inf") if lateral_error_m is None else abs(float(lateral_error_m))
    align = float("inf") if align_error_rad is None else abs(float(align_error_rad))
    pen = 0.0 if penetration_excess_m is None else max(0.0, float(penetration_excess_m))
    lat_force = 0.0 if lateral_force_n is None else abs(float(lateral_force_n))
    if (lateral > cfg.boundary_lateral_tol_m
            or align > cfg.boundary_axis_tol_rad
            or pen > cfg.boundary_penetration_excess_tol_m
            or lat_force > cfg.boundary_lateral_force_max_n):
        return 0.0

    lo = float(np.clip(cfg.boundary_depth_start, 0.0, 0.999))
    span = max(1.0 - lo, 1e-6)

    def potential(x: float) -> float:
        return float(np.clip((float(x) - lo) / span, 0.0, 1.0))

    delta = potential(depth_norm) - potential(prev_depth_norm)
    return cfg.boundary_progress_bonus * max(0.0, delta)


def r_level_success(term_status: Optional[str],
                    curriculum_level: Optional[float],
                    cfg: RewardConfig) -> float:
    """Extra sparse credit for clean successes at the hard curriculum frontier."""
    if term_status != "success" or curriculum_level is None:
        return 0.0
    level = float(curriculum_level)
    if level >= cfg.top_level_success_start:
        return cfg.top_level_success_bonus
    if level >= cfg.boundary_level:
        return cfg.high_level_success_bonus
    return 0.0


def r_done(term_status: Optional[str],
           cfg: Optional[RewardConfig] = None,
           time_frac: Optional[float] = None) -> float:
    """Sparse terminal outcome."""
    c = cfg or RewardConfig()
    if term_status == "success":
        return c.success_bonus
    if term_status == "bad_collision":
        return -c.bad_collision_penalty
    if term_status == "force_abort":
        return -c.force_abort_penalty
    if term_status == "off_limit":
        return -c.off_limit_penalty
    if term_status == "timeout":
        return -c.timeout_penalty
    return 0.0


def r_events(off_limit_event: bool, force_sustain_event: bool,
             cfg: RewardConfig) -> tuple[float, float]:
    """One-shot non-terminal safety events."""
    off = -cfg.off_limit_event_penalty if off_limit_event else 0.0
    force = -cfg.force_sustain_event_penalty if force_sustain_event else 0.0
    return off, force


# --------------------------------------------------------------------------- #
# total
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class RewardBreakdown:
    """Weighted per-component contributions."""

    image: float = 0.0
    force: float = 0.0
    depth: float = 0.0
    xy: float = 0.0
    action: float = 0.0
    lateral: float = 0.0
    axis: float = 0.0
    collision: float = 0.0
    boundary: float = 0.0
    level_success: float = 0.0
    off_limit: float = 0.0
    force_sustain: float = 0.0
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
    depth_norm: Optional[float] = None,
    prev_depth_norm: Optional[float] = None,
    axis_error_rad: Optional[float] = None,
    roll_error_rad: Optional[float] = None,
    penetration_excess_m: Optional[float] = None,
    off_limit_event: bool = False,
    force_sustain_event: bool = False,
    success_time_frac: Optional[float] = None,
    curriculum_level: Optional[float] = None,
) -> tuple[float, RewardBreakdown]:
    """Compute the simple reward and a weighted breakdown."""
    b = RewardBreakdown()
    img_raw, b.image_l1_norm = r_image(image_curr, image_goal, cfg,
                                       bonus_eligible=bonus_eligible)

    dnorm = 1.0 if depth_norm is None else float(depth_norm)
    align_error = axis_error_rad
    if roll_error_rad is not None and math.isfinite(float(roll_error_rad)):
        if align_error is None or not math.isfinite(float(align_error)):
            align_error = float(roll_error_rad)
        else:
            align_error = max(float(align_error), float(roll_error_rad))
    b.image = cfg.w_image * img_raw
    b.depth = r_depth(depth_norm, prev_depth_norm, cfg)
    b.xy = cfg.w_xy * r_xy(tip_xy, port_xy, cfg)
    b.axis = cfg.w_axis * r_axis(align_error, cfg)
    b.force = cfg.w_force * r_force(f_z, cfg, depth_norm=dnorm)
    b.lateral = cfg.w_lateral * r_lateral(f_xy, cfg)
    b.collision = cfg.w_collision * r_collision(penetration_excess_m, cfg)
    b.boundary = r_boundary_progress(
        depth_norm=depth_norm,
        prev_depth_norm=prev_depth_norm,
        curriculum_level=curriculum_level,
        lateral_error_m=float(np.linalg.norm(np.asarray(tip_xy, dtype=np.float32))),
        align_error_rad=align_error,
        penetration_excess_m=penetration_excess_m,
        lateral_force_n=float(np.linalg.norm(np.asarray(f_xy, dtype=np.float32))),
        cfg=cfg,
    )
    b.action = cfg.w_action * r_action(a_t, a_prev, cfg)
    b.off_limit, b.force_sustain = r_events(off_limit_event, force_sustain_event, cfg)
    b.done = r_done(term_status, cfg, success_time_frac)
    b.level_success = r_level_success(term_status, curriculum_level, cfg)

    total = (b.image + b.depth + b.xy + b.axis + b.force + b.lateral
             + b.collision + b.boundary + b.action + b.off_limit
             + b.force_sustain + b.done + b.level_success)
    return float(total), b


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
    """Legacy toy-env termination helper."""
    if not bypass_image_success and image_l1_norm < cfg_rew.eps:
        return "success"
    if abs(f_z) > cfg_term.f_abort_n:
        return "force_abort"
    if off_limit_contact:
        return "off_limit"
    if abs(f_z) > cfg_term.f_linger_n and f_linger_dwell_s >= cfg_term.f_linger_sec:
        return "force_abort"
    if step >= cfg_term.max_steps:
        return "timeout"
    return None


__all__ = [
    "RewardConfig",
    "TerminationConfig",
    "RewardBreakdown",
    "r_image",
    "r_force",
    "r_depth",
    "r_xy",
    "r_action",
    "r_lateral",
    "r_axis",
    "r_collision",
    "r_boundary_progress",
    "r_level_success",
    "r_events",
    "r_done",
    "compute_reward",
    "check_termination",
]
