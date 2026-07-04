"""
Reward function for image-SAC on the last-inch cable insertion task.

Implements the spec in REWARD_SPEC.md §2-§6, recalibrated for the *corrected*
MuJoCo scene (v0.3). All components are additive; the caller combines them with
the weights in RewardConfig (§3).

The reward is intentionally framework-agnostic: it consumes plain numpy arrays /
floats, so it works for SB3, RLlib, skrl, CleanRL, or hand-rolled SAC. No
PyTorch / Gym dependency at import time.

v0.2 → v0.3 (after the scene fix that made insertion + contact force real):
  - Killed the DOUBLE +50 success bonus: `beta_s` (image bonus) is now 0; the
    single terminal success bonus lives in `r_done` (+50). Previously the agent
    got +100 at the success step.
  - Recalibrated the force band to the MEASURED sim contact force (0-8 N,
    seating 3-5 N) instead of the old 2-20 N (which was never realized, so the
    term was a flat constant). f_low/f_optimal/f_max = 0.5/4.0/8.0 N.
  - Added `r_depth`: a POTENTIAL-BASED insertion-depth progress term
    (Ng et al. 1999). Telescopes to ~+w_depth over a full insertion, is
    policy-invariant, and gives a low-variance last-inch signal that survives
    the image term's near-goal plateau / wrist-cam occlusion.
  - Fixed the numerically-dead scaling of `r_xy` and `r_lateral` (distances
    were in metres, so the old terms were ~1e-4 — effectively off). Both are
    now normalised to O(1).
  - Terminal penalties softened (-10 not -30) since contact is now REQUIRED for
    success and large abort spikes teach contact-phobia.
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
    """All reward hyperparameters in one place. Defaults from REWARD_SPEC §8."""

    # §2.1 image
    alpha: float = 1.0           # image L1 coefficient (primary dense signal)
    beta_s: float = 0.0          # image success bonus — 0: the +50 lives in r_done
    eps: float = 0.02            # image-L1 success threshold (guard only)

    # §2.2 force piecewise log — calibrated to the MEASURED 0-8 N sim band
    f_low: float = 0.5           # below = no meaningful contact
    f_optimal: float = 4.0       # middle of the measured 3-5 N seating band
    f_max: float = 8.0           # top of the normal seating band; penalise beyond
    beta_low: float = 0.5
    gamma: float = 0.5           # force ramp slope (NOT the RL discount)
    force_depth_gate: float = 0.3  # positive force branch only when depth_norm>gate

    # §2.2b insertion-depth progress (potential-based)
    w_depth: float = 2.0         # weight on the depth potential term
    disc_gamma: float = 0.99     # discount used for potential shaping (match SAC's)

    # §2.3 tip–port xy progress (normalised)
    delta_xy: float = 0.5
    xy_ref: float = 0.005        # 5 mm normaliser so the term is O(1), not O(1e-3)

    # §2.4 action smoothness
    delta_a: float = 0.1

    # §2.5 lateral force penalty (per Newton)
    delta_lat: float = 0.05

    # §2.7 geometry validity penalties (scene env only; 0 when signals omitted)
    axis_free_rad: float = 0.03       # ~1.7 deg: straight enough is free
    axis_ref_rad: float = 0.12        # ~6.9 deg beyond free => O(1) penalty
    collision_ref_m: float = 0.002    # 2 mm excess penetration => O(1) penalty

    # §3 component weights in total
    w_image: float = 1.0
    w_force: float = 0.05
    w_xy: float = 0.05
    w_action: float = 0.5        # eff. smoothness coef = w_action*delta_a = 0.05
    w_lateral: float = 1.0       # eff. lateral coef = w_lateral*delta_lat = 0.05
    w_axis: float = 3.0
    w_collision: float = 12.0


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    """Termination thresholds for the env wrapper. Calibrated to the 0-8 N band."""

    f_abort_n: float = 15.0      # instant force abort (well above 8 N seating)
    f_linger_n: float = 10.0     # sustained-force threshold
    f_linger_sec: float = 0.5    # sustained-force dwell time
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
    """Dense image reward. Returns (scalar, l1_normalised) so the caller can
    re-use the L1 for termination checks.

    The `+β_s` bonus defaults to 0 (the success bonus now lives in r_done). If
    a non-zero β_s is configured, it is only added when `bonus_eligible` — set
    by the env only at the true geometric-success step.
    """
    if image_curr.shape != image_goal.shape:
        raise ValueError(
            f"image_curr {image_curr.shape} != image_goal {image_goal.shape}"
        )
    a = image_curr.astype(np.float32)
    b = image_goal.astype(np.float32)
    l1 = float(np.sum(np.abs(a - b)))
    norm = l1 / max(a.size, 1)
    bonus = cfg.beta_s if (bonus_eligible and norm < cfg.eps) else 0.0
    scalar = -cfg.alpha * norm + bonus
    return scalar, np.array(norm, dtype=np.float32)


def r_force(f_z: float, cfg: RewardConfig, depth_norm: float = 1.0) -> float:
    """Piecewise-logarithmic force reward from §2.2, calibrated to 0-8 N.

    f_z is signed; we use |f_z| because contact along the insertion axis (in
    either sign convention) counts. `depth_norm` gates the *positive* (contact-
    reward) branch: below `force_depth_gate` only the penalty branch applies, so
    the agent cannot farm the force bonus by leaning on the port entrance
    without inserting.
    """
    if not math.isfinite(f_z):
        return 0.0
    f = abs(float(f_z))

    log_low = cfg.beta_low * math.log1p(cfg.f_low)
    peak = log_low + cfg.gamma * (cfg.f_optimal - cfg.f_low)
    gated = depth_norm >= cfg.force_depth_gate

    if f < cfg.f_low:
        val = cfg.beta_low * math.log1p(f)
    elif f <= cfg.f_optimal:
        val = log_low + cfg.gamma * (f - cfg.f_low)
    elif f < cfg.f_max:
        val = peak - cfg.gamma * (f - cfg.f_optimal)
    else:
        delta = f - cfg.f_max
        val = peak - cfg.gamma * (cfg.f_max - cfg.f_optimal) - math.log1p(delta) ** 2

    if not gated:
        # suppress any positive (bonus) contribution before real insertion
        val = min(val, 0.0)
    return val


def r_depth(
    depth_norm: Optional[float],
    prev_depth_norm: Optional[float],
    cfg: RewardConfig,
) -> float:
    """Potential-based insertion-depth progress (Ng et al. 1999).

    Φ(s) = clip(depth_norm, 0, 1), reward = w_depth * (γ·Φ(s') − Φ(s)).
    Telescopes to ~+w_depth over a full insertion, is policy-invariant, and
    penalises backing out. Returns 0 if depth info is unavailable (keeps the
    standalone unit test working without depth signals).
    """
    if depth_norm is None or prev_depth_norm is None:
        return 0.0
    phi_next = float(np.clip(depth_norm, 0.0, 1.0))
    phi_prev = float(np.clip(prev_depth_norm, 0.0, 1.0))
    return cfg.w_depth * (cfg.disc_gamma * phi_next - phi_prev)


def r_xy(tip_xy: np.ndarray, port_xy: np.ndarray, cfg: RewardConfig) -> float:
    """Tip-XY minus port-entrance XY, normalised by `xy_ref` so it is O(1).

    A small centring nudge; the heavy lifting is done by the image term and
    the depth potential, so this stays a minor auxiliary.
    """
    diff = np.asarray(tip_xy, dtype=np.float32) - np.asarray(port_xy, dtype=np.float32)
    dist = float(np.linalg.norm(diff))
    return float(-cfg.delta_xy * min(dist / cfg.xy_ref, 1.0))


def r_action(a_t: np.ndarray, a_prev: Optional[np.ndarray], cfg: RewardConfig) -> float:
    """Smoothness penalty on residual action difference (jerk is scored)."""
    if a_prev is None:
        return 0.0
    diff = np.asarray(a_t, dtype=np.float32) - np.asarray(a_prev, dtype=np.float32)
    return float(-cfg.delta_a * float(np.linalg.norm(diff)))


def r_lateral(f_xy: np.ndarray, cfg: RewardConfig) -> float:
    """Linear penalty on lateral force magnitude (side-loading bends cables)."""
    if f_xy is None or len(f_xy) == 0:
        return 0.0
    mag = float(np.linalg.norm(np.asarray(f_xy, dtype=np.float32)))
    return float(-cfg.delta_lat * mag)


def r_axis(axis_error_rad: Optional[float], cfg: RewardConfig) -> float:
    """Penalty for inserting with the plug axis bent away from the port axis."""
    if axis_error_rad is None or not math.isfinite(axis_error_rad):
        return 0.0
    excess = max(0.0, float(axis_error_rad) - cfg.axis_free_rad)
    return -min(excess / max(cfg.axis_ref_rad, 1e-6), 3.0)


def r_collision(penetration_excess_m: Optional[float], cfg: RewardConfig) -> float:
    """Penalty for excess plug-port penetration beyond the calibrated seated baseline."""
    if penetration_excess_m is None or not math.isfinite(penetration_excess_m):
        return 0.0
    excess = max(0.0, float(penetration_excess_m))
    return -min(excess / max(cfg.collision_ref_m, 1e-6), 4.0)


def r_done(term_status: Optional[str]) -> float:
    """One-shot termination bonus/penalty (§2.6). This IS the success bonus."""
    if term_status == "success":
        return 50.0
    if term_status == "bad_collision":
        return -25.0
    if term_status == "force_abort":
        return -10.0
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
    """Per-component reward for the step (post-weight). Sum into reward_total
    is done by compute_reward; each field is the WEIGHTED contribution so the
    logger can show what actually moved the total."""

    image: float = 0.0
    force: float = 0.0
    depth: float = 0.0
    xy: float = 0.0
    action: float = 0.0
    lateral: float = 0.0
    axis: float = 0.0
    collision: float = 0.0
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
    penetration_excess_m: Optional[float] = None,
) -> tuple[float, RewardBreakdown]:
    """Compute every component and return the weighted total + breakdown.

    `bonus_eligible` gates the (now default-off) +β_s image bonus.
    `depth_norm` / `prev_depth_norm` (insertion fraction in [0,1], current and
    previous step) drive the potential-based depth term and the force gate;
    pass both from the env. If omitted, r_depth is 0 and the force gate opens.

    NOTE on breakdown: fields store the WEIGHTED contribution (component ×
    weight), so `sum(fields) == total`. This differs from v0.2 where fields
    were pre-weight; the logger/dashboard sums these directly.
    """
    b = RewardBreakdown()
    img_raw, b.image_l1_norm = r_image(image_curr, image_goal, cfg,
                                       bonus_eligible=bonus_eligible)
    dnorm = 1.0 if depth_norm is None else float(depth_norm)
    force_raw = r_force(f_z, cfg, depth_norm=dnorm)

    b.image = cfg.w_image * img_raw
    b.force = cfg.w_force * force_raw
    b.depth = r_depth(depth_norm, prev_depth_norm, cfg)   # already weighted
    b.xy = cfg.w_xy * r_xy(tip_xy, port_xy, cfg)
    b.action = cfg.w_action * r_action(a_t, a_prev, cfg)
    b.lateral = cfg.w_lateral * r_lateral(f_xy, cfg)
    b.axis = cfg.w_axis * r_axis(axis_error_rad, cfg)
    b.collision = cfg.w_collision * r_collision(penetration_excess_m, cfg)
    b.done = r_done(term_status)                          # unweighted one-shot

    total = (b.image + b.force + b.depth + b.xy + b.action + b.lateral
             + b.axis + b.collision + b.done)
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

    `bypass_image_success=True` skips the image-L1 success check (the env uses
    the more reliable geometric seated-and-in-contact criterion). Keep it True
    for the last-inch env — the goal image also matches an empty cavity.
    """
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
    "r_done",
    "compute_reward",
    "check_termination",
]
