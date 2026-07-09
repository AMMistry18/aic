"""Validate the current residual-insertion reward and base-script gate.

Run in-container:
    MUJOCO_GL=egl pixi run python RL/scripts/validate_reward_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL.reward import (  # noqa: E402
    RewardConfig,
    alignment_score,
    approach_potential,
    compute_reward,
    entry_align_potential,
    r_force,
    r_miss,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def unit_probes() -> None:
    cfg = RewardConfig()
    print("== unit probes ==")

    good = alignment_score(0.001, 0.02, 0.02, cfg)
    bad = alignment_score(0.010, 0.12, 0.12, cfg)
    check("alignment kernel separates good/bad", good > 0.8 and bad < 0.01,
          f"good={good:.3f} bad={bad:.5f}")

    phi_far = approach_potential(0.055, 0.001, 0.02, 0.02, cfg)
    phi_near = approach_potential(0.035, 0.001, 0.02, 0.02, cfg)
    phi_bad_near = approach_potential(0.035, 0.010, 0.12, 0.12, cfg)
    check("aligned approach potential grows with depth", phi_near > phi_far,
          f"{phi_far:.2f}->{phi_near:.2f}")
    check("misaligned depth has almost no progress potential", phi_bad_near < 0.2,
          f"{phi_bad_near:.3f}")

    ea_bad = entry_align_potential(0.010, 0.12, 0.12, cfg)
    ea_good = entry_align_potential(0.001, 0.02, 0.02, cfg)
    check("entry alignment potential rewards lining up", ea_good - ea_bad > 4.0,
          f"{ea_bad:.3f}->{ea_good:.3f}")

    miss_clear = r_miss(0.001, True, cfg)
    miss_full = r_miss(0.020, True, cfg)
    check("predicted miss is zero inside free clearance", miss_clear == 0.0)
    check("predicted miss reaches full side-hit cost", abs(miss_full + cfg.w_miss) < 1e-6,
          f"{miss_full:+.3f}")

    check("force is free at/below 20 N", r_force(20.0, cfg) == 0.0)
    check("force guard is sharp at 25 N",
          abs(r_force(25.0, cfg) + cfg.w_force) < 1e-6,
          f"{r_force(25.0, cfg):+.3f}")

    total, b = compute_reward(
        dist_to_seated_m=0.040,
        lateral_error_m=0.002,
        axis_error_rad=0.03,
        roll_error_rad=0.03,
        predicted_miss_m=0.002,
        outside_port=True,
        f_peak_n=10.0,
        overinsert_m=0.0,
        pen_excess_m=0.0,
        a_t=np.zeros(6, np.float32),
        a_prev=np.zeros(6, np.float32),
        a_prev2=np.zeros(6, np.float32),
        prev_potential=0.0,
        prev_entry_align_potential=0.0,
        term_status=None,
        cfg=cfg,
    )
    parts = (
        b.progress + b.entry_align + b.miss + b.force + b.overinsert + b.wedge
        + b.action_rate + b.action_accel + b.action_mag + b.done
    )
    check("total == sum(breakdown)", abs(total - parts) < 1e-6, f"{total:+.3f}")


def sim_probes() -> None:
    from RL.scene_env import SceneEnvConfig, SceneInsertEnv

    print("== in-sim probes (real scene) ==")
    env = SceneInsertEnv(SceneEnvConfig(
        include_images=False,
        privileged_obs=True,
        action_mode="cartesian_residual",
        base_script_enabled=True,
        start_curriculum_enabled=False,
    ))
    action = np.zeros(env.action_space.shape, np.float32)

    obs, _ = env.reset(seed=3, options={
        "retract_m": env.cfg.seated_depth_m + 0.020,
        "jitter": False,
    })
    _obs, _r, _term, _trunc, info = env.step(action)
    check("centered reset opens the base-script gate",
          info.get("cart_base_gate", 0.0) == 1.0 and info.get("cart_base_adv_mm", 0.0) > 0.0,
          f"gate={info.get('cart_base_gate')} adv={info.get('cart_base_adv_mm')}")

    obs, _ = env.reset(seed=1, options={"rsi": False})
    start_lat = env._geometry_diag()["lateral_error_m"]
    _obs, _r, _term, _trunc, info = env.step(action)
    if start_lat > env.cfg.base_script_gate_lat_m:
        check("misaligned reset pauses base-script advance",
              info.get("cart_base_gate", 1.0) == 0.0
              and info.get("cart_base_adv_mm", 1.0) == 0.0,
              f"lat={start_lat*1000:.1f}mm gate={info.get('cart_base_gate')}")
    else:
        print("  [SKIP] misaligned gate probe: sampled start was already aligned")

    env.close()


def main() -> int:
    unit_probes()
    sim_probes()
    print("\nVERDICT:", "PASS" if not FAILURES else f"FAIL ({len(FAILURES)}): {FAILURES}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
