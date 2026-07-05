"""Validate the geometry-first reward (RL/reward.py, 2026-07-04 rewrite).

Two layers:

1. UNIT probes (pure numpy): every component has the sign/caps/one-shot
   semantics the design table claims — depth is the only positive dense term,
   penalties are capped, oscillating in/out cannot farm reward under the SAC
   discount, terminal bonuses dominate accumulated shaping.

2. IN-SIM probes (real scene, deterministic): scripted straight-in insertion
   from mid-curriculum must earn ~w_depth * depth-gained + success bonus;
   sitting still must earn ~0; backing out must lose what insertion gains.

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

from RL.reward import (RewardConfig, compute_reward, r_action, r_axis,  # noqa: E402
                       r_collision, r_depth, r_done, r_events, r_force,
                       r_lateral, r_xy)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def unit_probes() -> None:
    cfg = RewardConfig()
    img = np.zeros((1, 1, 3), np.float32)
    print("== unit probes ==")

    # depth: only positive dense term, symmetric, telescoping
    check("depth + for inserting", r_depth(0.6, 0.5, cfg) > 0,
          f"{r_depth(0.6, 0.5, cfg):+.2f}")
    check("depth - for backing out", r_depth(0.4, 0.5, cfg) < 0)
    check("depth 0 when still", r_depth(0.5, 0.5, cfg) == 0.0)
    path = [0.5, 0.3, 0.5, 0.9, 1.0]           # out, back, then in
    tele = sum(r_depth(b, a, cfg) for a, b in zip(path, path[1:]))
    check("depth telescopes to net gain", abs(tele - cfg.w_depth * 0.5) < 1e-6,
          f"sum={tele:.2f} vs {cfg.w_depth * 0.5:.2f}")
    # discounted farming test: leave-and-return is strictly unprofitable at gamma<1
    gamma, r_out, r_back = 0.99, r_depth(0.3, 0.5, cfg), r_depth(0.5, 0.3, cfg)
    check("no in/out farming under discount", r_out + gamma * r_back < 0,
          f"{r_out + gamma * r_back:+.3f}")

    # every non-depth dense term is <= 0 (nothing to farm)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        vals = [
            cfg.w_xy * r_xy(rng.normal(0, 0.01, 2), np.zeros(2), cfg),
            cfg.w_axis * r_axis(abs(rng.normal(0, 0.3)), cfg),
            cfg.w_force * r_force(rng.normal(0, 40), cfg),
            cfg.w_lateral * r_lateral(rng.normal(0, 20, 2), cfg),
            cfg.w_collision * r_collision(abs(rng.normal(0, 0.005)), cfg),
            cfg.w_action * r_action(rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6), cfg),
        ]
        worst = max(worst, max(vals))
    check("all non-depth dense terms <= 0", worst <= 0.0, f"max seen {worst:+.4f}")

    # caps: worst-case per-step dense penalty is bounded and small vs success
    worst_step = (cfg.w_xy * 2 + cfg.w_axis * 2 + cfg.w_force * 1
                  + cfg.w_lateral * 1 + cfg.w_collision * 4 + cfg.w_action * 4)
    check("worst-case dense step penalty bounded", worst_step < cfg.success_bonus / 5,
          f"worst {worst_step:.2f}/step vs success +{cfg.success_bonus}")
    # NOTE: w_collision*cap dominates that bound, but bad_collision terminates at
    # 3mm excess (= penalty 3.0), so the realized bound is ~3.4/step for <=few steps.

    # free zones: gentle contact is not punished
    check("axial force free below 12 N", r_force(11.0, cfg) == 0.0)
    check("lateral force free below 3 N", r_lateral(np.array([2.9, 0.0]), cfg) == 0.0)

    # terminal + events
    check("success +50", r_done("success", cfg) == cfg.success_bonus)
    check("force_abort -10", r_done("force_abort", cfg) == -cfg.force_abort_penalty)
    check("bad_collision -10", r_done("bad_collision", cfg) == -cfg.bad_collision_penalty)
    off, fs = r_events(True, True, cfg)
    check("events -5/-5 one-shot values", off == -5.0 and fs == -5.0)

    # compute_reward totals == sum of breakdown, image inert at w_image=0.
    # Inputs = a REALISTIC clean insertion step (measured seating band: ~5 N
    # axial, ~3 N lateral, ~0.2 mm excess penetration, 1.5 mm lateral error).
    tot, b = compute_reward(
        image_curr=img, image_goal=img, f_z=5.0, f_xy=np.array([3.5, 0.0]),
        tip_xy=np.array([0.0015, 0.0]), port_xy=np.zeros(2),
        a_t=np.ones(6, np.float32) * 0.5, a_prev=np.zeros(6, np.float32),
        term_status=None, cfg=cfg, depth_norm=0.6, prev_depth_norm=0.55,
        axis_error_rad=0.06, penetration_excess_m=0.0002)
    parts = (b.image + b.depth + b.xy + b.axis + b.force + b.lateral
             + b.collision + b.action + b.off_limit + b.force_sustain + b.done)
    check("total == sum(breakdown)", abs(tot - parts) < 1e-6, f"{tot:+.3f}")
    check("image term inert (w=0)", b.image == 0.0)
    check("clean insertion step is net positive", tot > 0.5,
          f"{tot:+.3f} (depth {b.depth:+.2f} vs costs {parts - b.depth - b.done:+.2f})")
    # same progress but WEDGED (2 mm excess pen, heavy side-load): must be net
    # negative — pushing through a jam should never pay
    tot_bad, _ = compute_reward(
        image_curr=img, image_goal=img, f_z=40.0, f_xy=np.array([25.0, 0.0]),
        tip_xy=np.array([0.005, 0.0]), port_xy=np.zeros(2),
        a_t=np.ones(6, np.float32) * 0.5, a_prev=np.zeros(6, np.float32),
        term_status=None, cfg=cfg, depth_norm=0.6, prev_depth_norm=0.55,
        axis_error_rad=0.15, penetration_excess_m=0.002)
    check("wedged push is net negative", tot_bad < 0, f"{tot_bad:+.3f}")


def sim_probes() -> None:
    import mujoco
    from RL.scene_env import SceneEnvConfig, SceneInsertEnv
    print("== in-sim probes (real scene) ==")
    env = SceneInsertEnv(SceneEnvConfig(include_images=False))
    cfg = env.cfg.reward

    def run_scripted(level: float, target_tip, steps: int, label: str):
        env.set_curriculum_level(level)
        env._reset_to_level(level, jitter=False)
        env._step_count = 0
        env._last_action = np.zeros(6, np.float32)
        env._prev_depth_norm = env._depth_norm()
        env._f_ax_buf = []
        env._off_limit_event_fired = False
        env._force_sustain_event_fired = False
        env._reset_score_state()
        d0 = env._depth_norm()
        q_goal, _, _, _ = env._ik_tip_axis(target_tip, env._insert_axis, env._arm_target)
        tot, dep, done_r, term = 0.0, 0.0, 0.0, None
        for _ in range(steps):
            delta = (q_goal - env._arm_target) / max(env.cfg.action_joint_scale, 1e-9)
            a = np.clip(delta, -1.0, 1.0).astype(np.float32)
            _, r, t, tr, info = env.step(a)
            b = info["breakdown"]
            tot += r; dep += b.depth; done_r += b.done
            term = info["term_status"]
            if t or tr:
                break
        d1 = env._depth_norm()
        print(f"  {label}: depth {d0:.2f}->{d1:.2f}  sum_r={tot:+.1f}  "
              f"sum_depth={dep:+.1f}  done={done_r:+.0f}  term={term}")
        return d0, d1, tot, dep, done_r, term

    # 1. scripted straight-in insertion from mid-curriculum (in-cage start)
    d0, d1, tot, dep, done_r, term = run_scripted(
        0.5, env._inserted_tip, 120, "insert lvl 0.5")
    expected = cfg.w_depth * (d1 - d0)
    check("insertion: depth sum ~ w_depth*gain", abs(dep - expected) < 2.0,
          f"{dep:+.1f} vs {expected:+.1f}")
    check("insertion: reaches success (+50)", term == "success" and done_r == cfg.success_bonus)
    check("insertion: total return strongly positive", tot > 30, f"{tot:+.1f}")

    # 2. scripted approach + insertion from OUTSIDE the cage
    d0, d1, tot, dep, done_r, term = run_scripted(
        0.8, env._inserted_tip, 200, "insert lvl 0.8 (outside)")
    check("outside-in: net positive progress reward", dep > 5.0, f"{dep:+.1f}")
    check("outside-in: no spurious abort", term in ("success", None), f"term={term}")

    # 3. sit-still: reward ~ 0 (no free lunch, no harsh ambient penalty)
    env.set_curriculum_level(0.3)
    env._reset_to_level(0.3, jitter=False)
    env._step_count = 0; env._last_action = np.zeros(6, np.float32)
    env._prev_depth_norm = env._depth_norm(); env._f_ax_buf = []
    env._off_limit_event_fired = env._force_sustain_event_fired = False
    env._reset_score_state()
    rs = []
    for _ in range(30):
        _, r, t, tr, info = env.step(np.zeros(6, np.float32))
        rs.append(r)
        if t or tr:
            break
    check("sit-still: |mean step reward| small", abs(float(np.mean(rs))) < 0.5,
          f"mean {np.mean(rs):+.3f} term={info['term_status']}")

    # 4. back-out: loses roughly what insertion gains (no exploit)
    retract_tip = env._inserted_tip + 0.055 * env._retract_dir
    d0, d1, tot, dep, done_r, term = run_scripted(0.15, retract_tip, 80, "back out lvl 0.15")
    check("back-out: depth sum negative", dep < -2.0, f"{dep:+.1f}")
    env.close()


def main() -> int:
    unit_probes()
    sim_probes()
    print("\nVERDICT:", "PASS" if not FAILURES else f"FAIL ({len(FAILURES)}): {FAILURES}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
