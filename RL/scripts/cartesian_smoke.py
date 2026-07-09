"""No-learning scripted smoke test for the `cartesian_residual` action mode.

Verifies the Cartesian impedance (Jacobian-transpose) controller branch of
RL/scene_env.py without any training:

  1. HOLD      zero action for 20 steps -> TCP drift stays small, no spikes.
  2. TRANSLATE +1 mm/step residuals along port-frame x, y, z (10 steps each)
               -> actual TCP motion matches the commanded direction (cosine)
               and rough magnitude, with bounded wrench/torque/contact force.
  3. ROTATE    +1 deg/step residuals about port-frame x (roll), y (pitch)
               [and z (yaw) with 6-D actions] -> same direction/magnitude and
               boundedness checks on the achieved TCP rotation.
  4. BASESCRIPT (--base-script) zero residuals; the scripted base target pushes
               along the insertion axis -> insertion depth must increase with
               no force_abort/bad_collision termination.

Run from the repository's Pixi environment:
    MUJOCO_GL=egl pixi run python RL/scripts/cartesian_smoke.py

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL.scene_env import SceneEnvConfig, SceneInsertEnv  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


def _tcp_pose(env):
    return env.data.site_xpos[env._tcp_sid].copy(), env._site_quat()


def _run_steps(env, action, n):
    """Step `action` n times; return per-step maxima pulled from env/info."""
    stats = dict(force_n=0.0, wrench_f=0.0, wrench_t=0.0, tau=0.0,
                 clip=0, term=None)
    for _ in range(n):
        _, _, term, trunc, info = env.step(action)
        stats["force_n"] = max(stats["force_n"],
                               float(np.linalg.norm(env._contact_force())))
        stats["wrench_f"] = max(stats["wrench_f"], info.get("cart_wrench_force_n", 0.0))
        stats["wrench_t"] = max(stats["wrench_t"], info.get("cart_wrench_torque_nm", 0.0))
        stats["tau"] = max(stats["tau"], info.get("cart_tau_max_nm", 0.0))
        stats["clip"] = int(info.get("cart_clip_events", 0))
        stats["info"] = info
        if term or trunc:
            stats["term"] = info.get("term_status")
            break
    return stats


def _probe(env, level, axis_idx, translate, scale, n_steps, settle_steps, dims):
    """Command a single-axis residual burst; measure achieved port-frame motion."""
    env.reset(seed=0, options={"level": level, "jitter": False})
    R = env._cart_frame_R                      # columns = port-frame axes in world
    p0, q0 = _tcp_pose(env)
    a = np.zeros(dims, np.float32)
    a[axis_idx if translate else 3 + axis_idx] = 1.0
    stats = _run_steps(env, a, n_steps)
    if stats["term"] is None:                  # let the impedance tracking settle
        settle = _run_steps(env, np.zeros(dims, np.float32), settle_steps)
        for k in ("force_n", "wrench_f", "wrench_t", "tau"):
            stats[k] = max(stats[k], settle[k])
        stats["term"] = settle["term"]
    p1, q1 = _tcp_pose(env)
    if translate:
        achieved = R.T @ (p1 - p0)
    else:
        rotvec_world = SceneInsertEnv._quat_to_rotvec(
            env._qmul(q1, env._qinv(q0)))
        achieved = R.T @ rotvec_world
    commanded = n_steps * scale
    unit = np.zeros(3)
    unit[axis_idx] = 1.0
    mag = float(np.linalg.norm(achieved))
    cosine = float(np.dot(achieved, unit) / max(mag, 1e-12))
    ratio = mag / commanded
    nan_free = bool(np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qacc)))
    return dict(achieved=achieved, cosine=cosine, ratio=ratio, nan_free=nan_free, **stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", type=float, default=1.0,
                    help="curriculum level for the probe resets (1.0 = fully "
                         "retracted, tip in free space outside the cage)")
    ap.add_argument("--steps", type=int, default=10, help="probe steps per axis")
    ap.add_argument("--settle", type=int, default=80,
                    help="zero-action settle steps after each probe burst")
    ap.add_argument("--cart-action-dims", type=int, choices=[5, 6], default=6)
    ap.add_argument("--base-script", action="store_true",
                    help="also run the scripted-base insertion check")
    ap.add_argument("--base-script-steps", type=int, default=160)
    args = ap.parse_args()

    cfg = SceneEnvConfig(action_mode="cartesian_residual",
                         cart_action_dims=args.cart_action_dims,
                         include_images=False)
    print(f"[cart-smoke] building env (action_mode={cfg.action_mode}, "
          f"dims={cfg.cart_action_dims}, kp_pos={cfg.cart_kp_pos}, "
          f"kp_rot={cfg.cart_kp_rot}, max_wrench={cfg.cart_max_wrench[0]})", flush=True)
    env = SceneInsertEnv(cfg)
    dims = env._action_dim
    results = []

    def record(name, ok, detail):
        results.append((name, ok, detail))
        print(f"[cart-smoke] {PASS if ok else FAIL}  {name:<18} {detail}", flush=True)

    # ---- 1. hold ----
    env.reset(seed=0, options={"level": args.level, "jitter": False})
    p0, _ = _tcp_pose(env)
    hold = _run_steps(env, np.zeros(dims, np.float32), 20)
    drift_mm = float(np.linalg.norm(_tcp_pose(env)[0] - p0)) * 1e3
    ok = drift_mm < 2.0 and hold["force_n"] < 10.0 and hold["term"] is None
    record("hold", ok, f"drift={drift_mm:.2f}mm force={hold['force_n']:.1f}N "
                       f"tau={hold['tau']:.1f}Nm term={hold['term']}")

    # ---- 2. translation probes (+1 mm/step, port frame x/y/z) ----
    for i, name in enumerate(("trans_x", "trans_y", "trans_z(insert)")):
        r = _probe(env, args.level, i, True, cfg.cart_trans_scale_m,
                   args.steps, args.settle, dims)
        ok = (r["nan_free"] and r["term"] is None
              and r["cosine"] > 0.7 and 0.4 < r["ratio"] < 1.6
              and r["force_n"] < 20.0 and r["tau"] < 100.0)
        record(name, ok,
               f"moved={1e3 * np.asarray(r['achieved']).round(4)}mm "
               f"cos={r['cosine']:.2f} ratio={r['ratio']:.2f} "
               f"force={r['force_n']:.1f}N wrench={r['wrench_f']:.1f}N "
               f"tau={r['tau']:.1f}Nm term={r['term']}")

    # ---- 3. rotation probes (+1 deg/step about port frame x/y[/z]) ----
    rot_axes = ("rot_roll(x)", "rot_pitch(y)", "rot_yaw(z)")[:dims - 3]
    for i, name in enumerate(rot_axes):
        r = _probe(env, args.level, i, False, cfg.cart_rot_scale_rad,
                   args.steps, args.settle, dims)
        ok = (r["nan_free"] and r["term"] is None
              and r["cosine"] > 0.7 and 0.4 < r["ratio"] < 1.6
              and r["force_n"] < 20.0 and r["tau"] < 100.0)
        record(name, ok,
               f"rotated={np.degrees(np.asarray(r['achieved'])).round(2)}deg "
               f"cos={r['cosine']:.2f} ratio={r['ratio']:.2f} "
               f"force={r['force_n']:.1f}N wrench_t={r['wrench_t']:.1f}Nm "
               f"tau={r['tau']:.1f}Nm term={r['term']}")

    # ---- 4. optional scripted-base insertion ----
    if args.base_script:
        env.cfg.base_script_enabled = True
        env.reset(seed=0, options={"level": args.level, "jitter": False})
        d0 = float(env._insertion_depth_m())
        bs = _run_steps(env, np.zeros(dims, np.float32), args.base_script_steps)
        d1 = float(env._insertion_depth_m())
        info = bs.get("info", {})
        ok = (d1 - d0 > 0.015 and bs["force_n"] < 60.0
              and bs["term"] not in ("force_abort", "bad_collision", "off_limit"))
        record("base_script", ok,
               f"depth {1e3 * d0:.1f}->{1e3 * d1:.1f}mm "
               f"base_adv={info.get('cart_base_progress_mm', 0.0):.1f}mm "
               f"force={bs['force_n']:.1f}N term={bs['term']}")
        env.cfg.base_script_enabled = False

    env.close()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"[cart-smoke] {'ALL OK' if n_fail == 0 else f'{n_fail} FAILURES'} "
          f"({len(results)} checks)", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
