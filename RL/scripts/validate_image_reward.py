"""Validate the image-distance reward signal on the real scene.

The dense image term (`RL/reward.py:r_image`) is only useful if the L1
distance to the seated goal image is (a) monotonically increasing as the plug
moves away from the seated pose and (b) has usable dynamic range across the
whole curriculum span — otherwise SAC gets a flat or misleading gradient.

This script places the plug at controlled offsets in the PORT FRAME (the same
placement pipeline `SceneInsertEnv._reset_to_level` uses: TCP IK seed +
finite-difference tip/axis IK + settle), renders the reward image, and sweeps:

  1. axial retraction 0 .. last_inch_m  (seated -> out of the port -> approach)
  2. lateral offset at a fixed standoff  (does L1 see x/y error?)
  3. optional low-res comparison         (--compare-res 256)

Outputs a verdict + Spearman rank correlations to stdout and a plot to
RL/tests/new_image_reward_validation.png.

Run in-container:
    MUJOCO_GL=egl pixi run python RL/scripts/validate_image_reward.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mujoco  # noqa: E402

from RL.scene_env import SceneEnvConfig, SceneInsertEnv  # noqa: E402
from RL.reward import r_image  # noqa: E402


def place(env: SceneInsertEnv, tip: np.ndarray, settle: int = 80) -> float:
    """Drive the arm so the welded SFP tip sits at `tip` (goal orientation)."""
    quat = env._goal_quat.copy()
    tcp = env._tcp_for_tip(tip, quat)
    q = env._ik(tcp, quat, env._home)
    q = env._ik_tcp_position(tcp, q)
    q, _tip_err, _axis_err, _roll_err = env._ik_tip_axis(tip, env._insert_axis, q)
    env._rigid_home(q)
    for _ in range(settle):
        env.data.ctrl[:6] = env._base_torque(q)
        env.data.ctrl[6] = env.cfg.gripper_ctrl
        mujoco.mj_step(env.model, env.data)
    actual = env.data.xpos[env._plug_tip_id].copy()
    return float(np.linalg.norm(actual - tip))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy."""
    def rank(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r
    rx, ry = rank(np.asarray(x, dtype=np.float64)), rank(np.asarray(y, dtype=np.float64))
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reward-res", type=int, default=0,
                   help="reward render res (0 = native 1152x1024 center cam)")
    p.add_argument("--compare-res", type=int, default=256,
                   help="second, lower render res to compare (0 = skip)")
    p.add_argument("--axial-points", type=int, default=13)
    p.add_argument("--lateral-points", type=int, default=9)
    p.add_argument("--lateral-max-mm", type=float, default=6.0)
    p.add_argument("--lateral-standoff-m", type=float, default=0.056,
                   help="retraction at which the lateral sweep runs "
                        "(default: ~1 cm outside the entrance)")
    p.add_argument("--out", type=Path,
                   default=_REPO / "RL" / "tests" / "new_image_reward_validation.png")
    args = p.parse_args()

    cfg = SceneEnvConfig(include_images=True, reward_image_res=args.reward_res)
    env = SceneInsertEnv(cfg)
    span = float(env.cfg.last_inch_m)
    seated = env._inserted_tip.copy()

    cmp_renderer = None
    if args.compare_res > 0:
        cmp_renderer = mujoco.Renderer(env.model, height=args.compare_res,
                                       width=args.compare_res)

    def render_pair():
        native = env._render_reward_image().astype(np.float32)
        low = None
        if cmp_renderer is not None:
            cmp_renderer.update_scene(env.data, camera="center_camera")
            low = cmp_renderer.render().astype(np.float32)
        return native, low

    # goal images from OUR seated placement so both resolutions share a baseline
    place(env, seated)
    goal_native, goal_low = render_pair()

    # ---- 1. axial sweep -------------------------------------------------- #
    retracts = np.linspace(0.0, span, args.axial_points)
    ax_l1, ax_l1_low, ax_place_err = [], [], []
    for r in retracts:
        err = place(env, seated + r * env._retract_dir)
        native, low = render_pair()
        _, l1 = r_image(native, goal_native, env.cfg.reward)
        ax_l1.append(float(l1))
        if low is not None:
            _, l1l = r_image(low, goal_low, env.cfg.reward)
            ax_l1_low.append(float(l1l))
        ax_place_err.append(err)
        print(f"  axial retract={r * 1000:6.1f}mm  l1={float(l1):.5f}"
              + (f"  l1@{args.compare_res}={ax_l1_low[-1]:.5f}" if low is not None else "")
              + f"  place_err={err * 1000:.2f}mm", flush=True)

    # ---- 2. lateral sweep ------------------------------------------------ #
    standoff = float(args.lateral_standoff_m)
    base = seated + standoff * env._retract_dir
    laterals = np.linspace(-args.lateral_max_mm, args.lateral_max_mm,
                           args.lateral_points) * 1e-3
    lat_l1, lat_place_err = [], []
    for off in laterals:
        err = place(env, base + off * env._lat_x)
        native, _low = render_pair()
        _, l1 = r_image(native, goal_native, env.cfg.reward)
        lat_l1.append(float(l1))
        lat_place_err.append(err)
        print(f"  lateral off={off * 1000:+5.1f}mm  l1={float(l1):.5f}"
              f"  place_err={err * 1000:.2f}mm", flush=True)

    # ---- verdict ---------------------------------------------------------- #
    rho_axial = spearman(retracts, np.asarray(ax_l1))
    # lateral: L1 should grow with |offset| -> correlate against abs(offset)
    rho_lateral = spearman(np.abs(laterals), np.asarray(lat_l1))
    ax_arr = np.asarray(ax_l1)
    dyn_range = float((ax_arr.max() - ax_arr.min()) / max(ax_arr.max(), 1e-9))
    near_goal_gain = float(ax_arr[1] - ax_arr[0])   # first 6mm of retraction
    monotone_frac = float(np.mean(np.diff(ax_arr) > 0))

    rho_axial_low = None
    if ax_l1_low:
        rho_axial_low = spearman(retracts, np.asarray(ax_l1_low))

    ok = rho_axial >= 0.9 and dyn_range >= 0.2 and near_goal_gain > 0
    print("\n===== image-distance reward validation =====")
    print(f"axial Spearman rho          = {rho_axial:+.3f}   (want >= +0.9)")
    if rho_axial_low is not None:
        print(f"axial Spearman rho @ {args.compare_res:4d}   = {rho_axial_low:+.3f}")
    print(f"lateral |off| Spearman rho  = {rho_lateral:+.3f}   (want > 0)")
    print(f"axial monotone step frac    = {monotone_frac:.2f}")
    print(f"dynamic range (max-min)/max = {dyn_range:.2f}   (want >= 0.2)")
    print(f"near-goal L1 gain (6mm)     = {near_goal_gain:+.5f} (want > 0)")
    print(f"placement err max           = {1000 * max(ax_place_err + lat_place_err):.2f} mm")
    print("VERDICT:", "PASS" if ok else "FAIL")

    # ---- plot -------------------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(retracts * 1000, ax_l1, "o-", label="native res")
        if ax_l1_low:
            axes[0].plot(retracts * 1000, ax_l1_low, "s--",
                         label=f"{args.compare_res}x{args.compare_res}")
        axes[0].axvline(env.cfg.seated_depth_m * 1000, color="gray", ls=":",
                        label="port entrance")
        axes[0].set_xlabel("axial retraction from seated (mm)")
        axes[0].set_ylabel("image L1 (norm) vs seated goal")
        axes[0].set_title(f"axial sweep  (Spearman {rho_axial:+.3f})")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(laterals * 1000, lat_l1, "o-")
        axes[1].set_xlabel(f"lateral offset at {standoff * 1000:.0f}mm standoff (mm)")
        axes[1].set_ylabel("image L1 (norm)")
        axes[1].set_title(f"lateral sweep  (Spearman vs |off| {rho_lateral:+.3f})")
        axes[1].grid(alpha=0.3)
        fig.suptitle(f"image-distance reward validation — "
                     f"{'PASS' if ok else 'FAIL'}", fontweight="bold")
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=120)
        print(f"plot: {args.out}")
    except ImportError:
        print("matplotlib unavailable — skipped plot")

    if cmp_renderer is not None:
        cmp_renderer.close()
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
