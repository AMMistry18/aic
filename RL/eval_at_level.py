#!/usr/bin/env python3
"""Evaluate a controller at a FIXED (pinned) curriculum level.

Standalone eval harness that faithfully replicates the way RL/train.py builds
the SCENE env (RL/scene_env.py:SceneInsertEnv), but pins the curriculum level to
an arbitrary ``--level`` for EVERY reset and rolls out N deterministic episodes.

Level pinning (per project spec):
  * reset_mode = "curriculum"
  * NO level_file (leave it None -> nothing overwrites _curriculum_level)
  * set_curriculum_level(L)
With no level_file and no training callback, the level stays pinned at L for
every reset, so any --level works uniformly.

Success signal is authoritative from the env: info["term_status"] captured on the
step where done is True. success iff term_status == "success".

Does NOT train, does NOT write under RL/output, does NOT edit any other file.
"""
from __future__ import annotations

import argparse
import os
import sys

# Headless GL: force EGL before any RL/mujoco import. The RL modules use
# os.environ.setdefault(...), which does NOT override an EXISTING empty-string
# MUJOCO_GL (this box exports MUJOCO_GL=""), so force it explicitly here.
if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = os.environ.get("AIC_MUJOCO_GL") or "egl"

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval a controller at a pinned level.")
    p.add_argument("--level", type=float, default=1.0,
                   help="curriculum level to pin for every reset (0..1)")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", type=str, default=None,
                   help="path to an SB3 SAC model.zip (required for --controller policy)")
    p.add_argument("--controller",
                   choices=["policy", "zero", "base_script", "teacher"],
                   default="policy")
    p.add_argument("--port-type", type=str, default="sfp")
    p.add_argument("--teacher-variant", choices=["v1", "funnel"], default="v1",
                   help="[--controller teacher] which scripted teacher: 'v1' "
                        "(known-good ~60%, seating-solved) or 'funnel' (passive "
                        "chamfer-funnel entry A/B variant). Default v1.")
    # deterministic (default) vs stochastic policy actions.
    #   --deterministic  -> model.predict(obs, deterministic=True)   [default]
    #   --stochastic     -> model.predict(obs, deterministic=False)
    p.add_argument("--deterministic", dest="deterministic", action="store_true",
                   default=True,
                   help="use deterministic policy actions (default)")
    p.add_argument("--stochastic", dest="deterministic", action="store_false",
                   help="sample stochastic policy actions (seeded via --seed)")
    p.add_argument("--diagnose", action="store_true",
                   help="[--controller teacher] seating-wrench diagnostic: is a "
                        "jammed peg force-limited (wrench pinned at cart_max_wrench) "
                        "or does the controller have headroom? Prints an aggregate "
                        "SUCCESS-vs-TIMEOUT table instead of the normal eval.")
    return p.parse_args()


TERM_BUCKETS = ["success", "force_abort", "bad_collision", "off_limit", "timeout", "other"]


def _run_diagnose(gym_env, base, teacher, env_cfg, args) -> int:
    """COMPLETE wedge diagnosis for the scripted teacher (measure-only).

    Steps the RAW gym env (not the VecEnv) so DummyVecEnv's auto-reset does not
    clobber env._cart_diag / env._cart_resid_* on the terminal step. Records a
    full per-step time-series plus initial reset conditions, then prints sections
    A-F (aggregate, timeout depth histogram, ROLL test, start-condition driver,
    saturation summary, representative timeout traces).

    Port-frame residual channels (env._cart_resid_rotvec / _cart_resid_pos):
      rotvec[0] = tilt about lat_x, rotvec[1] = tilt about lat_y,
      rotvec[2] = KEYED ROLL about insert_axis  (clip +/- base_script_residual_limit_rad)
      pos[0]/pos[1] = lateral about lat_x/lat_y (clip +/- base_script_residual_limit_m),
      pos[2] = axial lead.
    """
    cfg = base.cfg
    rot_clip_deg = float(np.degrees(cfg.base_script_residual_limit_rad))  # ~5.73
    pos_clip_mm = float(cfg.base_script_residual_limit_m) * 1e3           # 10.0
    SAT = 0.99   # fraction of the clip counted as "saturated"
    WIN = 100
    episodes = []

    for ep in range(int(args.episodes)):
        if ep == 0:
            obs, _ = gym_env.reset(seed=int(args.seed))
        else:
            obs, _ = gym_env.reset()
        teacher.reset()

        # ---- INITIAL conditions right after reset (privileged GT) ----
        tip0 = base.data.xpos[base._plug_tip_id].copy()
        iax, ilat = base._tip_port_errors(tip0)
        init = dict(
            roll=float(np.degrees(base._plug_roll_error())),
            axis=float(np.degrees(base._plug_axis_error())),
            lat=float(np.linalg.norm(ilat)) * 1e3,
            axial=float(iax) * 1e3,   # axial gap (retracted distance), mm
        )

        ts = []
        info = {}
        done = False
        while not done:
            a = teacher.act(base, obs)
            obs, _r, terminated, truncated, info = gym_env.step(a)
            rr = np.asarray(base._cart_resid_rotvec, dtype=np.float64)  # port-frame rotvec
            rp = np.asarray(base._cart_resid_pos, dtype=np.float64)     # port-frame pos
            ts.append(dict(
                depth=float(info.get("depth_norm", 0.0)),
                roll=float(np.degrees(base._plug_roll_error())),
                axis=float(np.degrees(base._plug_axis_error())),
                lat_mm=float(info.get("lateral_error_m", 0.0)) * 1e3,
                fz=abs(float(info.get("f_z", 0.0))),
                pen_mm=float(info.get("plug_port_penetration_excess_m", 0.0)) * 1e3,
                over_mm=float(info.get("overinsert_m", 0.0)) * 1e3,
                contacts=int(info.get("plug_port_contacts", 0)),
                kroll_res=float(np.degrees(rr[2])),               # keyed roll residual (deg)
                tilt_res=float(np.degrees(max(abs(rr[0]), abs(rr[1])))),
                latx_res=float(rp[0]) * 1e3, laty_res=float(rp[1]) * 1e3,
                axz_res=float(rp[2]) * 1e3,
            ))
            done = bool(terminated or truncated)

        depths = [s["depth"] for s in ts]
        maxd = max(depths)
        jam_i = next(i for i, s in enumerate(ts) if s["depth"] >= maxd - 1e-3)
        jam, fin = ts[jam_i], ts[-1]
        w = ts[-WIN:]
        # per-episode channel authority: max |residual| reached over the episode
        max_kroll = max(abs(s["kroll_res"]) for s in ts)
        max_tilt = max(s["tilt_res"] for s in ts)
        max_lat = max(max(abs(s["latx_res"]), abs(s["laty_res"])) for s in ts)
        max_ax = max(s["axz_res"] for s in ts)
        episodes.append(dict(
            term=str(info.get("term_status")), init=init, ts=ts,
            jam=jam, fin=fin, jam_i=jam_i, maxd=maxd,
            fz_mean=float(np.mean([s["fz"] for s in w])),
            fz_max=float(np.max([s["fz"] for s in w])),
            max_kroll=max_kroll, max_tilt=max_tilt, max_lat=max_lat, max_ax=max_ax,
            kroll_sat=max_kroll >= rot_clip_deg * SAT,
            tilt_sat=max_tilt >= rot_clip_deg * SAT,
            lat_sat=max_lat >= pos_clip_mm * SAT,
            ax_sat=max_ax >= pos_clip_mm * SAT,
        ))

    groups = {}
    for r in episodes:
        groups.setdefault(r["term"], []).append(r)
    succ = groups.get("success", [])
    tos = groups.get("timeout", [])
    W = 120

    print("")
    print("=" * W)
    print(f" TEACHER WEDGE DIAGNOSIS   level={args.level}  episodes={args.episodes}  seed={args.seed}")
    print(f" residual clips: KEYED-ROLL & TILT = +/-{rot_clip_deg:.2f} deg "
          f"(base_script_residual_limit_rad={cfg.base_script_residual_limit_rad}); "
          f"LATERAL & axial-lead = +/-{pos_clip_mm:.1f} mm")
    print(f" reset yaw jitter up to +/-{np.degrees(cfg.jitter_yaw_rad):.2f} deg; "
          f"success roll tol = {np.degrees(cfg.success_roll_tol_rad):.2f} deg; "
          f"keyed-roll = rotvec[2] about insert_axis")
    print("=" * W)

    # ---------------- A) aggregate table ----------------
    print(" A) AGGREGATE  (final-state geometry; residual %sat = episode ever hit the clip)")
    hdr = (f"{'group':>13} {'n':>3} {'f_depth':>7} {'roll_mn':>7} {'roll_mx':>7} "
           f"{'axis_mn':>7} {'lat_mn':>6} {'kroll_res_mn':>12} {'kroll%sat':>9} "
           f"{'tilt%sat':>8} {'lat%sat':>7} {'|f_z|mn':>7} {'pen_mm':>6} {'over_mm':>7}")
    print(hdr)
    print("-" * W)

    def _row(g, rs):
        n = len(rs)
        def fm(key):
            return float(np.mean([r["fin"][key] for r in rs]))
        roll_mx = max(r["fin"]["roll"] for r in rs)
        kroll_mn = float(np.mean([r["max_kroll"] for r in rs]))
        pk = 100.0 * sum(r["kroll_sat"] for r in rs) / n
        pt = 100.0 * sum(r["tilt_sat"] for r in rs) / n
        pl = 100.0 * sum(r["lat_sat"] for r in rs) / n
        fzmn = float(np.mean([r["fz_mean"] for r in rs]))
        print(f"{g:>13} {n:>3} {fm('depth'):>7.3f} {fm('roll'):>7.2f} {roll_mx:>7.2f} "
              f"{fm('axis'):>7.2f} {fm('lat_mm'):>6.2f} {kroll_mn:>12.2f} {pk:>8.0f}% "
              f"{pt:>7.0f}% {pl:>6.0f}% {fzmn:>7.2f} {fm('pen_mm'):>6.3f} {fm('over_mm'):>7.3f}")

    for g in ("success", "timeout", "bad_collision", "force_abort", "off_limit", "None"):
        if groups.get(g):
            _row(g, groups[g])
    print("-" * W)

    # ---------------- B) timeout depth histogram ----------------
    print("\n B) TIMEOUT final depth_norm histogram")
    if tos:
        for lo, hi in [(0.0, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 1.001)]:
            c = sum(1 for r in tos if lo <= r["fin"]["depth"] < hi)
            print(f"   [{lo:.2f},{hi:.2f}): {c:>3}  {'#' * c}")
    else:
        print("   (no timeouts)")

    # ---------------- C) ROLL TEST ----------------
    print("\n C) ROLL TEST")
    if succ and tos:
        print(f"   mean final roll_err_deg:  success={np.mean([r['fin']['roll'] for r in succ]):.2f}"
              f"   timeout={np.mean([r['fin']['roll'] for r in tos]):.2f}")
        print(f"   mean INITIAL roll_err_deg: success={np.mean([r['init']['roll'] for r in succ]):.2f}"
              f"   timeout={np.mean([r['init']['roll'] for r in tos]):.2f}")
    if tos:
        print("   within TIMEOUTS: final roll_err binned vs wedge depth (maxd):")
        for lo, hi in [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 90.0)]:
            sub = [r for r in tos if lo <= r["fin"]["roll"] < hi]
            if sub:
                print(f"     roll[{lo:.0f},{hi:.0f})deg  n={len(sub):>2}  "
                      f"mean_wedge_depth={np.mean([r['maxd'] for r in sub]):.3f}  "
                      f"kroll%sat={100.0*sum(s['kroll_sat'] for s in sub)/len(sub):.0f}%")
        pk = 100.0 * sum(r["kroll_sat"] for r in tos) / len(tos)
        print(f"   %% timeout episodes with KEYED-ROLL residual SATURATED at "
              f"{rot_clip_deg:.2f} deg clip: {pk:.0f}%")

    # ---------------- D) START-CONDITION DRIVER ----------------
    print("\n D) START-CONDITION DRIVER  (success_rate binned by initial condition)")

    def _sr_bins(key, edges, unit):
        print(f"   by initial {key} ({unit}):")
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = [r for r in episodes if lo <= r["init"][key] < hi]
            if sub:
                sr = 100.0 * sum(r["term"] == "success" for r in sub) / len(sub)
                print(f"     [{lo:>4.1f},{hi:>4.1f}) n={len(sub):>3}  success_rate={sr:5.1f}%")
    _sr_bins("roll", [0, 2, 4, 6, 7, 90], "deg, keyed roll")
    _sr_bins("lat", [0, 2, 4, 6, 8, 100], "mm, lateral")
    _sr_bins("axis", [0, 1, 2, 3, 90], "deg, tilt")

    # ---------------- E) SATURATION SUMMARY ----------------
    print("\n E) SATURATION SUMMARY  (across TIMEOUT episodes, which channel hit its clip)")
    if tos:
        n = len(tos)
        for name, key in [("keyed-roll", "kroll_sat"), ("tilt", "tilt_sat"),
                          ("lateral", "lat_sat"), ("axial-lead", "ax_sat")]:
            c = sum(r[key] for r in tos)
            print(f"   {name:>11}: {c:>3}/{n}  ({100.0*c/n:5.1f}%)")
    else:
        print("   (no timeouts)")

    # ---------------- F) representative timeout traces ----------------
    print("\n F) REPRESENTATIVE TIMEOUT TIME-SERIES (every ~10 steps)")
    if tos:
        picks = sorted(tos, key=lambda r: r["maxd"])
        idx = sorted(set([0, len(picks) // 4, len(picks) // 2,
                          3 * len(picks) // 4, len(picks) - 1]))
        for j in idx:
            r = picks[j]
            print(f"   --- timeout wedge_depth={r['maxd']:.3f} final_depth={r['fin']['depth']:.3f}"
                  f"  init(roll={r['init']['roll']:.1f} tilt={r['init']['axis']:.1f} "
                  f"lat={r['init']['lat']:.1f}mm axial_gap={r['init']['axial']:.1f}mm)"
                  f"  kroll_sat={r['kroll_sat']} lat_sat={r['lat_sat']} jam@step={r['jam_i']}")
            print(f"      {'step':>4} {'depth':>6} {'roll':>6} {'axis':>6} {'lat_mm':>6} "
                  f"{'|f_z|':>6} {'kroll_res':>9}")
            tsr = r["ts"]
            for k in range(0, len(tsr), 10):
                s = tsr[k]
                print(f"      {k:>4} {s['depth']:>6.3f} {s['roll']:>6.2f} {s['axis']:>6.2f} "
                      f"{s['lat_mm']:>6.2f} {s['fz']:>6.2f} {s['kroll_res']:>9.2f}")
    print("=" * W)
    return 0


def main() -> int:
    args = parse_args()

    # heavy imports (torch/mujoco/sb3) after arg parse
    import dataclasses

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    from gymnasium.wrappers import TimeLimit

    from RL.env import EnvConfig
    from RL.scene_env import SceneInsertEnv, SceneEnvConfig

    # ------------------------------------------------------------------ #
    # env config (mirror train.py: EnvConfig() gives term.max_steps, reward)
    # ------------------------------------------------------------------ #
    env_cfg = EnvConfig()
    # scene env uses the redesigned residual-insertion reward (RL/reward.py)
    from RL.reward import RewardConfig as SceneRewardConfig
    reward_cfg = SceneRewardConfig()

    # controller-driven action mode:
    #  * policy / zero  -> joint_residual (matches existing curriculum checkpoints)
    #  * base_script    -> cartesian_residual + scripted base, ZERO residual
    #  * teacher (Fix A)-> cartesian_residual, base_script OFF: the scripted GT
    #                      teacher drives the WHOLE last-inch motion via the
    #                      residual (same action space as the distilled student).
    if args.controller == "base_script":
        action_mode = "cartesian_residual"
        base_script_enabled = True
    elif args.controller == "teacher":
        action_mode = "cartesian_residual"
        base_script_enabled = False
    else:
        action_mode = "joint_residual"
        base_script_enabled = False

    # ------------------------------------------------------------------ #
    # controller: load policy FIRST so we can size the env's camera images
    # to exactly what the checkpoint expects (checkpoints may have been
    # trained at a non-default --image-size, e.g. 128 vs train.py default 256).
    # ------------------------------------------------------------------ #
    model = None
    image_h = image_w = 256  # train.py default
    if args.controller == "policy":
        if not args.policy:
            print("ERROR: --controller policy requires --policy <model.zip>",
                  file=sys.stderr)
            return 2
        model = SAC.load(args.policy)
        img_space = model.observation_space.spaces.get("image")
        if img_space is not None and len(img_space.shape) == 3:
            # saved obs is channels-first (C, H, W) after VecTransposeImage
            _, image_h, image_w = img_space.shape

    # scene_kwargs built exactly as train.py's scene path (train.py ~443-465),
    # using train.py defaults for anything not exposed by this CLI.
    scene_kwargs = dict(
        image_h=image_h, image_w=image_w,
        reward=reward_cfg,
        max_episode_steps=env_cfg.term.max_steps,
        action_mode=action_mode,
        cart_action_dims=6,
        cart_trans_scale_m=1.0 * 1e-3,          # --cart-trans-scale-mm default 1.0
        cart_rot_scale_rad=float(np.radians(1.0)),  # --cart-rot-scale-deg default 1.0
        base_script_enabled=base_script_enabled,
        base_script_step_m=0.5 * 1e-3,          # --base-script-step-mm default 0.5
        # Residual ENVELOPE (Fix A, teacher path only: base_script OFF). Measured
        # worst-case axial travel to seat at level 1.0 = 0.092 m; set the axial
        # residual limit to 0.20 m so the residual can reach the seated goal AND
        # lead ~0.11 m past it, letting the UNCHANGED impedance controller
        # (cart_kp_pos=100 N/m, cart_max_wrench=10 N) ramp seating force to the
        # 10 N cap. cart_rot_limit_rad left at the env default 0.35 rad (~20 deg;
        # reset roll/tilt are <3 deg so rotation authority is ample).
        cart_pos_limit_m=0.20,
        cart_rot_limit_rad=0.35,
    )

    def _thunk():
        e = SceneInsertEnv(SceneEnvConfig(**scene_kwargs))
        # Curriculum levels are gone (fixed RSI-mix start distribution).
        # --level now only selects the reset mode: >=0.5 -> free 1-inch
        # starts ('random'), <0.5 -> RSI in-port starts ('near_goal').
        e.set_reset_mode("random" if float(args.level) >= 0.5 else "near_goal")
        e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
        return e

    # single deterministic env (n=1), same vec-wrapping as train.py
    dummy = DummyVecEnv([_thunk])
    venv = VecTransposeImage(dummy)

    act_shape = venv.action_space.shape  # e.g. (6,)

    # scripted privileged teacher: reads GT off the unwrapped SceneInsertEnv
    teacher = None
    base_env = None
    if args.controller == "teacher":
        if args.teacher_variant == "funnel":
            from RL.teacher.scripted_teacher_funnel import ScriptedTeacher
        else:
            from RL.teacher.scripted_teacher import ScriptedTeacher
        teacher = ScriptedTeacher(action_dim=int(act_shape[0]))
        # DummyVecEnv -> TimeLimit -> SceneInsertEnv
        base_env = dummy.envs[0].unwrapped

    # ------------------------------------------------------------------ #
    # seating-wrench diagnostic (measure-only): step the RAW gym env directly
    # (dummy.envs[0]) so DummyVecEnv's auto-reset does not clobber env._cart_diag
    # on the terminal step. Reuses the identical scene_kwargs + teacher.
    # ------------------------------------------------------------------ #
    if args.diagnose:
        if args.controller != "teacher":
            print("ERROR: --diagnose requires --controller teacher", file=sys.stderr)
            return 2
        return _run_diagnose(dummy.envs[0], base_env, teacher, env_cfg, args)

    def get_action(obs):
        if args.controller == "policy":
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            return action
        if args.controller == "teacher":
            a = teacher.act(base_env, obs)
            return np.asarray(a, dtype=np.float32).reshape(venv.num_envs, *act_shape)
        # zero residual for both 'zero' and 'base_script'
        return np.zeros((venv.num_envs, *act_shape), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # deterministic rollout
    # ------------------------------------------------------------------ #
    venv.seed(int(args.seed))
    if model is not None:
        try:
            model.set_random_seed(int(args.seed))
        except Exception:
            pass

    counts = {k: 0 for k in TERM_BUCKETS}
    successes = 0
    score_success_hits = 0
    score_success_seen = 0
    term_status_none_on_done = 0

    obs = venv.reset()
    for _ep in range(int(args.episodes)):
        done = False
        while not done:
            action = get_action(obs)
            obs, _rewards, dones, infos = venv.step(action)
            done = bool(dones[0])
            if done:
                if teacher is not None:
                    teacher.reset()   # DummyVecEnv auto-reset -> fresh episode state
                info0 = infos[0]
                ts = info0.get("term_status")
                if ts is None:
                    # fallbacks in case a wrapper tucked the terminal info away
                    ti = info0.get("terminal_info")
                    if isinstance(ti, dict):
                        ts = ti.get("term_status")
                if ts is None:
                    term_status_none_on_done += 1
                    ts = "timeout" if info0.get("TimeLimit.truncated") else "other"
                bucket = ts if ts in counts else "other"
                counts[bucket] += 1
                if ts == "success":
                    successes += 1
                # cross-check (non-authoritative)
                sc = info0.get("score_success")
                if sc is None and isinstance(info0.get("terminal_info"), dict):
                    sc = info0["terminal_info"].get("score_success")
                if sc is not None:
                    score_success_seen += 1
                    if bool(sc):
                        score_success_hits += 1

    total = int(args.episodes)
    success_rate = successes / total if total else 0.0

    # ------------------------------------------------------------------ #
    # report
    # ------------------------------------------------------------------ #
    print("")
    print("=" * 56)
    print(" eval_at_level  results")
    print("=" * 56)
    print(f" controller     : {args.controller}")
    print(f" level (pinned) : {args.level}")
    print(f" port_type      : {args.port_type}")
    print(f" action_mode    : {action_mode}"
          + ("  (base_script, zero residual)" if base_script_enabled else ""))
    if args.controller == "policy":
        print(f" policy         : {args.policy}")
        print(f" deterministic  : {bool(args.deterministic)}")
    print(f" seed           : {args.seed}")
    print("-" * 56)
    print(f" total episodes : {total}")
    print(f" success_rate   : {success_rate:.4f}  ({100.0 * success_rate:.2f}%)")
    print("-" * 56)
    print(" term_status breakdown:")
    for k in TERM_BUCKETS:
        n = counts[k]
        pct = 100.0 * n / total if total else 0.0
        print(f"   {k:<14}: {n:>5d}  ({pct:6.2f}%)")
    print("-" * 56)
    if score_success_seen:
        print(f" score_success (cross-check): {score_success_hits}/{score_success_seen} "
              f"({100.0 * score_success_hits / score_success_seen:.2f}%)")
    if term_status_none_on_done:
        print(f" WARNING: term_status was None on {term_status_none_on_done} "
              f"terminal step(s); used fallback")
    print("=" * 56)

    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
