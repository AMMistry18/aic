"""Scoring-proxy evaluator for AIC insertion policies.

Runs N trials with a trained rsl_rl checkpoint and computes metrics that
mirror the official Gazebo scoring:
    Tier 1: validity (did the policy run without crashing)
    Tier 2: smoothness (jerk), duration, efficiency (path length), force penalty
    Tier 3: success / partial insertion / proximity

This is a sim2sim proxy — absolute scores won't match Gazebo exactly, but
relative scores across checkpoints are highly predictive of real progress.

Usage:
    isaaclab -p scripts/eval_policy.py --task AIC-Insert-SC-v0 \\
        --checkpoint logs/rsl_rl/aic_insert_sc/<run>/model_*.pt \\
        --num_trials 20 --headless

Writes eval_<timestamp>.json with per-trial and summary metrics.
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

# Argparse BEFORE AppLauncher (required)
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument(
    "--agent_cfg_entry_point",
    type=str,
    default=None,
    help="Optional module:Class runner cfg override for checkpoints trained with a different registered agent cfg.",
)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.num_envs = args.num_trials
args.enable_cameras = False  # state-only eval

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Heavy imports AFTER AppLauncher
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import aic_task  # registers envs
from aic_task.tasks.manager_based.aic_task.kinematic_last_mile_env import KinematicLastMileRLEnv


# Tier 2 scoring thresholds (from scoring.md)
JERK_MAX = 50.0          # m/s^3 -> 0 points
JERK_FULL = 0.0          # 0 m/s^3 -> 6 points
DURATION_FULL = 5.0      # s -> 12 points
DURATION_MAX = 60.0      # s -> 0 points
FORCE_THRESH = 20.0      # N
FORCE_DURATION = 1.0     # s
STRICT_XY_THRESH = 0.010
STRICT_AXIS_THRESH = 0.45
STRICT_TWIST_THRESH = 0.45
STRICT_DEPTH_FRACTION = 0.50


def score_trial(traj, dt):
    """Compute per-trial Tier 2/3 metrics from a trajectory dict."""
    t = traj["t"]
    T = len(t)
    pos = np.array(traj["ee_pos"])          # (T, 3)
    force = np.array(traj["force_mag"])      # (T,)
    depth = np.array(traj["depth_below"])    # (T,), positive = deeper
    xy_err = np.array(traj["xy_err"])        # (T,)
    axis = np.array(traj.get("axis_alignment", [0.0] * T))
    twist = np.array(traj.get("twist_alignment", [0.0] * T))
    strict_gate = np.array(traj.get("strict_success_gate", [False] * T), dtype=bool)
    port_depth = traj["port_depth"]

    duration = t[-1] - t[0] if T > 1 else 0.0

    # --- Tier 3 first: success / partial / proximity ---
    final_depth = depth[-1]
    final_xy = xy_err[-1]
    final_axis = axis[-1]
    final_twist = twist[-1]
    max_axis = float(axis.max()) if T > 0 else 0.0
    max_twist = float(twist.max()) if T > 0 else 0.0
    max_depth = float(depth.max()) if T > 0 else 0.0
    min_xy = float(xy_err.min()) if T > 0 else 0.0
    ever_strict_gate = bool(strict_gate.any()) or bool(traj.get("success", False))
    final_depth_gate = bool(final_depth > STRICT_DEPTH_FRACTION * port_depth)
    final_xy_gate = bool(final_xy < STRICT_XY_THRESH)
    final_axis_gate = bool(final_axis > STRICT_AXIS_THRESH)
    final_twist_gate = bool(final_twist > STRICT_TWIST_THRESH)
    diagnostics = {
        "final_axis_alignment": float(final_axis),
        "final_twist_alignment": float(final_twist),
        "max_axis_alignment": float(max_axis),
        "max_twist_alignment": float(max_twist),
        "max_depth_mm": float(max_depth * 1000),
        "min_xy_mm": float(min_xy * 1000),
        "final_depth_gate": final_depth_gate,
        "final_xy_gate": final_xy_gate,
        "final_axis_gate": final_axis_gate,
        "final_twist_gate": final_twist_gate,
        "ever_strict_gate": ever_strict_gate,
    }

    if traj.get("success", False):
        t3_name = "success"
        t3_score = 75.0
    elif final_depth > 0.95 * port_depth and final_xy < 0.003:
        t3_name = "success"
        t3_score = 75.0
    elif final_xy < 0.005 and final_depth > 0:
        # Partial insertion
        frac = min(final_depth / port_depth, 1.0)
        t3_name = "partial"
        t3_score = 38.0 + (50.0 - 38.0) * frac
    else:
        # Proximity (0-25)
        initial_xy = xy_err[0]
        max_dist = max(initial_xy * 0.5, 0.05)
        dist = np.linalg.norm([final_xy, max(-final_depth, 0)])
        t3_score = max(0.0, 25.0 * (1.0 - dist / max_dist))
        t3_name = "proximity"

    # --- Tier 2: only awarded if Tier 3 > 0 ---
    if t3_score <= 0:
        return {
            "t3_name": t3_name, "t3_score": t3_score,
            "smoothness": 0, "duration_score": 0, "efficiency": 0,
            "force_penalty": 0,
            "final_depth_mm": float(final_depth * 1000),
            "final_xy_mm": float(final_xy * 1000),
            "duration_s": float(duration),
            "total": t3_score,
            **diagnostics,
        }

    # Smoothness — jerk approximated by finite diffs of acceleration
    if T >= 3:
        vel = np.diff(pos, axis=0) / dt
        speed = np.linalg.norm(vel, axis=-1)
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        jerk_mag = np.linalg.norm(jerk, axis=-1)
        # Only count when moving
        moving_mask = speed[2:] > 0.01
        if moving_mask.any():
            avg_jerk = float(jerk_mag[moving_mask].mean())
        else:
            avg_jerk = 0.0
    else:
        avg_jerk = 0.0
    smooth = 6.0 * max(0.0, 1.0 - avg_jerk / JERK_MAX)

    # Duration
    dur_score = 12.0 * max(0.0, 1.0 - max(0.0, duration - DURATION_FULL) / (DURATION_MAX - DURATION_FULL))

    # Efficiency — path length vs initial plug-port distance
    if T >= 2:
        path_len = float(np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum())
    else:
        path_len = 0.0
    initial_dist = xy_err[0]  # rough initial distance
    max_extra = 1.0
    extra = max(0.0, path_len - initial_dist)
    eff_score = 6.0 * max(0.0, 1.0 - extra / max_extra)

    # Force penalty
    over = force > FORCE_THRESH
    # Consecutive-run length of over-threshold
    longest_run = 0
    cur = 0
    for o in over:
        if o:
            cur += 1
            longest_run = max(longest_run, cur)
        else:
            cur = 0
    force_pen = -12.0 if (longest_run * dt) > FORCE_DURATION else 0.0

    total = 1.0 + smooth + dur_score + eff_score + force_pen + t3_score

    return {
        "t3_name": t3_name, "t3_score": float(t3_score),
        "smoothness": float(smooth), "duration_score": float(dur_score),
        "efficiency": float(eff_score), "force_penalty": float(force_pen),
        "final_depth_mm": float(final_depth * 1000),
        "final_xy_mm": float(final_xy * 1000),
        "duration_s": float(duration),
        "avg_jerk": float(avg_jerk),
        "path_length_m": float(path_len),
        "total": float(total),
        **diagnostics,
    }


def main():
    # Env cfg
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_trials)
    env_cls = KinematicLastMileRLEnv if getattr(env_cfg, "use_kinematic_tip", False) else ManagerBasedRLEnv
    env = env_cls(cfg=env_cfg)

    # Wrap for rsl_rl
    env_wrapped = RslRlVecEnvWrapper(env)

    # Load policy — use rsl_rl's runner to get an actor with the right obs normalization
    from isaaclab_tasks.utils import load_cfg_from_registry
    if args.agent_cfg_entry_point:
        import importlib

        module_name, class_name = args.agent_cfg_entry_point.split(":")
        runner_cfg_cls = getattr(importlib.import_module(module_name), class_name)
        runner_cfg = runner_cfg_cls()
    else:
        runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner_cfg_dict = runner_cfg.to_dict() if hasattr(runner_cfg, "to_dict") else dict(runner_cfg.__dict__)
    runner = OnPolicyRunner(env_wrapped, runner_cfg_dict, log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    # Target key comes from the env cfg
    target_key = env_cfg._target_key
    from aic_task.tasks.manager_based.aic_task.mdp.mdp_insert import _TARGETS
    port_depth = _TARGETS[target_key]["depth"]

    dt = env.step_dt
    trials = [
        {
            "t": [],
            "ee_pos": [],
            "force_mag": [],
            "depth_below": [],
            "xy_err": [],
            "axis_alignment": [],
            "twist_alignment": [],
            "strict_success_gate": [],
            "port_depth": port_depth,
            "success": False,
        }
        for _ in range(args.num_trials)
    ]
    active = torch.ones(args.num_trials, dtype=torch.bool, device=args.device)

    obs, _ = env.reset()
    t = 0.0
    max_steps = int(15.0 / dt) + 1  # episode_length_s

    from aic_task.tasks.manager_based.aic_task.mdp.mdp_insert import (
        _tip_in_port_frame,
        plug_port_axis_alignment,
        plug_port_twist_alignment,
    )
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene["robot"]
    ee_idx = robot.data.body_names.index("wrist_3_link")

    for step in range(max_steps):
        # Log the pre-step state. ManagerBasedRLEnv resets terminated envs inside
        # step(), so post-step poses for done envs are reset poses, not terminal poses.
        tip_frame = _tip_in_port_frame(env, target_key)
        delta_port = tip_frame[0]
        tip_w = tip_frame[3]
        xy = torch.norm(delta_port[:, :2], dim=-1).cpu().numpy()
        depth_t = -delta_port[:, 2]
        axis_t = plug_port_axis_alignment(env, target_key)
        twist_t = plug_port_twist_alignment(env, target_key)
        strict_gate_t = (
            (depth_t > STRICT_DEPTH_FRACTION * port_depth)
            & (torch.norm(delta_port[:, :2], dim=-1) < STRICT_XY_THRESH)
            & (axis_t > STRICT_AXIS_THRESH)
            & (twist_t > STRICT_TWIST_THRESH)
        )
        depth = depth_t.cpu().numpy()
        axis = axis_t.cpu().numpy()
        twist = twist_t.cpu().numpy()
        strict_gate = strict_gate_t.cpu().numpy()
        force_w = robot.root_physx_view.get_link_incoming_joint_force()
        body_ids = [ee_idx]
        force_w = force_w[:, body_ids, :3].reshape(args.num_trials, -1)
        force_mag = torch.norm(force_w, dim=-1).cpu().numpy()
        ee_pos = tip_w.cpu().numpy()

        for i in range(args.num_trials):
            if active[i]:
                trials[i]["t"].append(t)
                trials[i]["ee_pos"].append(ee_pos[i].tolist())
                trials[i]["force_mag"].append(float(force_mag[i]))
                trials[i]["depth_below"].append(float(depth[i]))
                trials[i]["xy_err"].append(float(xy[i]))
                trials[i]["axis_alignment"].append(float(axis[i]))
                trials[i]["twist_alignment"].append(float(twist[i]))
                trials[i]["strict_success_gate"].append(bool(strict_gate[i]))

        with torch.no_grad():
            actions = policy(obs)
        obs, _, term, trunc, _ = env.step(actions)

        success_done = env.termination_manager.get_term("insertion_success").detach().cpu()
        for i in range(args.num_trials):
            if active[i] and bool(success_done[i]):
                trials[i]["success"] = True

        done = (term | trunc).cpu()
        active = active & ~done.to(active.device)
        if not active.any():
            break
        t += dt

    # Score
    results = [score_trial(tr, dt) for tr in trials]
    summary = {
        "num_trials": args.num_trials,
        "success_rate": sum(r["t3_name"] == "success" for r in results) / args.num_trials,
        "partial_rate": sum(r["t3_name"] == "partial" for r in results) / args.num_trials,
        "proximity_rate": sum(r["t3_name"] == "proximity" for r in results) / args.num_trials,
        "mean_total": float(np.mean([r["total"] for r in results])),
        "mean_final_depth_mm": float(np.mean([r["final_depth_mm"] for r in results])),
        "mean_final_xy_mm": float(np.mean([r["final_xy_mm"] for r in results])),
        "mean_duration_s": float(np.mean([r["duration_s"] for r in results])),
        "mean_final_axis_alignment": float(np.mean([r["final_axis_alignment"] for r in results])),
        "mean_final_twist_alignment": float(np.mean([r["final_twist_alignment"] for r in results])),
        "mean_max_axis_alignment": float(np.mean([r["max_axis_alignment"] for r in results])),
        "mean_max_twist_alignment": float(np.mean([r["max_twist_alignment"] for r in results])),
        "strict_gate_rate": sum(r["ever_strict_gate"] for r in results) / args.num_trials,
    }

    out_path = args.out or f"eval_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "trials": results, "checkpoint": args.checkpoint, "task": args.task}, f, indent=2)

    print("\n=== Eval Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
