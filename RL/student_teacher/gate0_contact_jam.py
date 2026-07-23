"""Gate 0: prove randomized MuJoCo can express the Flowstate contact jam.

The deterministic guided controller is the sentinel because it is already known
to seat at calibrated nominal poses. Gate 0 passes only when that same controller
remains successful nominally and at least one randomized episode shows sustained
force with little progress in the observed 5--9 mm insertion-depth band.

The v2 diagnostic preserves that mouth-band check and additionally tests the new
Flowstate signature: low-force descent past 20 mm, a low-force lateral wedge at a
variable depth, and recovery after a small scripted lateral correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import hashlib

import mujoco
import numpy as np

from RL.student_teacher.parity.evaluate_guided_controller import guided_action
from RL.student_teacher.student_env_a import DEPLOY_POS_SCALE, make_student_env_a


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nominal-episodes", type=int, default=10)
    p.add_argument("--randomized-episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--level", type=float, default=0.65,
                   help="fixed reverse-curriculum start near the port mouth")
    p.add_argument("--nominal-min-success", type=float, default=0.90)
    p.add_argument("--jam-min-force-n", type=float, default=8.0)
    p.add_argument("--jam-window-s", type=float, default=1.2)
    p.add_argument("--jam-max-progress-mm", type=float, default=1.0)
    p.add_argument("--jam-depth-min-mm", type=float, default=5.0)
    p.add_argument("--jam-depth-max-mm", type=float, default=9.0)
    p.add_argument("--randomized-perception-noise", type=float, default=0.25)
    p.add_argument("--randomized-grasp-noise", type=float, default=0.25)
    p.add_argument("--v2-episodes", type=int, default=40)
    p.add_argument("--v2-deep-min-mm", type=float, default=20.0)
    p.add_argument("--v2-low-force-max-n", type=float, default=10.0)
    p.add_argument("--v2-stall-depth-min-mm", type=float, default=4.0)
    p.add_argument("--v2-stall-window-s", type=float, default=1.2)
    p.add_argument("--v2-stall-max-progress-mm", type=float, default=1.0)
    p.add_argument("--v2-lateral-growth-min-mm", type=float, default=0.25)
    p.add_argument("--v2-wedge-max-lateral-mm", type=float, default=5.0,
                   help="exclude gross lateral misses from real-wedge counts")
    p.add_argument("--v2-nudge-mm", type=float, default=0.75)
    p.add_argument("--v2-nudge-observe-steps", type=int, default=40)
    p.add_argument("--v2-unstick-progress-mm", type=float, default=2.0)
    p.add_argument("--v2-min-deep-fraction", type=float, default=0.20)
    p.add_argument("--v2-min-lateral-wedge-fraction", type=float, default=0.20)
    p.add_argument("--v2-min-stall-depth-range-mm", type=float, default=15.0)
    p.add_argument("--v2-min-unstick-fraction", type=float, default=0.20)
    p.add_argument("--v2-max-lateral-mm", type=float, default=30.0,
                   help="gross-ejection bound retained from the contact calibration")
    p.add_argument("--v2-max-one-step-lateral-mm", type=float, default=30.0,
                   help="one-step numerical-ejection bound")
    p.add_argument("--controller", choices=("guided", "torchscript", "hybrid", "teacher"),
                   default="hybrid")
    p.add_argument("--model", type=Path,
                   default=Path("models/final_insert_sfp_flowstate_v1.ts"))
    p.add_argument("--teacher-zip", type=Path,
                   default=Path("RL/student_teacher/weights/teacher_level1.zip"))
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _finite_distribution(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not finite.size:
        return {
            "count": 0, "min": float("nan"), "p10": float("nan"),
            "median": float("nan"), "p90": float("nan"),
            "max": float("nan"), "mean": float("nan"),
            "std": float("nan"), "range": float("nan"),
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "range": float(np.max(finite) - np.min(finite)),
    }


def _v2_stall_window(trace: list[dict[str, Any]], args: argparse.Namespace):
    """Return the first sustained contact stall at any insertion depth."""
    for end in range(len(trace)):
        start = end
        while (start > 0
               and trace[end]["time_s"] - trace[start]["time_s"]
               < args.v2_stall_window_s):
            start -= 1
        window = trace[start:end + 1]
        if (not window
                or window[-1]["time_s"] - window[0]["time_s"]
                < args.v2_stall_window_s):
            continue
        depths = np.asarray([x["insertion_depth_mm"] for x in window])
        lateral = np.asarray([x["lateral_mm"] for x in window])
        forces = np.asarray([x["force_n"] for x in window])
        contacts = np.asarray([x["plug_port_contacts"] for x in window])
        progress_mm = float(np.max(depths) - np.min(depths))
        contact_fraction = float(np.mean(contacts > 0))
        if (float(np.median(depths)) < args.v2_stall_depth_min_mm
                or progress_mm > args.v2_stall_max_progress_mm
                or contact_fraction < 0.5):
            continue
        lateral_growth_mm = float(lateral[-1] - lateral[0])
        lateral_range_mm = float(np.max(lateral) - np.min(lateral))
        lateral_creep_mm = float(np.max(lateral) - lateral[0])
        peak_force_n = float(np.max(forces))
        low_force = peak_force_n <= args.v2_low_force_max_n
        bounded_lateral = float(np.max(lateral)) <= args.v2_wedge_max_lateral_mm
        if (low_force
                and bounded_lateral
                and lateral_creep_mm >= args.v2_lateral_growth_min_mm):
            classification = "lateral_wedge"
        elif (low_force
              and not bounded_lateral
              and lateral_creep_mm >= args.v2_lateral_growth_min_mm):
            classification = "excess_lateral"
        elif (not low_force
              and lateral_creep_mm < args.v2_lateral_growth_min_mm):
            classification = "axial_force_jam"
        else:
            classification = "mixed_or_flat_low_force"
        return {
            "start_index": int(start),
            "end_index": int(end),
            "start_step": int(window[0]["step"]),
            "end_step": int(window[-1]["step"]),
            "duration_s": float(window[-1]["time_s"] - window[0]["time_s"]),
            "stall_depth_mm": float(np.median(depths)),
            "depth_range_mm": [float(np.min(depths)), float(np.max(depths))],
            "depth_progress_mm": progress_mm,
            "lateral_start_mm": float(lateral[0]),
            "lateral_end_mm": float(lateral[-1]),
            "lateral_growth_mm": lateral_growth_mm,
            "lateral_creep_mm": lateral_creep_mm,
            "lateral_range_mm": lateral_range_mm,
            "peak_force_n": peak_force_n,
            "mean_force_n": float(np.mean(forces)),
            "contact_fraction": contact_fraction,
            "classification": classification,
        }
    return None


def _trace_row(step: int, env, info: dict[str, Any]) -> dict[str, Any]:
    remaining_mm = max(
        0.0,
        (float(env.scene.cfg.seated_depth_m)
         - float(info["insertion_depth_m"])) * 1e3,
    )
    return {
        "step": int(step),
        "time_s": float(info["wallclock"]),
        "depth_m": float(info["insertion_depth_m"]),
        "insertion_depth_mm": float(info["insertion_depth_m"]) * 1e3,
        "remaining_mm": remaining_mm,
        "mouth_gap_mm": float(info["approach_gap_m"]) * 1e3,
        "lateral_mm": float(info["lateral_error_m"]) * 1e3,
        "axis_deg": float(np.degrees(info["plug_axis_error_rad"])),
        "force_n": float(info["contact_force_norm"]),
        "plug_port_contacts": int(info["plug_port_contacts"]),
    }


def _reset_evidence(env, reset_info: dict[str, Any]):
    reset_diag = reset_info.get("reset_diag") or {}
    evidence = {
        key: float(reset_diag.get(key, float("nan")))
        for key in (
            "tip_error_m", "plug_axis_error_rad", "plug_roll_error_rad",
            "plug_port_penetration_m", "contact_force_norm",
            "lateral_error_m", "approach_gap_m",
        )
    }
    cfg = env.scene.cfg
    outside_port = evidence["approach_gap_m"] > 0.0
    lateral_limit = 0.0127 if outside_port else cfg.reset_inport_lateral_tol_m
    angle_limit = float(np.radians(5.0)) if outside_port else cfg.ik_axis_tol_rad
    valid = bool(
        evidence["lateral_error_m"] <= lateral_limit
        and evidence["plug_axis_error_rad"] <= angle_limit
        and evidence["plug_roll_error_rad"] <= angle_limit
        and evidence["plug_port_penetration_m"]
            <= cfg.reset_max_plug_port_penetration_m
        and evidence["contact_force_norm"] <= cfg.reset_contact_abort_n
    )
    return valid, evidence


def _jam_window(trace: list[dict[str, Any]], status: str,
                args: argparse.Namespace):
    for end in range(len(trace)):
        start = end
        while (start > 0
               and trace[end]["time_s"] - trace[start]["time_s"] < args.jam_window_s):
            start -= 1
        window = trace[start:end + 1]
        if (not window
                or window[-1]["time_s"] - window[0]["time_s"] < args.jam_window_s):
            continue
        depths = np.asarray([x["depth_m"] for x in window])
        insertion_depths = np.asarray([x["insertion_depth_mm"] for x in window])
        forces = np.asarray([x["force_n"] for x in window])
        contacts = np.asarray([x["plug_port_contacts"] for x in window])
        progress_mm = float((np.max(depths) - np.min(depths)) * 1e3)
        in_band = float(np.mean(
            (insertion_depths >= args.jam_depth_min_mm)
            & (insertion_depths <= args.jam_depth_max_mm)))
        joint_fraction = float(np.mean(
            (forces >= args.jam_min_force_n)
            & (contacts > 0)
            & (insertion_depths >= args.jam_depth_min_mm)
            & (insertion_depths <= args.jam_depth_max_mm)))
        if (progress_mm <= args.jam_max_progress_mm
                and joint_fraction >= 0.5
                and in_band >= 0.5):
            return {
                "kind": "sustained_contact_stall",
                "start_step": int(window[0]["step"]),
                "end_step": int(window[-1]["step"]),
                "duration_s": float(window[-1]["time_s"] - window[0]["time_s"]),
                "depth_progress_mm": progress_mm,
                "insertion_depth_mm_range": [
                    float(np.min(insertion_depths)),
                    float(np.max(insertion_depths)),
                ],
                "peak_force_n": float(np.max(forces)),
                "joint_force_contact_band_fraction": joint_fraction,
                "peak_lateral_mm": float(max(x["lateral_mm"] for x in window)),
                "peak_axis_deg": float(max(x["axis_deg"] for x in window)),
            }
    # The real failure ends with a 171 N safety abort. Preserve the simulator's
    # safety termination and accept the equivalent terminal signature without
    # requiring it to survive the full sustained-stall window.
    if status == "force_abort" and trace:
        final = trace[-1]
        recent = [x for x in trace
                  if final["time_s"] - x["time_s"] <= min(0.5, args.jam_window_s)]
        depths = np.asarray([x["depth_m"] for x in recent])
        progress_mm = float((np.max(depths) - np.min(depths)) * 1e3)
        if (args.jam_depth_min_mm
                <= final["insertion_depth_mm"] <= args.jam_depth_max_mm
                and final["plug_port_contacts"] > 0
                and final["force_n"] >= args.jam_min_force_n
                and progress_mm <= args.jam_max_progress_mm):
            return {
                "kind": "contact_force_abort",
                "start_step": int(recent[0]["step"]),
                "end_step": int(final["step"]),
                "duration_s": float(final["time_s"] - recent[0]["time_s"]),
                "depth_progress_mm": progress_mm,
                "insertion_depth_mm_range": [
                    float(min(x["insertion_depth_mm"] for x in recent)),
                    float(max(x["insertion_depth_mm"] for x in recent)),
                ],
                "peak_force_n": float(max(x["force_n"] for x in recent)),
                "peak_lateral_mm": float(max(x["lateral_mm"] for x in recent)),
                "peak_axis_deg": float(max(x["axis_deg"] for x in recent)),
            }
    return None


def _run(domain_randomization: bool, episodes: int, args: argparse.Namespace):
    student_action = None
    if args.controller in ("torchscript", "hybrid"):
        import torch
        policy = torch.jit.load(str(args.model), map_location="cpu").eval()

        def student_action(obs):
            with torch.no_grad():
                action = policy(torch.as_tensor(obs, dtype=torch.float32))
            return np.clip(
                np.asarray(action.cpu().numpy(), dtype=np.float32).reshape(6),
                -1.0, 1.0)

    env = make_student_env_a(
        perception_noise=(args.randomized_perception_noise
                          if domain_randomization else 0.0),
        grasp_noise=(args.randomized_grasp_noise
                     if domain_randomization else 0.0),
        level=args.level,
        action_convention="deploy",
        wrench_mode="baseline",
        domain_randomization=domain_randomization,
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )
    teacher_action = teacher_reset = None
    if args.controller == "teacher":
        from stable_baselines3 import SAC
        from RL.student_teacher.scripted_teacher_funnel import ScriptedTeacher
        from RL.student_teacher.train_student_a import (
            RESIDUAL_SCALE_DEFAULT, _teacher_target)
        teacher = SAC.load(args.teacher_zip, device="cpu")
        funnel = ScriptedTeacher(action_dim=6)

        def teacher_action(_obs):
            _sim, deploy = _teacher_target(
                env, funnel, teacher, RESIDUAL_SCALE_DEFAULT, "deploy")
            return deploy

        teacher_reset = funnel.reset
    results = []
    for episode in range(episodes):
        obs, reset_info = env.reset(
            seed=args.seed + episode,
            options={"level": args.level, "jitter": False},
        )
        if teacher_reset is not None:
            teacher_reset()
        reset_diag = reset_info.get("reset_diag") or {}
        cfg = env.scene.cfg
        reset_evidence = {
            key: float(reset_diag.get(key, float("nan")))
            for key in (
                "tip_error_m", "plug_axis_error_rad", "plug_roll_error_rad",
                "plug_port_penetration_m", "contact_force_norm",
                "lateral_error_m", "approach_gap_m",
            )
        }
        outside_port = reset_evidence["approach_gap_m"] > 0.0
        lateral_limit = 0.0127 if outside_port else cfg.reset_inport_lateral_tol_m
        angle_limit = float(np.radians(5.0)) if outside_port else cfg.ik_axis_tol_rad
        reset_valid = bool(
            reset_evidence["lateral_error_m"] <= lateral_limit
            and reset_evidence["plug_axis_error_rad"] <= angle_limit
            and reset_evidence["plug_roll_error_rad"] <= angle_limit
            and reset_evidence["plug_port_penetration_m"]
                <= cfg.reset_max_plug_port_penetration_m
            and reset_evidence["contact_force_norm"] <= cfg.reset_contact_abort_n
        )
        trace = []
        status = None if reset_valid else "invalid_reset"
        rl_engaged = args.controller == "torchscript"
        rl_engaged_step = 0 if rl_engaged else None
        for step in range(args.max_steps if reset_valid else 0):
            if args.controller == "guided":
                action = guided_action(obs)
            elif args.controller == "torchscript":
                action = student_action(obs)
            elif args.controller == "teacher":
                action = teacher_action(obs)
            else:
                delta = np.asarray(obs[32:35], dtype=np.float64)
                rotation = np.asarray(obs[35:38], dtype=np.float64)
                if (not rl_engaged
                        and delta[2] >= -0.002
                        and float(np.linalg.norm(delta[:2])) <= 0.002
                        and float(np.linalg.norm(rotation)) <= np.radians(2.0)):
                    rl_engaged = True
                    rl_engaged_step = step + 1
                action = student_action(obs) if rl_engaged else guided_action(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            remaining_mm = max(
                0.0,
                (float(env.scene.cfg.seated_depth_m)
                 - float(info["insertion_depth_m"])) * 1e3,
            )
            trace.append({
                "step": step + 1,
                "time_s": float(info["wallclock"]),
                "depth_m": float(info["insertion_depth_m"]),
                "insertion_depth_mm": float(info["insertion_depth_m"]) * 1e3,
                "remaining_mm": remaining_mm,
                "mouth_gap_mm": float(info["approach_gap_m"]) * 1e3,
                "lateral_mm": float(info["lateral_error_m"]) * 1e3,
                "axis_deg": float(np.degrees(info["plug_axis_error_rad"])),
                "force_n": float(info["contact_force_norm"]),
                "plug_port_contacts": int(info["plug_port_contacts"]),
                "rl_engaged": bool(rl_engaged),
            })
            if terminated or truncated:
                status = str(info.get("term_status") or "timeout")
                break
        jam = (None if status in ("success", "invalid_reset")
               else _jam_window(trace, status or "timeout", args))
        results.append({
            "episode": episode,
            "seed": args.seed + episode,
            "status": status or "timeout",
            "reset_valid": reset_valid,
            "reset_diag": reset_evidence,
            "jam": jam,
            "randomization": reset_info.get("domain_randomization", {}),
            "steps": len(trace),
            "rl_engaged_step": rl_engaged_step,
            "final": trace[-1] if trace else {},
            "trace": trace,
        })
    env.close()
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return {
        "episodes": episodes,
        "success_rate": counts.get("success", 0) / max(episodes, 1),
        "jam_count": sum(r["jam"] is not None for r in results),
        "counts": counts,
        "results": results,
    }


def _run_v2(args: argparse.Namespace) -> dict[str, Any]:
    """Probe the deep, low-force, lateral-wedge signature one variant per run."""
    directions = (
        ("plus_x", np.array([1.0, 0.0], dtype=np.float64)),
        ("minus_x", np.array([-1.0, 0.0], dtype=np.float64)),
        ("plus_y", np.array([0.0, 1.0], dtype=np.float64)),
        ("minus_y", np.array([0.0, -1.0], dtype=np.float64)),
    )
    results: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    bad_qacc_total = 0

    for episode in range(args.v2_episodes):
        variant_seed = args.seed + episode
        env = make_student_env_a(
            perception_noise=args.randomized_perception_noise,
            grasp_noise=args.randomized_grasp_noise,
            level=args.level,
            action_convention="deploy",
            wrench_mode="baseline",
            domain_randomization=True,
            max_episode_steps=args.max_steps,
            seed=variant_seed,
        )
        try:
            obs, reset_info = env.reset(
                seed=variant_seed,
                options={"level": args.level, "jitter": False},
            )
            reset_valid, reset_diag = _reset_evidence(env, reset_info)
            trace: list[dict[str, Any]] = []
            status = None if reset_valid else "invalid_reset"
            stall = None
            nudge_step = None
            nudge_direction_name = None
            nudge_start_depth_mm = None
            nudge_start_lateral_mm = None
            post_nudge_steps = 0

            for step in range(args.max_steps if reset_valid else 0):
                action = guided_action(obs)
                if stall is not None and nudge_step is None:
                    nudge_direction_name, direction = directions[episode % len(directions)]
                    action = action.copy()
                    action[:2] = direction * (
                        (args.v2_nudge_mm * 1e-3)
                        / np.asarray(DEPLOY_POS_SCALE[:2], dtype=np.float64)
                    )
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)
                    nudge_step = step + 1
                    nudge_start_depth_mm = float(trace[-1]["insertion_depth_mm"])
                    nudge_start_lateral_mm = float(trace[-1]["lateral_mm"])

                obs, _reward, terminated, truncated, info = env.step(action)
                trace.append(_trace_row(step + 1, env, info))

                if stall is None:
                    stall = _v2_stall_window(trace, args)
                elif nudge_step is not None:
                    post_nudge_steps += 1

                if terminated or truncated:
                    status = str(info.get("term_status") or "timeout")
                    break
                if (nudge_step is not None
                        and post_nudge_steps >= args.v2_nudge_observe_steps):
                    status = "probe_complete"
                    break

            status = status or "timeout"
            counts[status] = counts.get(status, 0) + 1

            prefix = trace[:stall["start_index"] + 1] if stall else trace
            running_peak_n = 0.0
            low_force_depths = []
            for row in prefix:
                running_peak_n = max(running_peak_n, float(row["force_n"]))
                if running_peak_n <= args.v2_low_force_max_n:
                    low_force_depths.append(float(row["insertion_depth_mm"]))
            low_force_max_depth_mm = (
                max(low_force_depths) if low_force_depths else float("nan"))
            deep_low_force = bool(
                np.isfinite(low_force_max_depth_mm)
                and low_force_max_depth_mm >= args.v2_deep_min_mm)

            post_nudge = (
                trace[nudge_step - 1:] if nudge_step is not None else [])
            post_nudge_max_depth_mm = (
                max(x["insertion_depth_mm"] for x in post_nudge)
                if post_nudge else float("nan"))
            post_nudge_depth_progress_mm = (
                post_nudge_max_depth_mm - float(nudge_start_depth_mm)
                if post_nudge and nudge_start_depth_mm is not None
                else float("nan"))
            post_nudge_lateral_move_mm = (
                max(abs(x["lateral_mm"] - float(nudge_start_lateral_mm))
                    for x in post_nudge)
                if post_nudge and nudge_start_lateral_mm is not None
                else float("nan"))
            unstick_success = bool(
                nudge_step is not None
                and (status == "success"
                     or post_nudge_depth_progress_mm
                     >= args.v2_unstick_progress_mm))

            bad_qacc = int(env.scene.data.warning[
                int(mujoco.mjtWarning.mjWARN_BADQACC)].number)
            bad_qacc_total += bad_qacc
            lateral_mm = np.asarray(
                [x["lateral_mm"] for x in trace], dtype=np.float64)
            max_lateral_mm = (
                float(np.max(lateral_mm)) if lateral_mm.size else float("nan"))
            max_one_step_lateral_mm = (
                float(np.max(np.abs(np.diff(lateral_mm))))
                if lateral_mm.size > 1 else 0.0)
            numerical_ejection = bool(
                max_lateral_mm > args.v2_max_lateral_mm
                or max_one_step_lateral_mm > args.v2_max_one_step_lateral_mm)
            results.append({
                "episode": episode,
                "variant_seed": variant_seed,
                "status": status,
                "reset_valid": reset_valid,
                "reset_diag": reset_diag,
                "randomization": reset_info.get("domain_randomization", {}),
                "steps": len(trace),
                "deep_low_force": deep_low_force,
                "low_force_max_depth_mm": low_force_max_depth_mm,
                "prefix_peak_force_n": (
                    max((x["force_n"] for x in prefix), default=float("nan"))),
                "stall": stall,
                "nudge": {
                    "eligible": bool(stall is not None and nudge_step is not None),
                    "step": nudge_step,
                    "direction": nudge_direction_name,
                    "commanded_mm": args.v2_nudge_mm,
                    "observed_lateral_move_mm": post_nudge_lateral_move_mm,
                    "depth_progress_mm": post_nudge_depth_progress_mm,
                    "max_depth_mm": post_nudge_max_depth_mm,
                    "seated": status == "success",
                    "unstick_success": unstick_success,
                },
                "bad_qacc_warnings": bad_qacc,
                "max_lateral_mm": max_lateral_mm,
                "max_one_step_lateral_mm": max_one_step_lateral_mm,
                "numerical_ejection": numerical_ejection,
                "final": trace[-1] if trace else {},
                "trace": trace,
            })
        finally:
            env.close()

    valid = [r for r in results if r["reset_valid"]]
    stable_valid = [r for r in valid if not r["numerical_ejection"]]
    stalls = [r for r in stable_valid if r["stall"] is not None]
    lateral_wedges = [
        r for r in stalls if r["stall"]["classification"] == "lateral_wedge"]
    axial_jams = [
        r for r in stalls if r["stall"]["classification"] == "axial_force_jam"]
    mixed_stalls = [
        r for r in stalls
        if r["stall"]["classification"] == "mixed_or_flat_low_force"]
    excess_lateral_stalls = [
        r for r in stalls if r["stall"]["classification"] == "excess_lateral"]
    nudged = [r for r in lateral_wedges if r["nudge"]["eligible"]]
    unstuck = [r for r in nudged if r["nudge"]["unstick_success"]]
    denominator = max(len(valid), 1)
    stall_denominator = max(len(stalls), 1)
    nudge_denominator = max(len(nudged), 1)
    low_force_distribution = _finite_distribution([
        r["low_force_max_depth_mm"] for r in valid])
    stall_depth_distribution = _finite_distribution([
        r["stall"]["stall_depth_mm"] for r in stalls])
    wedge_depth_distribution = _finite_distribution([
        r["stall"]["stall_depth_mm"] for r in lateral_wedges])
    nudge_progress_distribution = _finite_distribution([
        r["nudge"]["depth_progress_mm"] for r in nudged])
    max_lateral_distribution = _finite_distribution([
        r["max_lateral_mm"] for r in valid])
    max_one_step_lateral_distribution = _finite_distribution([
        r["max_one_step_lateral_mm"] for r in valid])
    numerical_ejection_count = sum(r["numerical_ejection"] for r in valid)
    deep_fraction = sum(r["deep_low_force"] for r in valid) / denominator
    lateral_wedge_fraction = len(lateral_wedges) / stall_denominator
    unstick_fraction = len(unstuck) / nudge_denominator
    stable_denominator = max(len(stable_valid), 1)
    stable_deep_fraction = (
        sum(r["deep_low_force"] for r in stable_valid) / stable_denominator)
    matches_flowstate = bool(
        stable_valid
        and stable_deep_fraction >= args.v2_min_deep_fraction
        and len(lateral_wedges) >= 2
        and lateral_wedge_fraction >= args.v2_min_lateral_wedge_fraction
        and wedge_depth_distribution["range"]
            >= args.v2_min_stall_depth_range_mm
        and nudged
        and unstick_fraction >= args.v2_min_unstick_fraction
        and bad_qacc_total == 0
        and numerical_ejection_count == 0
    )
    return {
        "episodes": args.v2_episodes,
        "valid_episodes": len(valid),
        "counts": counts,
        "criteria": {
            "deep_min_mm": args.v2_deep_min_mm,
            "low_force_max_n": args.v2_low_force_max_n,
            "stall_depth_min_mm": args.v2_stall_depth_min_mm,
            "stall_window_s": args.v2_stall_window_s,
            "stall_max_progress_mm": args.v2_stall_max_progress_mm,
            "lateral_growth_min_mm": args.v2_lateral_growth_min_mm,
            "wedge_max_lateral_mm": args.v2_wedge_max_lateral_mm,
            "nudge_mm": args.v2_nudge_mm,
            "unstick_progress_mm": args.v2_unstick_progress_mm,
            "max_lateral_mm": args.v2_max_lateral_mm,
            "max_one_step_lateral_mm": args.v2_max_one_step_lateral_mm,
        },
        "deep_low_force": {
            "count": int(sum(r["deep_low_force"] for r in stable_valid)),
            "fraction": stable_deep_fraction,
            "max_depth_distribution_mm": _finite_distribution([
                r["low_force_max_depth_mm"] for r in stable_valid]),
            "all_reset_valid_fraction_including_ejections": deep_fraction,
            "all_reset_valid_depth_distribution_mm": low_force_distribution,
        },
        "stall": {
            "count": len(stalls),
            "lateral_wedge_count": len(lateral_wedges),
            "lateral_wedge_fraction": lateral_wedge_fraction,
            "axial_force_jam_count": len(axial_jams),
            "axial_force_jam_fraction": len(axial_jams) / stall_denominator,
            "mixed_or_flat_low_force_count": len(mixed_stalls),
            "mixed_or_flat_low_force_fraction": len(mixed_stalls) / stall_denominator,
            "excess_lateral_count": len(excess_lateral_stalls),
            "excess_lateral_fraction": len(excess_lateral_stalls) / stall_denominator,
            "depth_distribution_mm": stall_depth_distribution,
            "lateral_wedge_depth_distribution_mm": wedge_depth_distribution,
        },
        "lateral_unstick": {
            "eligible_count": len(nudged),
            "success_count": len(unstuck),
            "success_fraction": unstick_fraction,
            "depth_progress_distribution_mm": nudge_progress_distribution,
        },
        "bad_qacc_warnings": bad_qacc_total,
        "stability": {
            "numerical_ejection_count": int(numerical_ejection_count),
            "max_lateral_distribution_mm": max_lateral_distribution,
            "max_one_step_lateral_distribution_mm": (
                max_one_step_lateral_distribution),
        },
        "matches_flowstate": matches_flowstate,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    if args.controller in ("torchscript", "hybrid") and not args.model.is_file():
        raise SystemExit(f"TorchScript model not found: {args.model}")
    if args.controller == "teacher":
        expected_teacher = (
            "fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4")
        actual_teacher = hashlib.sha256(args.teacher_zip.read_bytes()).hexdigest()
        if actual_teacher != expected_teacher:
            raise SystemExit("frozen teacher hash mismatch")
    nominal = _run(False, args.nominal_episodes, args)
    randomized = _run(True, args.randomized_episodes, args)
    legacy_mouth_passed = (
        nominal["success_rate"] >= args.nominal_min_success
        and randomized["jam_count"] >= 1
    )
    v2 = _run_v2(args)
    # The old mouth-band gate stays in the report as a regression diagnostic,
    # but it is intentionally not a v2 acceptance condition: Flowstate now
    # demonstrates catches well beyond that obsolete 5--9 mm band.
    passed = bool(
        nominal["success_rate"] >= args.nominal_min_success
        and v2["matches_flowstate"])
    report = {
        "gate": "gate0_contact_jam_v2",
        "controller": args.controller,
        "model": (str(args.model)
                  if args.controller in ("torchscript", "hybrid") else None),
        "model_sha256": (
            hashlib.sha256(args.model.read_bytes()).hexdigest()
            if args.controller in ("torchscript", "hybrid") else None),
        "teacher_zip": (str(args.teacher_zip) if args.controller == "teacher" else None),
        "teacher_sha256": (
            hashlib.sha256(args.teacher_zip.read_bytes()).hexdigest()
            if args.controller == "teacher" else None),
        "passed": passed,
        "criteria": {
            "fixed_curriculum_level": args.level,
            "randomized_perception_noise": args.randomized_perception_noise,
            "randomized_grasp_noise": args.randomized_grasp_noise,
            "nominal_min_success": args.nominal_min_success,
            "jam_min_force_n": args.jam_min_force_n,
            "jam_window_s": args.jam_window_s,
            "jam_max_progress_mm": args.jam_max_progress_mm,
            "jam_insertion_depth_mm": [
                args.jam_depth_min_mm, args.jam_depth_max_mm],
        },
        "legacy_mouth_gate_passed": legacy_mouth_passed,
        "nominal": nominal,
        "randomized": randomized,
        "v2": v2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "gate": report["gate"],
        "passed": passed,
        "nominal_success_rate": nominal["success_rate"],
        "nominal_counts": nominal["counts"],
        "legacy_mouth_gate_passed": legacy_mouth_passed,
        "randomized_jam_count": randomized["jam_count"],
        "randomized_counts": randomized["counts"],
        "v2_valid_episodes": v2["valid_episodes"],
        "v2_deep_low_force_fraction": v2["deep_low_force"]["fraction"],
        "v2_low_force_depth_distribution_mm": (
            v2["deep_low_force"]["max_depth_distribution_mm"]),
        "v2_stall_split": v2["stall"],
        "v2_lateral_unstick": v2["lateral_unstick"],
        "v2_bad_qacc_warnings": v2["bad_qacc_warnings"],
        "v2_stability": v2["stability"],
        "v2_matches_flowstate": v2["matches_flowstate"],
        "output": str(args.output),
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
