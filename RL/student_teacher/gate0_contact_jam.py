"""Gate 0: prove randomized MuJoCo can express the Flowstate contact jam.

The deterministic guided controller is the sentinel because it is already known
to seat at calibrated nominal poses. Gate 0 passes only when that same controller
remains successful nominally and at least one randomized episode shows sustained
force with little progress in the observed 5--9 mm insertion-depth band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import hashlib

import numpy as np

from RL.student_teacher.parity.evaluate_guided_controller import guided_action
from RL.student_teacher.student_env_a import make_student_env_a


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
    p.add_argument("--controller", choices=("guided", "torchscript", "hybrid", "teacher"),
                   default="hybrid")
    p.add_argument("--model", type=Path,
                   default=Path("models/final_insert_sfp_flowstate_v1.ts"))
    p.add_argument("--teacher-zip", type=Path,
                   default=Path("RL/student_teacher/weights/teacher_level1.zip"))
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


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
    passed = (
        nominal["success_rate"] >= args.nominal_min_success
        and randomized["jam_count"] >= 1
    )
    report = {
        "gate": "gate0_contact_jam_v1",
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
        "nominal": nominal,
        "randomized": randomized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "gate": report["gate"],
        "passed": passed,
        "nominal_success_rate": nominal["success_rate"],
        "nominal_counts": nominal["counts"],
        "randomized_jam_count": randomized["jam_count"],
        "randomized_counts": randomized["counts"],
        "output": str(args.output),
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
