"""Capture deterministic MuJoCo observations at a Flowstate-matched handoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from RL.student_teacher.student_env_a import make_student_env_a
from RL.student_teacher.train_student_a import build_policy


FIELDS = (
    ("joint_offset", 0, 6),
    ("joint_velocity", 6, 12),
    ("tcp_pose_world", 12, 19),
    ("tcp_velocity_port", 19, 25),
    ("port_pose_world", 25, 32),
    ("tip_delta_port", 32, 35),
    ("tip_rotation_error_port", 35, 38),
    ("alignment_hint", 38, 44),
    ("scripted_hint", 44, 50),
    ("bias", 50, 51),
    ("wrench", 51, 57),
    ("last_action", 57, 63),
    ("tip_axes_port", 63, 69),
)


def _named(obs: np.ndarray) -> dict[str, list[float]]:
    return {name: obs[start:end].astype(float).tolist() for name, start, end in FIELDS}


def _load_policy(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    config = dict(checkpoint.get("config", {}))
    hidden = int(config.get("hidden", state["net.0.weight"].shape[0]))
    feature_mode = config.get("feature_mode", "legacy")
    policy = build_policy(hidden=hidden, feature_mode=feature_mode)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--torchscript", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--flowstate-depth-mm", type=float, default=-21.887)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 132)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy, config = _load_policy(args.checkpoint)
    scripted = (
        torch.jit.load(str(args.torchscript), map_location="cpu")
        if args.torchscript
        else None
    )
    if scripted is not None:
        scripted.eval()

    env = make_student_env_a(
        perception_noise=0.0,
        grasp_noise=0.0,
        level=1.0,
        action_convention="deploy",
        wrench_mode="baseline",
        seed=args.seeds[0],
    )
    seated_depth = float(env.scene.cfg.seated_depth_m)
    last_inch = float(env.scene.cfg.last_inch_m)
    target_depth = args.flowstate_depth_mm / 1000.0
    retract = seated_depth - target_depth
    level = float(np.clip(retract / last_inch, 0.0, 1.0))

    records = []
    try:
        for seed in args.seeds:
            obs, info = env.reset(seed=seed, options={"level": level, "jitter": False})
            obs = np.asarray(obs, dtype=np.float32)
            with torch.no_grad():
                checkpoint_action = policy(torch.from_numpy(obs[None])).numpy().reshape(-1)
                scripted_action = (
                    scripted(torch.from_numpy(obs[None])).numpy().reshape(-1)
                    if scripted is not None
                    else None
                )
            diag = dict(info.get("reset_diag") or {})
            records.append({
                "seed": seed,
                "observation": obs.astype(float).tolist(),
                "named": _named(obs),
                "checkpoint_action": checkpoint_action.astype(float).tolist(),
                "torchscript_action": (
                    scripted_action.astype(float).tolist() if scripted_action is not None else None
                ),
                "reset_diag": {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in diag.items()
                },
            })
    finally:
        env.close()

    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "torchscript": str(args.torchscript.resolve()) if args.torchscript else None,
        "checkpoint_config": config,
        "flowstate_target_depth_mm": args.flowstate_depth_mm,
        "mujoco_reset_level": level,
        "seated_depth_m": seated_depth,
        "last_inch_m": last_inch,
        "perception_noise": 0.0,
        "grasp_noise": 0.0,
        "wrench_mode": "baseline",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "records": len(records),
        "target_depth_mm": args.flowstate_depth_mm,
        "level": level,
        "first_action": records[0]["checkpoint_action"],
    }, indent=2))


if __name__ == "__main__":
    main()
