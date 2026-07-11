"""Held-out multi-seed evaluation and directional checks for a trained student."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from RL.student_teacher.student_env_a import make_student_env_a
from RL.student_teacher.train_student_a import build_policy, eval_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10001, 20002, 30003])
    parser.add_argument("--min-success", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    config = dict(checkpoint.get("config", {}))
    hidden = int(config.get("hidden", state["net.0.weight"].shape[0]))
    feature_mode = config.get("feature_mode", "legacy")
    policy = build_policy(hidden=hidden, feature_mode=feature_mode)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    per_seed = []
    combined_counts: dict[str, int] = {}
    total_success = 0
    total_episodes = 0
    for seed in args.seeds:
        success_rate, counts = eval_student(
            policy,
            episodes=args.episodes_per_seed,
            perception_noise=1.0,
            level=1.0,
            action_convention="deploy",
            wrench_mode="baseline",
            grasp_noise=1.0,
            seed=seed,
            device="cpu",
        )
        successes = int(counts.get("success", 0))
        total_success += successes
        total_episodes += args.episodes_per_seed
        for name, count in counts.items():
            combined_counts[name] = combined_counts.get(name, 0) + int(count)
        per_seed.append({"seed": seed, "success_rate": success_rate, "counts": counts})

    env = make_student_env_a(
        perception_noise=1.0,
        grasp_noise=1.0,
        level=1.0,
        action_convention="deploy",
        wrench_mode="baseline",
        seed=40004,
    )
    axial_actions = []
    lateral_dot_error = []
    try:
        for seed in range(40004, 40104):
            obs, _ = env.reset(seed=seed)
            with torch.no_grad():
                action = policy(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            axial_actions.append(float(action[2]))
            lateral_error = np.asarray(obs[32:34], dtype=np.float64)
            if np.linalg.norm(lateral_error) > 1e-5:
                lateral_dot_error.append(float(np.dot(action[:2], lateral_error)))
    finally:
        env.close()

    overall = total_success / max(1, total_episodes)
    directional = {
        "samples": len(axial_actions),
        "positive_axial_fraction": float(np.mean(np.asarray(axial_actions) > 0.0)),
        "median_axial_action": float(np.median(axial_actions)),
        "lateral_opposition_fraction": float(
            np.mean(np.asarray(lateral_dot_error) < 0.0)
        ),
        "median_lateral_action_dot_error": float(np.median(lateral_dot_error)),
    }
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "wrench_mode": "baseline",
        "episodes_per_seed": args.episodes_per_seed,
        "per_seed": per_seed,
        "total_episodes": total_episodes,
        "overall_success_rate": overall,
        "combined_counts": combined_counts,
        "directional_sanity": directional,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    if overall < args.min_success:
        raise SystemExit(f"held-out success {overall:.3f} < {args.min_success:.3f}")
    if int(combined_counts.get("bad_collision", 0)) != 0:
        raise SystemExit("held-out evaluation contains bad_collision failures")
    if directional["median_axial_action"] <= 0.0:
        raise SystemExit("median outside-mouth axial action is not positive/inward")
    if directional["median_lateral_action_dot_error"] >= 0.0:
        raise SystemExit("median lateral action does not oppose lateral error")


if __name__ == "__main__":
    main()
