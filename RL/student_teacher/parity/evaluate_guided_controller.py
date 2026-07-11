"""Evaluate the perception-guided insertion controller in MuJoCo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from RL.student_teacher.student_env_a import (
    DEPLOY_POS_SCALE,
    DEPLOY_ROT_SCALE,
    make_student_env_a,
)


def guided_action(obs: np.ndarray) -> np.ndarray:
    """Cancel relative pose error and advance only along the port axis."""
    obs = np.asarray(obs, dtype=np.float64).reshape(69)
    delta = obs[32:35]
    rotation_error = obs[35:38]
    axial_step = 0.0015 if delta[2] < -0.003 else 0.00075
    translation = np.array([-delta[0], -delta[1], axial_step])
    action = np.concatenate(
        [translation / DEPLOY_POS_SCALE, -rotation_error / DEPLOY_ROT_SCALE]
    )
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10001, 20002, 30003])
    parser.add_argument("--perception-noise", type=float, default=1.0)
    parser.add_argument("--grasp-noise", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed = []
    combined_counts: dict[str, int] = {}
    for seed in args.seeds:
        env = make_student_env_a(
            perception_noise=args.perception_noise,
            grasp_noise=args.grasp_noise,
            level=1.0,
            action_convention="deploy",
            wrench_mode="baseline",
            seed=seed,
        )
        counts: dict[str, int] = {}
        obs, _ = env.reset(seed=seed)
        done = 0
        while done < args.episodes_per_seed:
            obs, _reward, terminated, truncated, info = env.step(guided_action(obs))
            if terminated or truncated:
                status = str(info.get("term_status"))
                counts[status] = counts.get(status, 0) + 1
                combined_counts[status] = combined_counts.get(status, 0) + 1
                done += 1
                obs, _ = env.reset()
        env.close()
        successes = counts.get("success", 0)
        per_seed.append(
            {
                "seed": seed,
                "success_rate": successes / args.episodes_per_seed,
                "counts": counts,
            }
        )

    total = args.episodes_per_seed * len(args.seeds)
    result = {
        "controller": "perception_guided_absolute_pose_proxy_v1",
        "episodes_per_seed": args.episodes_per_seed,
        "seeds": args.seeds,
        "perception_noise": args.perception_noise,
        "grasp_noise": args.grasp_noise,
        "total_episodes": total,
        "overall_success_rate": combined_counts.get("success", 0) / total,
        "combined_counts": combined_counts,
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)

    if combined_counts.get("bad_collision", 0):
        raise SystemExit("guided evaluation contains bad_collision failures")


if __name__ == "__main__":
    main()
