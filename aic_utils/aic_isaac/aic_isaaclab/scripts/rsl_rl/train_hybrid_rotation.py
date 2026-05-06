"""Launcher for hybrid insertion training.

This path replaces the old "learn everything at once" setup:
1) Classical priors/reset bring the plug near the target port.
2) RL focuses on final rotation/twist + seating for insertion success.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run hybrid rotation-focused RL training for AIC insertion."
    )
    parser.add_argument("--task", type=str, default="AIC-Insert-Hybrid-SC-v0")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--max_iterations", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_name", type=str, default="hybrid_rotation")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_interval", type=int, default=2000)
    parser.add_argument("--video_length", type=int, default=600)
    args, passthrough = parser.parse_known_args()

    train_py = os.path.join(os.path.dirname(__file__), "train.py")
    cmd = [
        sys.executable,
        train_py,
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--device",
        args.device,
        "--run_name",
        args.run_name,
        "--video_interval",
        str(args.video_interval),
        "--video_length",
        str(args.video_length),
    ]
    if args.video:
        cmd.append("--video")
    cmd.extend(passthrough)

    print("[INFO] Hybrid training command:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
