#!/usr/bin/env python3
"""Run finite-state and timing checks against the Isaac task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-LastInch-SFP-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import aic_isaac_sim  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env = None
    try:
        cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        assert abs(float(cfg.sim.dt) - 0.002) < 1.0e-12
        assert int(cfg.decimation) == 25
        env = gym.make(args.task, cfg=cfg)
        obs, _ = env.reset()
        policy = obs["policy"]
        assert policy.shape == (args.num_envs, 31), policy.shape
        assert torch.isfinite(policy).all()
        actions = torch.zeros((args.num_envs, 6), device=policy.device)
        for _ in range(args.steps):
            obs, rewards, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert terminated.shape == (args.num_envs,)
            assert truncated.shape == (args.num_envs,)
        print(
            f"PASS: {args.num_envs} envs, {args.steps} actions, "
            "500 Hz physics, 20 Hz policy, finite 31-D observations"
        )
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
