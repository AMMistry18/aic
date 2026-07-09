#!/usr/bin/env python3
"""Launch skrl SAC for the external AIC Isaac Lab task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-LastInch-SFP-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_timesteps", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import aic_isaac_sim  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        agent_cfg = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
        agent_cfg["seed"] = args.seed
        env_cfg.seed = args.seed
        if args.max_timesteps is not None:
            agent_cfg["trainer"]["timesteps"] = args.max_timesteps
        env = gym.make(args.task, cfg=env_cfg)
        runner = Runner(SkrlVecEnvWrapper(env), agent_cfg)
        runner.run()
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
