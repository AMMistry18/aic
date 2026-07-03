"""Smoke test: load RL/mujoco_env.py, run a random-agent loop, print a
one-line summary per step. No W&B yet — that's `connect_wandb.py`.

Usage:
    pixi run python RL/scripts/setup_mujoco.py --steps 50
    AIC_MJCF_SCENE=/abs/path/to/scene.xml pixi run python RL/scripts/setup_mujoco.py

Expected: prints `[smoke] env loaded: .../aic_utils/aic_mujoco/mjcf/scene.xml`,
a one-line per step, then `[smoke] 50 steps in ~Xs (~Y steps/s)`.

Goal: scene.xml parses, `mj_step` runs without error, step rate > 1000/s
on a single env (this is the headroom 4096 envs will need).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL.mujoco_env import MuJoCoEnvConfig, MuJoCoLastInchEnv


def main() -> int:
    p = argparse.ArgumentParser(
        description="Smoke test the SDF-exported MuJoCo scene + last-inch env."
    )
    p.add_argument("--steps", type=int, default=50,
                   help="number of steps to run (default 50)")
    p.add_argument("--scene", type=str, default=None,
                   help="override AIC_MJCF_SCENE for this run")
    p.add_argument("--seed", type=int, default=0,
                   help="reset seed (default 0)")
    args = p.parse_args()

    cfg = MuJoCoEnvConfig()
    if args.scene:
        cfg.scene_path = Path(args.scene)

    env = MuJoCoLastInchEnv(cfg)
    obs, _ = env.reset(seed=args.seed)
    print(f"[smoke] env loaded: {cfg.scene_path}", flush=True)
    print(f"[smoke] obs_space keys: {list(obs.keys())}", flush=True)
    print(f"[smoke] image shape: {obs['image'].shape}", flush=True)
    print(f"[smoke] action_space: {env.action_space}", flush=True)
    print(f"[smoke] max_episode_steps: {cfg.max_episode_steps}", flush=True)

    t0 = time.time()
    for step in range(args.steps):
        a = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(a)
        if step % 10 == 0 or step == args.steps - 1:
            print(
                f"[smoke] step {step:>4d} reward={rew:+.3f} "
                f"term={term} trunc={trunc}",
                flush=True,
            )
        if term or trunc:
            obs, _ = env.reset()
    elapsed = time.time() - t0
    print(
        f"[smoke] {args.steps} steps in {elapsed:.2f}s "
        f"({args.steps / elapsed:.0f} steps/s)",
        flush=True,
    )

    # one render to prove the renderer path works (skipped if --no-render)
    if args.steps > 0:
        try:
            frame = env.render()
            print(f"[smoke] render() -> shape={frame.shape} dtype={frame.dtype}",
                  flush=True)
        except Exception as exc:
            print(f"[smoke] render() failed: {exc}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())