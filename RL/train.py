"""
Train residual SAC for the last-inch insertion policy.

Usage:
    pixi run python RL/train.py --out outputs/residual_sac_v0 --steps 200000 \
        --port-type sc

This script:
    1. Builds the MuJoCo env (RL/env.py).
    2. Wraps it with a Dict obs space and a CombinedExtractor that has a
       small CNN for the image channel and an MLP for the state vector.
    3. Trains SAC (stable-baselines3) with the reward defined in
       RL/reward.py.
    4. Saves the policy as a TorchScript checkpoint so the deployment
       policy (LastInchInsert.py) can load it via the existing
       PerceptionInsert._load_final_insert_policy path.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np

# We expect this file to be invoked as `pixi run python RL/train.py` from
# the aic/ directory. Make `RL` importable even when run from elsewhere.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _has_tensorboard() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    p = argparse.ArgumentParser(description="Train residual SAC for last-inch insertion")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory for checkpoint + logs + videos. "
                        "Default: RL/output/<run-name>/ so artifacts stay "
                        "alongside the source code.")
    p.add_argument("--steps", type=int, default=200_000,
                   help="total env steps (default 200k per REWARD_SPEC §8)")
    p.add_argument("--port-type", choices=["sc", "sfp"], default="sc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--num-envs", type=int, default=1,
                   help="number of parallel envs (SubprocVecEnv). 1 = "
                        "DummyVecEnv (no forking). 8+ = strong speedup on "
                        "multi-core machines like the 5090 box.")
    p.add_argument("--log-every", type=int, default=500,
                   help="print progress to stdout every N env steps")
    p.add_argument("--no-torchscript", action="store_true",
                   help="skip the TorchScript export (useful for smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="continue training from existing model.zip in --out if it exists")
    p.add_argument("--no-skip-if-cached", dest="skip_if_cached", action="store_false",
                   help="re-train even if a cached run with the same key exists")
    p.add_argument("--video-every", type=int, default=25,
                   help="record a roll-out video every N episodes (0 to disable)")
    p.add_argument("--no-video", action="store_true",
                   help="disable video recording entirely")
    p.add_argument("--plot", action="store_true",
                   help="render a 2x2 dashboard PNG from metrics.jsonl after training")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing --out directory")
    p.add_argument("--recorded", action="store_true",
                   help="use RecordedRolloutEnv (offline RL from .npz "
                        "recordings of the AIC engine) instead of the "
                        "live MuJoCo LastInchInsertEnv.")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="directory containing .npz rollouts (used with "
                        "--recorded). Recursively searched for *.npz.")
    p.add_argument("--last-inch-start-frame", type=int, default=80,
                   help="recorded-only: how many frames from the end "
                        "of each rollout count as the 'last inch' for "
                        "curriculum starts. Default 80 (4s at 20Hz).")
    p.add_argument("--reset-mode", choices=["curriculum", "random", "near_goal"],
                   default="curriculum",
                   help="start-pose distribution. 'curriculum' = reverse "
                        "curriculum (start near goal, expand outward as "
                        "policy succeeds). 'random' = uniform over the "
                        "full last-inch envelope. 'near_goal' = always near.")
    p.add_argument("--curriculum-eval-window", type=int, default=100,
                   help="number of episodes to evaluate success rate over")
    p.add_argument("--curriculum-advance-threshold", type=float, default=0.6,
                   help="advance to next level when success rate >= this")
    p.add_argument("--curriculum-retreat-threshold", type=float, default=0.15,
                   help="retreat to previous level when success rate <= this")
    p.add_argument("--curriculum-step", type=float, default=0.05,
                   help="level increment per advance/retreat event")
    # ---- Weights & Biases ----
    p.add_argument("--wandb", dest="wandb", action="store_true",
                   default=None,
                   help="enable W&B logging (default: on if WANDB_API_KEY "
                        "is set, else off)")
    p.add_argument("--no-wandb", dest="wandb", action="store_false",
                   help="force-disable W&B even if WANDB_API_KEY is set")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="optional run name shown in the W&B UI")
    p.add_argument("--wandb-log-every", type=int, default=500,
                   help="SAC-loss / lr log cadence (env steps)")
    p.add_argument("--wandb-success-window", type=int, default=100,
                   help="rolling-window size for reward/mean and "
                        "eval/success_rate")
    args = p.parse_args()

    # default --out goes under RL/output/<run-name>/ so artifacts stay
    # next to the source. port_type + reset_mode tag the run name so
    # multiple runs don't clobber each other.
    if args.out is None:
        run_name = f"residual_sac_{args.port_type}_{args.reset_mode}"
        if args.recorded:
            run_name += "_recorded"
        args.out = Path(__file__).resolve().parent / "output" / run_name
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[train] saving to {args.out.resolve()}", flush=True)

    # ------------------------------------------------------------------ #
    # cache check
    # ------------------------------------------------------------------ #

    from RL.cache import cache_hit, make_cache_key, write_config

    cfg_dict = dict(vars(args))
    # remove non-config items from the key
    cfg_dict.pop("force", None)
    cache_key = make_cache_key(cfg_dict)
    print(f"[train] cache_key={cache_key}", flush=True)

    if args.skip_if_cached:
        hit = cache_hit(args.out, cache_key)
        if hit is not None and not args.resume:
            print(f"[train] CACHE HIT: {hit} matches cache_key={cache_key}", flush=True)
            print(f"[train] skipping training. Use --no-skip-if-cached to re-train, "
                  f"or --resume to continue.", flush=True)
            return

    write_config(args.out, cfg_dict, cache_key)

    # Heavy imports after sys.path fix
    import torch
    import torch.nn as nn
    from stable_baselines3 import SAC
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from gymnasium.wrappers import TimeLimit

    from RL.env import EnvConfig, LastInchInsertEnv

    # ------------------------------------------------------------------ #
    # env
    # ------------------------------------------------------------------ #

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

    env_cfg = EnvConfig()
    n_envs = int(args.num_envs)

    # ------------------------------------------------------------------ #
    # env factory — chooses between live MuJoCo sim and offline recorded
    # ------------------------------------------------------------------ #
    use_recorded = bool(args.recorded)
    if use_recorded:
        from RL.recorded_env import (
            RecordedEnvConfig, RecordedRolloutEnv, discover_dataset,
        )
        if args.dataset_dir is None or not Path(args.dataset_dir).exists():
            raise SystemExit(
                f"--recorded requires --dataset-dir pointing at a directory "
                f"of .npz rollouts (got {args.dataset_dir})"
            )
        paths = discover_dataset(str(args.dataset_dir), port_type=args.port_type)
        if not paths:
            raise SystemExit(
                f"no rollouts found under {args.dataset_dir} matching "
                f"port_type={args.port_type!r}"
            )
        print(f"[train] recorded env: {len(paths)} rollouts from {args.dataset_dir}",
              flush=True)
        rec_cfg = RecordedEnvConfig(
            obs=env_cfg.obs, reward=env_cfg.reward, term=env_cfg.term,
            port_type=args.port_type,
            last_inch_start_frame=int(args.last_inch_start_frame),
        )

        def _make_env(rank: int = 0):
            def _thunk():
                # shard the dataset per worker so each subprocess reads
                # a different subset (faster overall and avoids
                # contending on the same npz files)
                shard = paths[rank::n_envs]
                e = RecordedRolloutEnv(shard, rec_cfg)
                e.set_reset_mode(args.reset_mode)
                if args.reset_mode == "curriculum":
                    e.set_level_file(str(args.out / "curriculum_level.txt"))
                e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
                e = Monitor(e)
                return e
            return _thunk
    else:
        def _make_env(rank: int = 0):
            def _thunk():
                e = LastInchInsertEnv(env_cfg, port_type=args.port_type)
                e.set_reset_mode(args.reset_mode)
                if args.reset_mode == "curriculum":
                    e.set_level_file(str(args.out / "curriculum_level.txt"))
                e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
                e = Monitor(e)
                return e
            return _thunk

    if n_envs == 1:
        # single-env: no forking overhead, easier debugging
        env = DummyVecEnv([_make_env(0)])
    else:
        # parallel envs: one subprocess per env (each has its own MjModel)
        env = SubprocVecEnv([_make_env(i) for i in range(n_envs)],
                            start_method="fork" if os.name == "posix" else "spawn")
    # SB3 expects image as (B, C, H, W) for the CNN — transpose if needed
    env = VecTransposeImage(env)
    print(f"[train] env obs_space={env.observation_space.spaces.keys()} "
          f"num_envs={n_envs} reset_mode={args.reset_mode}", flush=True)

    # ------------------------------------------------------------------ #
    # custom features extractor (image CNN + state MLP)
    # ------------------------------------------------------------------ #

    class CombinedImageState(BaseFeaturesExtractor):
        """Image CNN + small MLP for the rest, concatenated then linear."""

        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            n_channels = observation_space["image"].shape[2]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # compute CNN output dim with a dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, n_channels,
                                    observation_space["image"].shape[0],
                                    observation_space["image"].shape[1])
                cnn_out = self.cnn(dummy).shape[1]

            state_keys = ("force", "tcp_pose", "port_pose", "tcp_pose_err", "last_action")
            state_dim = sum(int(np.prod(observation_space[k].shape)) for k in state_keys)

            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
            )
            self.fuse = nn.Sequential(
                nn.Linear(cnn_out + 64, features_dim), nn.ReLU(),
            )

        def forward(self, observations):
            # image: SB3 puts image as (B, H, W, C); convert to (B, C, H, W)
            img = observations["image"].float() / 255.0
            img = img.permute(0, 3, 1, 2)
            z_img = self.cnn(img)
            state_vecs = [
                observations["force"].float(),
                observations["tcp_pose"].float(),
                observations["port_pose"].float(),
                observations["tcp_pose_err"].float(),
                observations["last_action"].float(),
            ]
            z_state = self.state_mlp(torch.cat(state_vecs, dim=1))
            return self.fuse(torch.cat([z_img, z_state], dim=1))

    policy_kwargs = dict(
        features_extractor_class=CombinedImageState,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
    )

    # ------------------------------------------------------------------ #
    # per-step component-reward logging callback
    # ------------------------------------------------------------------ #

    class ComponentRewardLogger(BaseCallback):
        """Logs the per-component reward (image, force, xy, action, lateral, done).

        Reads from `info["breakdown"]` populated by LastInchInsertEnv.
        """
        def __init__(self, log_every: int = 100):
            super().__init__()
            self.log_every = log_every
            self._i = 0

        def _on_step(self) -> bool:
            self._i += 1
            if self._i % self.log_every == 0:
                infos = self.model.ep_info_buffer
                if infos and len(infos) > 0:
                    last = infos[-1]
                    self.logger.record("rollout/term_status", str(last.get("term_status", "")))
                    self.logger.record("rollout/image_l1_norm", float(last.get("image_l1_norm", 0.0)))
                    self.logger.record("rollout/f_z", float(last.get("f_z", 0.0)))
            return True

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #

    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.warmup_steps,
        seed=args.seed,
        verbose=1,
        # TensorBoard is optional — only enable if `tensorboard` is importable.
        tensorboard_log=str(args.out / "tb") if _has_tensorboard() else None,
    )

    # resume from existing checkpoint if --resume and model.zip exists
    sb3_zip = args.out / "model.zip"
    if args.resume and sb3_zip.exists():
        print(f"[train] resuming from {sb3_zip}", flush=True)
        model = SAC.load(str(sb3_zip), env=env, device=model.device)
        # reset learning rate to current args (in case we changed it)
        model.lr_schedule = model._get_schedule(model.learning_rate)

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #

    from RL.logging_utils import MetricsLogger, VideoRecorder, CurriculumScheduler, ProgressPrinter, WandbLogger

    callbacks = [ComponentRewardLogger(),
                 MetricsLogger(args.out / "metrics.jsonl"),
                 ProgressPrinter(log_every_steps=args.log_every)]

    # ------------------------------------------------------------------ #
    # W&B callback (optional)
    # ------------------------------------------------------------------ #
    # Decide whether wandb should be enabled. Default policy:
    #   --no-wandb / WANDB_MODE=disabled       -> off
    #   --wandb  AND  WANDB_API_KEY set        -> on  (online)
    #   --wandb  AND  WANDB_MODE=offline       -> on  (offline)
    #   WANDB_API_KEY set, no --wandb flag     -> on  (online)
    #   nothing set                             -> off (silent)
    wandb_explicit = args.wandb  # True / False / None
    wandb_env_off = os.environ.get("WANDB_MODE", "").strip().lower() in (
        "disabled", "false", "0")
    wandb_api_present = bool(os.environ.get("WANDB_API_KEY", "").strip())
    if wandb_env_off or wandb_explicit is False:
        wandb_enabled = False
    elif wandb_explicit is True or wandb_api_present:
        wandb_enabled = True
    else:
        wandb_enabled = False

    # config payload exposed in the W&B run
    wandb_config = dict(cfg_dict)
    # flatten dataclasses (reward + termination) into plain dicts
    try:
        wandb_config["reward"] = dataclasses.asdict(env_cfg.reward)
        wandb_config["termination"] = dataclasses.asdict(env_cfg.term)
    except Exception:
        pass
    wandb_config["env/port_type"] = args.port_type
    wandb_config["env/pos_scale"] = list(env_cfg.pos_scale)
    wandb_config["env/rot_scale"] = list(env_cfg.rot_scale)
    wandb_config["env/dt"] = env_cfg.dt
    wandb_config["env/start_xy_max_m"] = env_cfg.start_xy_max_m
    wandb_config["env/start_z_min_m"] = env_cfg.start_z_min_m
    wandb_config["env/start_z_max_m"] = env_cfg.start_z_max_m
    wandb_config["env/success_xy_max_m"] = env_cfg.success_xy_max_m
    wandb_config["env/success_z_max_m"] = env_cfg.success_z_max_m
    wandb_config["env/success_force_z_n"] = env_cfg.success_force_z_n
    wandb_config["algorithm"] = "SAC"
    wandb_config["features_extractor"] = "CombinedImageState"
    wandb_config["net_arch_pi"] = [256, 256]
    wandb_config["net_arch_qf"] = [256, 256]

    run_name = args.wandb_run_name or f"{args.port_type}-sac-{args.steps}-seed{args.seed}"
    wandb_cb = WandbLogger(
        run_name=run_name if wandb_enabled else None,
        config=wandb_config,
        log_every_steps=args.wandb_log_every,
        success_window=args.wandb_success_window,
        out_dir=args.out,
        enabled=wandb_enabled,
    )
    callbacks.append(wandb_cb)

    if args.reset_mode == "curriculum":
        callbacks.append(CurriculumScheduler(
            env=None,  # communicate via level_file only (works with SubprocVecEnv)
            eval_window=args.curriculum_eval_window,
            advance_threshold=args.curriculum_advance_threshold,
            retreat_threshold=args.curriculum_retreat_threshold,
            step_size=args.curriculum_step,
            level_path=args.out / "curriculum_level.txt",
        ))
    if not args.no_video and args.video_every > 0:
        callbacks.append(VideoRecorder(env, args.out / "videos",
                                       every_n_episodes=args.video_every,
                                       max_steps=env_cfg.term.max_steps))
    callback = callbacks  # SB3 accepts a list

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
    finally:
        # ensure wandb.finish() runs on early exit / KeyboardInterrupt /
        # exceptions / normal completion
        wandb_cb.finish()
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s ({args.steps/elapsed:.0f} steps/s)", flush=True)

    # save SB3 checkpoint
    sb3_path = args.out / "model.zip"
    model.save(str(sb3_path))
    print(f"[train] saved SB3 checkpoint to {sb3_path}", flush=True)

    # export TorchScript for deployment (matches the AIC _load_final_insert_policy path)
    if not args.no_torchscript:
        try:
            ts_path = args.out / "policy.pt"
            policy = model.policy
            policy.eval()
            # build a dummy input matching the observation space
            obs_space = env.observation_space
            dummy = {}
            for k, sp in obs_space.spaces.items():
                shape = sp.shape
                if "image" in k:
                    dummy[k] = torch.zeros(1, *shape, dtype=torch.uint8)
                else:
                    dummy[k] = torch.zeros(1, *shape, dtype=torch.float32)
            with torch.no_grad():
                traced = torch.jit.trace(policy, dummy, strict=False)
            traced.save(str(ts_path))
            print(f"[train] exported TorchScript policy to {ts_path}", flush=True)
        except Exception as exc:
            print(f"[train] WARN: TorchScript export failed: {exc}", flush=True)

    # write a small meta file
    meta_path = args.out / "train_meta.txt"
    meta_path.write_text(
        f"port_type={args.port_type}\n"
        f"steps={args.steps}\n"
        f"seed={args.seed}\n"
        f"lr={args.lr}\n"
        f"batch_size={args.batch_size}\n"
        f"buffer_size={args.buffer_size}\n"
        f"warmup_steps={args.warmup_steps}\n"
        f"elapsed_s={elapsed:.1f}\n"
        f"sb3_path={sb3_path}\n"
        f"cache_key={cache_key}\n"
        f"policy_class=aic_example_policies.ros.LastInchInsert\n"
    )
    print(f"[train] wrote {meta_path}", flush=True)

    # optional dashboard plot
    if args.plot:
        try:
            from RL.logging_utils import plot_dashboard
            png = plot_dashboard(args.out / "metrics.jsonl")
            if png is not None:
                print(f"[train] dashboard saved to {png}", flush=True)
            else:
                print("[train] WARN: dashboard skipped (no records or matplotlib missing)", flush=True)
        except Exception as exc:
            print(f"[train] WARN: dashboard failed: {exc}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main"]