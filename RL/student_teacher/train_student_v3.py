"""Train the contact-stage Student-v3 asymmetric residual SAC pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import gymnasium as gym
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prior-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("contact", "small", "medium", "full", "robust"),
                   default="contact")
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _evaluate(model, *, stage: str, episodes: int, seed: int):
    from RL.student_teacher.student_v3_env import make_student_v3_env
    env = make_student_v3_env(
        seed=seed, stage=stage, domain_randomization=True)
    outcomes: dict[str, int] = {}
    peak_force, peak_lateral, stalls, returns = [], [], [], []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            total, fmax, lmax, stall_n = 0.0, 0.0, 0.0, 0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                total += float(reward)
                fmax = max(fmax, float(info.get("contact_force_norm", 0.0)))
                lmax = max(lmax, float(info.get("lateral_error_m", 0.0)))
                stall_n += int(info.get("v3_stalled", False))
                done = bool(term or trunc)
            status = str(info.get("term_status") or "timeout")
            outcomes[status] = outcomes.get(status, 0) + 1
            peak_force.append(fmax)
            peak_lateral.append(lmax)
            stalls.append(stall_n)
            returns.append(total)
    finally:
        env.close()
    success = outcomes.get("success", 0)
    collisions = outcomes.get("bad_collision", 0) + outcomes.get("off_limit", 0)
    return {
        "episodes": episodes,
        "success_rate": success / max(episodes, 1),
        "collision_rate": collisions / max(episodes, 1),
        "force_abort_rate": outcomes.get("force_abort", 0) / max(episodes, 1),
        "outcomes": outcomes,
        "peak_force_mean_n": float(np.mean(peak_force)),
        "peak_force_p95_n": float(np.quantile(peak_force, 0.95)),
        "peak_lateral_mean_mm": 1e3 * float(np.mean(peak_lateral)),
        "stall_steps_mean": float(np.mean(stalls)),
        "return_mean": float(np.mean(returns)),
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from RL.student_teacher.student_v3_env import make_student_v3_env
    from RL.student_teacher.student_v3_sac import (
        AsymmetricSACPolicy, PriorReplayDataset, RLPDSAC)

    prior = PriorReplayDataset(args.prior_manifest)
    config = {**vars(args), "prior_manifest": str(args.prior_manifest),
              "out": str(args.out), "prior_rows": prior.size}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project="aic-student-v3",
            name=f"student_v3_{args.stage}_seed{args.seed}",
            dir=str(args.out),
            config=config,
            mode=os.environ.get("WANDB_MODE", "offline"),
            sync_tensorboard=True,
        )
    except Exception as exc:
        print(f"[wandb] disabled: {exc}", flush=True)

    def make_env(rank):
        def thunk():
            return Monitor(make_student_v3_env(
                seed=args.seed * 100_000 + rank,
                stage=args.stage,
                domain_randomization=True))
        return thunk

    if args.num_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv(
            [make_env(rank) for rank in range(args.num_envs)],
            start_method="fork" if os.name == "posix" else "spawn")

    class CheckpointCallback(BaseCallback):
        def __init__(self, freq):
            super().__init__()
            self.freq = max(1, int(freq) // max(1, args.num_envs))

        def _on_step(self):
            if self.n_calls % self.freq == 0:
                self.model.save(args.out / "model")
                self.model.save_replay_buffer(args.out / "replay_buffer.pkl")
                (args.out / "progress.json").write_text(json.dumps({
                    "timesteps": self.num_timesteps,
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, indent=2) + "\n")
            return True

    model_path = args.out / "model.zip"
    replay_path = args.out / "replay_buffer.pkl"
    common = dict(
        policy=AsymmetricSACPolicy,
        env=env,
        prior_manifest=args.prior_manifest,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        learning_starts=0,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto_0.01",
        target_entropy=-3.0,
        policy_kwargs=dict(net_arch=[256, 256], share_features_extractor=False),
        seed=args.seed,
        device="auto",
        verbose=1,
        tensorboard_log=str(args.out / "tensorboard"),
    )
    if args.resume and model_path.exists():
        model = RLPDSAC.load(
            model_path, env=env, prior_manifest=args.prior_manifest, device="auto")
        if replay_path.exists():
            model.load_replay_buffer(replay_path)
    else:
        model = RLPDSAC(**common)

    start = int(model.num_timesteps)
    remaining = max(0, int(args.steps) - start)
    if remaining:
        model.learn(
            total_timesteps=remaining,
            callback=CheckpointCallback(args.checkpoint_freq),
            reset_num_timesteps=False,
            progress_bar=False,
        )
    model.save(args.out / "model")
    model.save_replay_buffer(replay_path)
    evaluation = _evaluate(
        model, stage=args.stage, episodes=args.eval_episodes,
        seed=90_000 + args.seed * 1_000)
    evaluation.update({
        "seed": args.seed,
        "stage": args.stage,
        "timesteps": int(model.num_timesteps),
        "selection_order": [
            "min_collision_rate", "max_success_rate", "min_force_abort_rate",
            "min_peak_force_p95_n", "min_stall_steps_mean"],
    })
    (args.out / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n")
    (args.out / "RUN_COMPLETE").touch()
    if wandb_run is not None:
        wandb_run.log({f"eval/{key}": value for key, value in evaluation.items()
                       if isinstance(value, (int, float))})
        wandb_run.finish()
    print(json.dumps(evaluation, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
