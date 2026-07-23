"""Train the align-first alignment RL (single-RL, align-only) with asymmetric SAC.

The align env (`RL.student_teacher.align_env`) rewards squaring the plug over the
perceived port -- lateral x/y + rotation roll/pitch/yaw -- while holding z at a
standoff. It does NOT insert (no depth reward); success == alignment tolerance.
Unlike Student-v3, this trains FROM SCRATCH: there is no guided base to distill a
prior from, so we use plain SAC (not RLPD) with a real warmup. The asymmetric
policy/extractors are reused from `student_v3_sac` -- they read dims dynamically
from the observation space, so the align env's 34-dim actor frame needs no change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("align", "small", "medium", "full", "robust"),
                   default="align")
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _evaluate(model, *, stage: str, episodes: int, seed: int):
    """Align-first metrics: success == info['align_term_status'] == 'aligned'."""
    from RL.student_teacher.align_env import make_align_env
    env = make_align_env(stage=stage, seed=seed, domain_randomization=True)
    outcomes: dict[str, int] = {}
    final_lat_mm, final_rot_deg, returns = [], [], []
    steps_to_align = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            total, steps = 0.0, 0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                total += float(reward)
                steps += 1
                done = bool(term or trunc)
            status = str(info.get("align_term_status") or "timeout")
            outcomes[status] = outcomes.get(status, 0) + 1
            final_lat_mm.append(1e3 * float(info.get("align_lat_err_m", np.nan)))
            final_rot_deg.append(np.degrees(float(info.get("align_rot_err_rad", np.nan))))
            returns.append(total)
            if status == "aligned":
                steps_to_align.append(steps)
    finally:
        env.close()
    aligned = outcomes.get("aligned", 0)
    fails = (outcomes.get("bad_collision", 0) + outcomes.get("off_limit", 0)
             + outcomes.get("force_abort", 0))
    return {
        "episodes": episodes,
        "align_success_rate": aligned / max(episodes, 1),
        "fail_rate": fails / max(episodes, 1),
        "timeout_rate": outcomes.get("timeout", 0) / max(episodes, 1),
        "outcomes": outcomes,
        "final_lat_err_p50_mm": float(np.nanmedian(final_lat_mm)) if final_lat_mm else float("nan"),
        "final_rot_err_p50_deg": float(np.nanmedian(final_rot_deg)) if final_rot_deg else float("nan"),
        "steps_to_align_mean": float(np.mean(steps_to_align)) if steps_to_align else float("nan"),
        "return_mean": float(np.mean(returns)),
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from RL.student_teacher.align_env import make_align_env
    from RL.student_teacher.student_v3_sac import AsymmetricSACPolicy

    config = {**vars(args), "out": str(args.out)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project="aic-align-rl",
            name=f"align_{args.stage}_seed{args.seed}",
            dir=str(args.out),
            config=config,
            mode=os.environ.get("WANDB_MODE", "offline"),
            sync_tensorboard=True,
        )
    except Exception as exc:
        print(f"[wandb] disabled: {exc}", flush=True)

    def make_env(rank):
        def thunk():
            return Monitor(make_align_env(
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
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
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
        model = SAC.load(model_path, env=env, device="auto")
        if replay_path.exists():
            model.load_replay_buffer(replay_path)
    else:
        model = SAC(**common)

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
            "max_align_success_rate", "min_fail_rate",
            "min_final_lat_err_p50_mm", "min_final_rot_err_p50_deg",
            "min_steps_to_align_mean"],
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
