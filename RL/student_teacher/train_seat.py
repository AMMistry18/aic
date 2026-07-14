"""Train the force-reactive seat policy with plain asymmetric SAC.

The two real curriculum stages are ``nominal`` (``make_seat_env("tight")``)
and ``full`` (``make_seat_env("full")``).  The reset settle already funnels
lateral jitter to roughly 0.14--0.64 mm, so lateral bands are not a useful
curriculum lever here; hidden grasp noise is.  Train nominal first, then pass
its ``model.zip`` to the full run with ``--init-from``.

There is deliberately no teacher, prior replay, RLPD mixing, or BC auxiliary in
this trainer.  The actor sees only ``obs["actor"]``; privileged state is used by
the critic during training and is absent from the exported TorchScript actor.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


STAGE_TO_ENV = {"nominal": "tight", "full": "full"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_TO_ENV), required=True)
    parser.add_argument("--init-from", type=Path,
                        help="SB3 checkpoint used to warm-start a fresh run")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument(
        "--gradient-steps", type=int, default=-1,
        help="SAC updates per vector rollout; -1 matches collected transitions",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--video-freq", type=int,
                        help="video cadence; defaults to checkpoint cadence")
    parser.add_argument("--eval-freq", type=int,
                        help="evaluation cadence; defaults to checkpoint cadence")
    parser.add_argument("--wandb-log-freq", type=int, default=1_000,
                        help="frequent W&B scalar cadence in environment steps")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--tensorboard-log", type=Path,
                        help="TensorBoard root (keep under the work checkout)")
    parser.add_argument("--wandb-run-name",
                        help="unique W&B name; launcher includes commit/job id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.resume and args.init_from is not None:
        parser.error("--resume and --init-from are mutually exclusive")
    if args.init_from is not None and not args.init_from.is_file():
        parser.error(f"--init-from checkpoint does not exist: {args.init_from}")
    if args.eval_freq is not None and args.eval_freq <= 0:
        parser.error("--eval-freq must be positive")
    if args.wandb_log_freq <= 0:
        parser.error("--wandb-log-freq must be positive")
    return args


def _finite_stats(values: list[float], *, percentile: float | None = None) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not finite.size:
        return float("nan")
    if percentile is not None:
        return float(np.percentile(finite, percentile))
    return float(np.mean(finite))


def evaluate_seat(model, *, stage: str, episodes: int, seed: int):
    """Run deterministic seat eval and return W&B scalars plus trace rows."""
    from RL.student_teacher.seat_env import make_seat_env

    env_stage = STAGE_TO_ENV[stage]
    env = make_seat_env(env_stage, seed=seed, domain_randomization=True)
    traces: list[dict[str, Any]] = []
    final_depths, max_forces, final_lateral, steps_to_seat = [], [], [], []
    status_counts: dict[str, int] = {}
    jam_count = 0
    try:
        for episode in range(int(episodes)):
            obs, reset_info = env.reset(seed=seed + episode)
            start_depth = float(reset_info.get(
                "insertion_depth_m", env.unwrapped._insertion_depth_m()))
            max_depth = start_depth
            max_force = 0.0
            engaged_ridge = False
            done = False
            steps = 0
            info: dict[str, Any] = dict(reset_info)
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                steps += 1
                depth = float(info.get("insertion_depth_m", float("nan")))
                force = float(info.get("contact_force_norm", float("nan")))
                if np.isfinite(depth):
                    max_depth = max(max_depth, depth)
                if np.isfinite(force):
                    max_force = max(max_force, force)
                randomization = info.get("domain_randomization", {}) or {}
                ridge_depth = float(randomization.get(
                    "contact_ridge_depth_m", float("nan")))
                if (np.isfinite(ridge_depth) and max_depth >= ridge_depth
                        and int(info.get("plug_port_contacts", 0)) > 0):
                    engaged_ridge = True
                done = bool(terminated or truncated)

            status = str(info.get("term_status") or "timeout")
            status_counts[status] = status_counts.get(status, 0) + 1
            final_depth = float(info.get("insertion_depth_m", float("nan")))
            lateral_end = float(info.get("f_lateral", float("nan")))
            final_depths.append(1e3 * final_depth)
            max_forces.append(max_force)
            final_lateral.append(lateral_end)
            if status == "success":
                steps_to_seat.append(steps)
            if status == "timeout" and engaged_ridge:
                jam_count += 1
            traces.append({
                "episode": episode,
                "start_depth_mm": 1e3 * start_depth,
                "max_depth_mm": 1e3 * max_depth,
                "status": status,
                "max_force_n": max_force,
            })
    finally:
        env.close()

    n = max(int(episodes), 1)
    metrics = {
        "eval/seat_success_rate": status_counts.get("success", 0) / n,
        "eval/jam_rate": jam_count / n,
        "eval/final_depth_mm_mean": _finite_stats(final_depths),
        "eval/final_depth_mm_p50": _finite_stats(final_depths, percentile=50),
        "eval/final_depth_mm_max": float(np.nanmax(final_depths)),
        "eval/max_force_n_mean": _finite_stats(max_forces),
        "eval/max_force_n": _finite_stats(max_forces),
        "eval/max_force_n_p95": _finite_stats(max_forces, percentile=95),
        "eval/max_force_n_max": float(np.nanmax(max_forces)),
        "eval/f_lateral_at_end_n": _finite_stats(final_lateral),
        "eval/steps_to_seat": _finite_stats(steps_to_seat),
        "eval/force_abort_rate": status_counts.get("force_abort", 0) / n,
        "eval/bad_collision_rate": status_counts.get("bad_collision", 0) / n,
    }
    return metrics, traces, status_counts


def _init_wandb(args, config):
    """Initialize W&B online by default, falling back safely to offline."""
    try:
        import wandb
    except Exception as exc:
        print(f"[wandb] disabled: import failed: {exc}", flush=True)
        return None

    requested_mode = os.environ.get("WANDB_MODE", "online")
    init_kwargs = dict(
        project="aic-seat-rl",
        name=(args.wandb_run_name
              or f"seat_{args.stage}_seed{args.seed}"),
        dir=str(args.out),
        config=config,
        sync_tensorboard=True,
    )
    try:
        run = wandb.init(mode=requested_mode, **init_kwargs)
    except Exception as exc:
        if requested_mode == "offline":
            print(f"[wandb] disabled: offline init failed: {exc}", flush=True)
            return None
        print(f"[wandb] online init failed ({exc}); falling back to offline", flush=True)
        try:
            wandb.finish(exit_code=1, quiet=True)
        except Exception:
            pass
        try:
            run = wandb.init(mode="offline", **init_kwargs)
        except Exception as offline_exc:
            print(f"[wandb] disabled: offline fallback failed: {offline_exc}", flush=True)
            return None
    if run is not None:
        actual_offline = bool(getattr(run, "offline", False))
        if requested_mode == "offline" or actual_offline:
            sync_dir = Path(run.dir).parent
            print(
                f"[wandb] mode=offline; later run: wandb sync {sync_dir}",
                flush=True,
            )
        else:
            print(
                f"[wandb] mode=online url={getattr(run, 'url', 'unavailable')}",
                flush=True,
            )
        run.define_metric("global_step")
        for metric in (
                "rollout/ep_len_mean", "rollout/ep_rew_mean", "time/fps",
                "train/actor_loss", "train/critic_loss", "train/ent_coef",
                "train/ent_coef_loss", "eval/seat_success_rate",
                "eval/bad_collision_rate", "eval/max_force_n_mean",
                "eval/max_force_n", "eval/max_force_n_p95",
                "eval/max_force_n_max", "train/update_data_ratio"):
            run.define_metric(metric, step_metric="global_step")
        for metric_glob in ("reset/*", "termination/*", "reward/*"):
            run.define_metric(metric_glob, step_metric="global_step")
    return run


def export_actor(model, out: Path, sample_actor: np.ndarray) -> dict[str, Any]:
    """Export deterministic actor mean only: actor history (8,34) -> action (6,)."""
    import torch
    from torch import nn

    class ActorOnly(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, actor_observation):
            batched = actor_observation.unsqueeze(0)
            action = self.actor({"actor": batched}, deterministic=True)
            return action.squeeze(0)

    actor = copy.deepcopy(model.actor).to("cpu").eval()
    wrapper = ActorOnly(actor).eval()
    example = torch.as_tensor(sample_actor, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=True)
        traced = torch.jit.freeze(traced.eval())
        eager_action = wrapper(example)
        traced_action = traced(example)
    max_error = float(torch.max(torch.abs(eager_action - traced_action)).item())
    if max_error > 1e-6 or not bool(torch.isfinite(traced_action).all()):
        raise RuntimeError(
            f"actor export parity failed: max_abs_error={max_error:g}, "
            f"finite={bool(torch.isfinite(traced_action).all())}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out))
    metadata = {
        "checkpoint": str((out.parent / "model.zip").resolve()),
        "input": "actor_history",
        "input_shape": list(sample_actor.shape),
        "output_shape": list(traced_action.shape),
        "output_convention": "normalized seat policy action; environment applies deploy gain",
        "privileged_input": False,
        "torchscript_max_abs_error": max_error,
        "sample_action": traced_action.cpu().numpy().tolist(),
        "sample_action_finite": True,
    }
    out.with_suffix(out.suffix + ".contract.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(f"[export] actor-only TorchScript -> {out}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import wandb
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from RL.logging_utils import WandbVideoRecorder
    from RL.student_teacher.seat_env import make_seat_env
    from RL.student_teacher.student_v3_sac import AsymmetricSACPolicy

    config = {**vars(args), "out": str(args.out),
              "init_from": str(args.init_from) if args.init_from else None,
              "tensorboard_log": (
                  str(args.tensorboard_log) if args.tensorboard_log else None),
              "env_stage": STAGE_TO_ENV[args.stage], "gamma": 0.98,
              "trainer": "stable_baselines3.SAC"}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    wandb_run = _init_wandb(args, config)

    def make_train_env(rank):
        def thunk():
            return Monitor(make_seat_env(
                STAGE_TO_ENV[args.stage], seed=args.seed * 100_000 + rank,
                domain_randomization=True))
        return thunk

    if args.num_envs == 1:
        env = DummyVecEnv([make_train_env(0)])
    else:
        env = SubprocVecEnv(
            [make_train_env(rank) for rank in range(args.num_envs)],
            start_method="fork" if os.name == "posix" else "spawn")

    class CheckpointCallback(BaseCallback):
        def __init__(self, every_steps):
            super().__init__()
            self.every_steps = max(1, int(every_steps))
            self.next_step = self.every_steps

        def _on_step(self):
            if int(self.num_timesteps) >= self.next_step:
                self.model.save(args.out / "model")
                self.model.save_replay_buffer(args.out / "replay_buffer.pkl")
                (args.out / "progress.json").write_text(json.dumps({
                    "timesteps": int(self.num_timesteps),
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, indent=2) + "\n")
                self.next_step = int(self.num_timesteps) + self.every_steps
            return True

    class SeatEvaluator(BaseCallback):
        def __init__(self, every_steps):
            super().__init__()
            self.every_steps = max(1, int(every_steps))
            self.next_step = self.every_steps
            self.last_step = -1
            self.latest = None

        def _run_eval(self):
            metrics, traces, outcomes = evaluate_seat(
                self.model, stage=args.stage, episodes=args.eval_episodes,
                seed=90_000 + args.seed * 1_000 + int(self.num_timesteps))
            payload = {
                "timesteps": int(self.num_timesteps), "stage": args.stage,
                "episodes": int(args.eval_episodes), "metrics": metrics,
                "outcomes": outcomes, "traces": traces,
            }
            self.latest = payload
            self.last_step = int(self.num_timesteps)
            (args.out / "evaluation.json").write_text(
                json.dumps(payload, indent=2) + "\n")
            with (args.out / "evaluation_history.jsonl").open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
            if wandb_run is not None:
                columns = ["episode", "start_depth_mm", "max_depth_mm",
                           "status", "max_force_n"]
                table = wandb.Table(
                    columns=columns,
                    data=[[row[column] for column in columns] for row in traces])
                wandb_run.log({
                    "global_step": int(self.num_timesteps),
                    **metrics,
                    "eval/seat_traces": table,
                })
            print(json.dumps(payload, indent=2), flush=True)

        def _on_step(self):
            if int(self.num_timesteps) >= self.next_step:
                self._run_eval()
                self.next_step = int(self.num_timesteps) + self.every_steps
            return True

        def _on_training_end(self):
            if self.last_step != int(self.num_timesteps):
                self._run_eval()

    def make_video_env():
        seat_env = make_seat_env(
            STAGE_TO_ENV[args.stage], seed=190_000 + args.seed,
            domain_randomization=True)
        # Gymnasium wrappers do not expose arbitrary attributes through
        # ``hasattr``.  SB3 must step the wrapped seat env, while the existing
        # recorder needs the underlying SceneInsertEnv for render_camera().
        render_env = seat_env.unwrapped
        vec_env = DummyVecEnv([lambda: Monitor(seat_env)])
        vec_env.seed(190_000 + args.seed)
        return vec_env, render_env

    evaluator = SeatEvaluator(args.eval_freq or args.checkpoint_freq)

    class LiveWandbMetrics(BaseCallback):
        """Stream train scalars between expensive checkpoints/evaluations."""

        def __init__(self, every_steps):
            super().__init__()
            self.every_steps = max(1, int(every_steps))
            self.next_step = self.every_steps
            self.reset_count = 0
            self.reset_attempts = 0.0
            self.reset_fallbacks = 0
            self.reset_pool_sizes: list[float] = []
            self.term_counts = {
                status: 0 for status in (
                    "success", "bad_collision", "force_abort", "off_limit",
                    "timeout",
                )
            }
            self.reward_sums: dict[str, float] = {}
            self.reward_samples = 0

        def _accumulate_diagnostics(self):
            infos = list(self.locals.get("infos") or ())
            dones = np.asarray(
                self.locals.get("dones", np.zeros(len(infos), dtype=bool)),
                dtype=bool,
            ).reshape(-1)
            for index, info in enumerate(infos):
                if bool(info.get("seat_reset_first_step", False)):
                    self.reset_count += 1
                    self.reset_attempts += float(
                        info.get("seat_reset_attempts", 1))
                    self.reset_fallbacks += int(bool(
                        info.get("seat_reset_used_fallback", False)))
                    self.reset_pool_sizes.append(float(
                        info.get("seat_reset_pool_size", float("nan"))))
                terms = info.get("seat_reward_terms") or {}
                for name, value in terms.items():
                    value = float(value)
                    if np.isfinite(value):
                        self.reward_sums[name] = (
                            self.reward_sums.get(name, 0.0) + value)
                if terms:
                    self.reward_samples += 1
                if index < len(dones) and dones[index]:
                    status = str(info.get("term_status") or "timeout")
                    if status not in self.term_counts:
                        self.term_counts[status] = 0
                    self.term_counts[status] += 1

        def _reset_interval(self):
            self.reset_count = 0
            self.reset_attempts = 0.0
            self.reset_fallbacks = 0
            self.reset_pool_sizes.clear()
            self.term_counts = {name: 0 for name in self.term_counts}
            self.reward_sums.clear()
            self.reward_samples = 0

        def _on_step(self):
            self._accumulate_diagnostics()
            if wandb_run is None or int(self.num_timesteps) < self.next_step:
                return True
            payload = {"global_step": int(self.num_timesteps)}
            logger_values = self.model.logger.name_to_value
            for metric in (
                    "rollout/ep_len_mean", "rollout/ep_rew_mean", "time/fps",
                    "train/actor_loss", "train/critic_loss", "train/ent_coef",
                    "train/ent_coef_loss"):
                value = logger_values.get(metric)
                if value is not None and np.isfinite(float(value)):
                    payload[metric] = float(value)
            learned_transitions = max(
                int(self.num_timesteps) - int(args.learning_starts), 0)
            if learned_transitions > 0:
                payload["train/update_data_ratio"] = (
                    float(getattr(self.model, "_n_updates", 0))
                    / float(learned_transitions)
                )
            if self.reset_count:
                payload["reset/attempts_mean"] = (
                    self.reset_attempts / self.reset_count)
                payload["reset/fallback_rate"] = (
                    self.reset_fallbacks / self.reset_count)
                finite_pool_sizes = [
                    value for value in self.reset_pool_sizes
                    if np.isfinite(value)
                ]
                if finite_pool_sizes:
                    payload["reset/pool_size_mean"] = float(np.mean(
                        finite_pool_sizes))
            total_terms = sum(self.term_counts.values())
            if total_terms:
                for status, count in self.term_counts.items():
                    payload[f"termination/{status}_rate"] = (
                        count / total_terms)
            if self.reward_samples:
                for name, total in self.reward_sums.items():
                    payload[f"reward/{name}_mean"] = (
                        total / self.reward_samples)
            wandb_run.log(payload)
            self._reset_interval()
            self.next_step = int(self.num_timesteps) + self.every_steps
            return True

    live_wandb = LiveWandbMetrics(args.wandb_log_freq)
    video = WandbVideoRecorder(
        make_eval_env=make_video_env,
        every_steps=args.video_freq or args.checkpoint_freq,
        episodes_per_video=2,
        deterministic=True,
        key="eval/rollout_video",
        render_camera="center_camera",
        max_steps=400,
        record_on_training_end=True,
        record_at_start=False,
        explicit_wandb_step=False,
        enabled=args.video,
    )
    print(f"[video] WandbVideoRecorder attached enabled={args.video} "
          f"every_steps={args.video_freq or args.checkpoint_freq} "
          "camera=center_camera", flush=True)

    common = dict(
        policy=AsymmetricSACPolicy,
        env=env,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.98,
        train_freq=(1, "step"),
        gradient_steps=args.gradient_steps,
        ent_coef="auto_0.01",
        target_entropy=-3.0,
        policy_kwargs=dict(net_arch=[256, 256], share_features_extractor=False),
        seed=args.seed,
        device="auto",
        verbose=1,
        tensorboard_log=str(
            args.tensorboard_log or (args.out / "tensorboard")),
    )
    model_path = args.out / "model.zip"
    replay_path = args.out / "replay_buffer.pkl"
    if args.resume and model_path.exists():
        model = SAC.load(model_path, env=env, device="auto")
        if replay_path.exists():
            model.load_replay_buffer(replay_path)
        reset_num_timesteps = False
        remaining = max(0, int(args.steps) - int(model.num_timesteps))
        print(f"[train] resuming {model_path} at {model.num_timesteps} steps", flush=True)
    else:
        model = SAC(**common)
        if args.init_from is not None:
            model.set_parameters(str(args.init_from), exact_match=True)
            print(f"[train] warm-started parameters from {args.init_from}", flush=True)
        reset_num_timesteps = True
        remaining = int(args.steps)

    if remaining:
        model.learn(
            total_timesteps=remaining,
            callback=CallbackList([
                CheckpointCallback(args.checkpoint_freq), evaluator, video,
                live_wandb,
            ]),
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
    elif evaluator.latest is None:
        evaluator.model = model
        evaluator.num_timesteps = int(model.num_timesteps)
        evaluator._run_eval()

    model.save(args.out / "model")
    model.save_replay_buffer(replay_path)
    sample_env = make_seat_env(
        STAGE_TO_ENV[args.stage], seed=290_000 + args.seed,
        domain_randomization=True)
    try:
        sample_obs, _ = sample_env.reset(seed=290_000 + args.seed)
        export_actor(model, args.out / "seat_actor.ts", sample_obs["actor"])
    finally:
        sample_env.close()
    (args.out / "RUN_COMPLETE").touch()
    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
