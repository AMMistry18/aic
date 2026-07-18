"""Train the force-reactive seat policy with plain asymmetric SAC.

The two stages are ``bootstrap`` (synchronized sampling-boundary curriculum)
and ``deployment`` (the fixed 70/15/10/5 reset mixture).  Bootstrap keeps the
hard 3 mm boundary present from the first transition and removes easy deep
starts as the shared 100-episode success window improves.  Deployment then
warm-starts from the best bootstrap checkpoint.

There is deliberately no teacher, prior replay, RLPD mixing, or BC auxiliary in
this trainer.  The actor sees only ``obs["actor"]``; privileged state is used by
the critic during training and is absent from the exported TorchScript actor.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


STAGE_TO_ENV = {
    "bootstrap": "bootstrap",
    "deployment": "deployment",
    # Compatibility aliases for old launch commands; new scripts use the names
    # above and record the canonical environment stage in config.json.
    "nominal": "bootstrap",
    "full": "deployment",
}
CANONICAL_STAGE = {
    "bootstrap": "bootstrap", "nominal": "bootstrap",
    "deployment": "deployment", "full": "deployment",
}
EVAL_CLASS_WEIGHTS = {
    "live_shallow": 0.70,
    "centered_shallow": 0.15,
    "mid_tail": 0.10,
    "mastered_deep": 0.05,
}
EVAL_CONTACT_VARIANTS = 3


def fixed_eval_class_sequence(episodes: int, seed: int) -> list[str]:
    """Return a deterministic exact-weight evaluation suite."""
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    names = tuple(EVAL_CLASS_WEIGHTS)
    exact = np.asarray([EVAL_CLASS_WEIGHTS[name] * episodes for name in names])
    counts = np.floor(exact).astype(int)
    remainder = episodes - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    for index in order[:remainder]:
        counts[int(index)] += 1
    classes = [
        name for name, count in zip(names, counts, strict=True)
        for _ in range(int(count))
    ]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(classes)
    return classes


def fixed_eval_case_sequence(
        episodes: int, seed: int) -> list[tuple[str, int]]:
    """Pair the exact class mix with balanced compiled contact variants."""
    class_sequence = fixed_eval_class_sequence(episodes, seed)
    class_counts = {
        name: class_sequence.count(name) for name in EVAL_CLASS_WEIGHTS}
    cases: list[tuple[str, int]] = []
    for class_index, name in enumerate(EVAL_CLASS_WEIGHTS):
        for occurrence in range(class_counts[name]):
            # The 60- and 180-case suites have per-class counts divisible by
            # three. The class offset also balances the three singleton strata
            # in the nine-episode cluster smoke.
            variant = (
                occurrence + class_index + int(seed)
            ) % EVAL_CONTACT_VARIANTS
            cases.append((name, variant))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(cases)
    return cases


def checkpoint_selection_score(metrics: dict[str, float]) -> tuple[float, ...]:
    """Rank checkpoints by success, safety failures, then p95 force."""
    failure_rate = sum(float(metrics.get(name, 0.0)) for name in (
        "eval/bad_collision_rate", "eval/force_abort_rate",
        "eval/rotation_guard_rate"))
    return (
        float(metrics["eval/seat_success_rate"]),
        -failure_rate,
        -float(metrics["eval/max_force_n_p95"]),
    )


@dataclass
class SeatCurriculumState:
    easy_max_mm: float = 42.0
    outcomes: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    episodes_seen: int = 0
    episodes_since_update: int = 0
    updates: int = 0

    def record(self, statuses: list[str]) -> list[dict[str, float]]:
        events = []
        for status in statuses:
            self.outcomes.append(int(status == "success"))
            self.episodes_seen += 1
            self.episodes_since_update += 1
            if len(self.outcomes) < 100 or self.episodes_since_update < 100:
                continue
            success_rate = float(np.mean(self.outcomes))
            previous = float(self.easy_max_mm)
            if success_rate > 0.80:
                self.easy_max_mm = max(8.0, self.easy_max_mm - 5.0)
            elif success_rate < 0.10:
                self.easy_max_mm = min(42.0, self.easy_max_mm + 3.0)
            self.episodes_since_update -= 100
            self.updates += 1
            events.append({
                "success_rate": success_rate,
                "previous_easy_max_mm": previous,
                "easy_max_mm": float(self.easy_max_mm),
                "update": float(self.updates),
            })
        return events

    def to_dict(self) -> dict[str, Any]:
        return {
            "easy_max_mm": float(self.easy_max_mm),
            "outcomes": list(self.outcomes),
            "episodes_seen": int(self.episodes_seen),
            "episodes_since_update": int(self.episodes_since_update),
            "updates": int(self.updates),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SeatCurriculumState":
        state = cls(easy_max_mm=float(payload.get("easy_max_mm", 42.0)))
        state.outcomes.extend(int(value) for value in payload.get("outcomes", []))
        state.episodes_seen = int(payload.get("episodes_seen", 0))
        state.episodes_since_update = int(payload.get("episodes_since_update", 0))
        state.updates = int(payload.get("updates", 0))
        return state


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
    parser.add_argument("--learning-rate", type=float)
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
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--final-eval-episodes", type=int, default=180)
    parser.add_argument("--tensorboard-log", type=Path,
                        help="TensorBoard root (keep under the work checkout)")
    parser.add_argument("--wandb-run-name",
                        help="unique W&B name; launcher includes commit/job id")
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", "tar2"),
        help="W&B team entity; organizations cannot directly own runs",
    )
    parser.add_argument(
        "--wandb-organization",
        default=os.environ.get(
            "WANDB_ORGANIZATION",
            "anshulmistry1-the-university-of-texas-at-austin-org"),
    )
    parser.add_argument(
        "--require-online-wandb", action=argparse.BooleanOptionalAction,
        default=os.environ.get("WANDB_REQUIRE_ONLINE", "0") == "1",
    )
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
    if args.final_eval_episodes <= 0:
        parser.error("--final-eval-episodes must be positive")
    canonical = CANONICAL_STAGE[args.stage]
    if args.learning_rate is None:
        args.learning_rate = 3e-4 if canonical == "bootstrap" else 1e-4
    return args


def _finite_stats(values: list[float], *, percentile: float | None = None) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not finite.size:
        return float("nan")
    if percentile is not None:
        return float(np.percentile(finite, percentile))
    return float(np.mean(finite))


def evaluate_seat(model, *, stage: str, episodes: int, seed: int):
    """Run the fixed deployment-matched suite for checkpoint comparison."""
    from RL.student_teacher.seat_env import make_seat_env

    # `_compiled_seed_for_stage` selects a contact geometry by seed modulo 3.
    # Keep one persistent environment per geometry, then use the frozen case
    # sequence to select among them. A single environment here would silently
    # evaluate every checkpoint on only one compiled contact geometry.
    variant_seed_base = int(seed) - int(seed) % EVAL_CONTACT_VARIANTS
    envs = [
        make_seat_env(
            "deployment", seed=variant_seed_base + variant,
            domain_randomization=True)
        for variant in range(EVAL_CONTACT_VARIANTS)
    ]
    reset_cases = fixed_eval_case_sequence(episodes, seed)
    traces: list[dict[str, Any]] = []
    final_depths, max_forces, final_lateral, steps_to_seat = [], [], [], []
    status_counts: dict[str, int] = {}
    jam_count = 0
    try:
        for episode, (reset_class, variant) in enumerate(reset_cases):
            env = envs[variant]
            obs, reset_info = env.reset(
                seed=seed + episode,
                options={"seat_reset_class": reset_class},
            )
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
                "reset_class": reset_class,
                "contact_variant_index": int(variant),
                "compiled_contact_seed": int(env._compiled_seed),
                "start_depth_mm": 1e3 * start_depth,
                "start_actor_lateral_mm": 1e3 * float(reset_info.get(
                    "seat_reset_delivered_actor_lateral_m", float("nan"))),
                "start_physical_lateral_mm": 1e3 * float(reset_info.get(
                    "seat_reset_delivered_physical_lateral_m", float("nan"))),
                "max_depth_mm": 1e3 * max_depth,
                "status": status,
                "max_force_n": max_force,
            })
    finally:
        for env in envs:
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
        "eval/rotation_guard_rate": status_counts.get("rotation_guard", 0) / n,
    }
    for reset_class in EVAL_CLASS_WEIGHTS:
        rows = [row for row in traces if row["reset_class"] == reset_class]
        successes = sum(row["status"] == "success" for row in rows)
        metrics[f"eval/{reset_class}_success_rate"] = (
            successes / len(rows) if rows else float("nan"))
        for status in ("bad_collision", "force_abort", "rotation_guard"):
            metrics[f"eval/{reset_class}_{status}_rate"] = (
                sum(row["status"] == status for row in rows) / len(rows)
                if rows else float("nan"))
        metrics[f"eval/{reset_class}_max_force_n_p95"] = _finite_stats(
            [row["max_force_n"] for row in rows], percentile=95)
    for compiled_seed in sorted({
            row["compiled_contact_seed"] for row in traces}):
        rows = [
            row for row in traces
            if row["compiled_contact_seed"] == compiled_seed
        ]
        prefix = f"eval/contact_{compiled_seed}"
        metrics[f"{prefix}_success_rate"] = (
            sum(row["status"] == "success" for row in rows) / len(rows))
        metrics[f"{prefix}_safety_failure_rate"] = (
            sum(row["status"] in {
                "bad_collision", "force_abort", "rotation_guard",
            } for row in rows) / len(rows))
        metrics[f"{prefix}_max_force_n_p95"] = _finite_stats(
            [row["max_force_n"] for row in rows], percentile=95)
    return metrics, traces, status_counts


def _init_wandb(args, config):
    """Initialize W&B and fail closed when the launcher requires online logs."""
    try:
        import wandb
    except Exception as exc:
        print(f"[wandb] disabled: import failed: {exc}", flush=True)
        return None

    requested_mode = os.environ.get("WANDB_MODE", "online")
    init_kwargs = dict(
        entity=args.wandb_entity,
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
            if args.require_online_wandb:
                raise RuntimeError(
                    "online W&B is required but WANDB_MODE=offline") from exc
            print(f"[wandb] disabled: offline init failed: {exc}", flush=True)
            return None
        if args.require_online_wandb:
            raise RuntimeError(
                f"required online W&B init failed for entity "
                f"{args.wandb_entity!r}: {exc}") from exc
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
        if actual_offline and args.require_online_wandb:
            run.finish(exit_code=1, quiet=True)
            raise RuntimeError("W&B initialized offline while online mode is required")
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
        for metric_glob in (
                "reset/*", "termination/*", "reward/*", "curriculum/*"):
            run.define_metric(metric_glob, step_metric="global_step")
    return run


def export_actor(
        model, out: Path, sample_actor: np.ndarray, *,
        checkpoint_path: Path | None = None) -> dict[str, Any]:
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
        "checkpoint": str((checkpoint_path or (
            out.parent / "model.zip")).resolve()),
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
              "canonical_stage": CANONICAL_STAGE[args.stage],
              "requested_wandb_organization": args.wandb_organization,
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
                checkpoints = args.out / "checkpoints"
                checkpoints.mkdir(parents=True, exist_ok=True)
                versioned = checkpoints / (
                    f"model_{int(self.num_timesteps):09d}_steps")
                self.model.save(versioned)
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
            self.best_score = None
            best_path = args.out / "best_evaluation.json"
            if best_path.is_file():
                previous = json.loads(best_path.read_text())
                self.best_score = self._selection_score(previous["metrics"])

        @staticmethod
        def _selection_score(metrics):
            return checkpoint_selection_score(metrics)

        def _run_eval(self):
            metrics, traces, outcomes = evaluate_seat(
                self.model, stage=args.stage, episodes=args.eval_episodes,
                seed=90_000 + args.seed * 1_000)
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
            score = self._selection_score(metrics)
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.model.save(args.out / "best_model")
                (args.out / "best_evaluation.json").write_text(
                    json.dumps(payload, indent=2) + "\n")
                (args.out / "best_timestep.txt").write_text(
                    f"{int(self.num_timesteps)}\n")
            if wandb_run is not None:
                columns = [
                    "episode", "reset_class", "compiled_contact_seed",
                    "start_depth_mm",
                    "start_actor_lateral_mm", "start_physical_lateral_mm",
                    "max_depth_mm", "status", "max_force_n",
                ]
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

    class SynchronizedSeatCurriculum(BaseCallback):
        """Own one SBC state and push each boundary update to every worker."""

        def __init__(self):
            super().__init__()
            self.path = args.out / "curriculum_state.json"
            if args.resume and self.path.is_file():
                self.state = SeatCurriculumState.from_dict(
                    json.loads(self.path.read_text()))
            else:
                self.state = SeatCurriculumState()

        def _persist(self):
            payload = self.state.to_dict()
            payload["updated_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.path.write_text(json.dumps(payload, indent=2) + "\n")

        def _apply_boundary(self):
            self.training_env.env_method(
                "set_curriculum_easy_max_mm", self.state.easy_max_mm)

        def _on_training_start(self):
            if CANONICAL_STAGE[args.stage] == "bootstrap":
                self._apply_boundary()
                self._persist()

        def _on_step(self):
            if CANONICAL_STAGE[args.stage] != "bootstrap":
                return True
            infos = list(self.locals.get("infos") or ())
            dones = np.asarray(
                self.locals.get("dones", np.zeros(len(infos), dtype=bool)),
                dtype=bool).reshape(-1)
            statuses = [
                str(info.get("term_status") or "timeout")
                for index, info in enumerate(infos)
                if index < len(dones) and dones[index]
            ]
            for event in self.state.record(statuses):
                self._apply_boundary()
                self._persist()
                payload = {
                    "global_step": int(self.num_timesteps),
                    "curriculum/easy_max_mm": event["easy_max_mm"],
                    "curriculum/window_success_rate": event["success_rate"],
                    "curriculum/updates": event["update"],
                    "curriculum/episodes_seen": self.state.episodes_seen,
                }
                if wandb_run is not None:
                    wandb_run.log(payload)
                print("[curriculum] " + json.dumps(payload), flush=True)
            return True

        def _on_training_end(self):
            if CANONICAL_STAGE[args.stage] == "bootstrap":
                self._persist()

    evaluator = SeatEvaluator(args.eval_freq or args.checkpoint_freq)
    curriculum = SynchronizedSeatCurriculum()

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
            self.reset_classes: dict[str, int] = {}
            self.reset_depths_mm: list[float] = []
            self.reset_actor_lateral_mm: list[float] = []
            self.reset_physical_lateral_mm: list[float] = []
            self.term_counts = {
                status: 0 for status in (
                    "success", "bad_collision", "force_abort", "off_limit",
                    "rotation_guard", "timeout",
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
                    reset_class = str(info.get("seat_reset_class", "legacy"))
                    self.reset_classes[reset_class] = (
                        self.reset_classes.get(reset_class, 0) + 1)
                    self.reset_depths_mm.append(1e3 * float(info.get(
                        "seat_reset_delivered_depth_m", float("nan"))))
                    self.reset_actor_lateral_mm.append(1e3 * float(info.get(
                        "seat_reset_delivered_actor_lateral_m", float("nan"))))
                    self.reset_physical_lateral_mm.append(1e3 * float(info.get(
                        "seat_reset_delivered_physical_lateral_m", float("nan"))))
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
            self.reset_classes.clear()
            self.reset_depths_mm.clear()
            self.reset_actor_lateral_mm.clear()
            self.reset_physical_lateral_mm.clear()
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
                for values, name in (
                    (self.reset_depths_mm, "delivered_depth_mm"),
                    (self.reset_actor_lateral_mm, "delivered_actor_lateral_mm"),
                    (self.reset_physical_lateral_mm,
                     "delivered_physical_lateral_mm"),
                ):
                    finite = [value for value in values if np.isfinite(value)]
                    if finite:
                        payload[f"reset/{name}_mean"] = float(np.mean(finite))
                        payload[f"reset/{name}_min"] = float(np.min(finite))
                        payload[f"reset/{name}_max"] = float(np.max(finite))
                for reset_class, count in self.reset_classes.items():
                    payload[f"reset/class_{reset_class}_fraction"] = (
                        count / self.reset_count)
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
        learning_rate=args.learning_rate,
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
                curriculum, CheckpointCallback(args.checkpoint_freq),
                evaluator, video, live_wandb,
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
    best_model_path = args.out / "best_model.zip"
    if not best_model_path.is_file():
        raise RuntimeError("training completed without a selected best_model.zip")
    deploy_model = SAC.load(best_model_path, device="auto")
    final_metrics, final_traces, final_outcomes = evaluate_seat(
        deploy_model,
        stage=args.stage,
        episodes=args.final_eval_episodes,
        seed=190_000 + args.seed * 1_000,
    )
    final_payload = {
        "selected_checkpoint": str(best_model_path.resolve()),
        "selected_timestep": int((args.out / "best_timestep.txt").read_text()),
        "stage": args.stage,
        "episodes": int(args.final_eval_episodes),
        "metrics": final_metrics,
        "outcomes": final_outcomes,
        "traces": final_traces,
    }
    (args.out / "final_selection_evaluation.json").write_text(
        json.dumps(final_payload, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.log({
            "global_step": int(model.num_timesteps),
            **{key.replace("eval/", "final_eval/"): value
               for key, value in final_metrics.items()},
        })
    sample_env = make_seat_env(
        "deployment", seed=290_000 + args.seed,
        domain_randomization=True)
    try:
        sample_obs, _ = sample_env.reset(seed=290_000 + args.seed)
        export_actor(
            deploy_model, args.out / "best_seat_actor.ts", sample_obs["actor"],
            checkpoint_path=best_model_path)
        export_actor(
            deploy_model, args.out / "seat_actor.ts", sample_obs["actor"],
            checkpoint_path=best_model_path)
    finally:
        sample_env.close()
    (args.out / "RUN_COMPLETE").touch()
    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
