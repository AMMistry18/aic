"""
Logging + video callbacks for the residual-SAC trainer.

Provides SB3 callbacks:
    MetricsLogger  - writes per-episode JSONL records (return, term,
                     per-component reward, force, image-L1, etc.)
    VideoRecorder  - every N episodes, records a full roll-out of the
                     current policy and saves it as an MP4 (or .npz
                     fallback if imageio-ffmpeg is unavailable).
    WandbLogger    - optional Weights & Biases logger. `train.py` resolves
                     entity/project from CLI, env, or wandb/info.txt and
                     streams grouped metrics (reward/episode, loss/policy, ...).

The metrics log is plain JSONL so you can grep / jq / pandas-inspect
without needing TensorBoard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover
    class BaseCallback:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): pass


# --------------------------------------------------------------------------- #
# JSONL metrics
# --------------------------------------------------------------------------- #


class MetricsLogger(BaseCallback):
    """Write per-episode metrics to a JSONL file.

    The callback inspects the env's Monitor wrapper for episode returns
    and pulls the per-component reward breakdown out of `info` for the
    last step of each episode.
    """

    def __init__(self, log_path: Path, verbose: int = 0, append: bool = False):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # truncate on construction unless resuming (append=True keeps history)
        if not append:
            self.log_path.write_text("")
        self._f = open(self.log_path, "a", buffering=1)
        self._current_ep: dict[str, Any] = {}
        self._ep_count = 0

    def _on_step(self) -> bool:
        infos = self.model.ep_info_buffer
        # SB3's ep_info_buffer only contains completed episodes; on the
        # step they complete, the prior step's info dict is *not* in the
        # buffer. We instead patch the env's Monitor `info` output by
        # reading the last `info` arg from this step.
        # SB3 calls _on_step with locals that include the latest infos,
        # but to keep it generic we read from self.locals.
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            self._ep_count += 1
            rec = {
                "ep": self._ep_count,
                "step": int(self.num_timesteps),
                "return": float(ep["r"]),
                "length": int(ep["l"]),
                "term": info.get("term_status", "unknown"),
                "image_l1_final": float(info.get("image_l1_norm", float("nan"))),
                "f_z_mean": float(info.get("f_z_mean", float("nan"))),
                "f_z_max": float(info.get("f_z_max", float("nan"))),
                "plug_axis_error_rad": float(info.get("plug_axis_error_rad", float("nan"))),
                "plug_port_penetration_m": float(info.get("plug_port_penetration_m", float("nan"))),
                "plug_port_penetration_excess_m": float(
                    info.get("plug_port_penetration_excess_m", float("nan"))),
                "t_start": float(info.get("wallclock", 0.0)),
            }
            b = info.get("breakdown")
            if b is not None:
                rec.update({
                    "image": float(b.image),
                    "force": float(b.force),
                    "depth": float(getattr(b, "depth", 0.0)),
                    "xy": float(b.xy),
                    "action": float(b.action),
                    "lateral": float(b.lateral),
                    "axis": float(getattr(b, "axis", 0.0)),
                    "collision": float(getattr(b, "collision", 0.0)),
                    "done": float(b.done),
                })
            self._f.write(json.dumps(rec) + "\n")
        return True

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# video recording
# --------------------------------------------------------------------------- #


class VideoRecorder(BaseCallback):
    """Record a full episode roll-out every N episodes and save as MP4.

    Uses the env's own `render()` method (which our LastInchInsertEnv
    returns a (H, W, 3) uint8 frame from). Frames are stacked into an
    MP4 via imageio if available, else saved as a .npz of frames.
    """

    def __init__(self, env, video_dir: Path, every_n_episodes: int = 25,
                 max_steps: int = 600, fps: int = 20, verbose: int = 1):
        super().__init__(verbose=verbose)
        self._env = env
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.every_n = max(int(every_n_episodes), 1)
        self.max_steps = int(max_steps)
        self.fps = int(fps)
        self._ep_count = 0
        # detect writer
        self._writer_kind = self._detect_writer()
        self._ext = "mp4" if self._writer_kind == "imageio-ffmpeg" else "npz"

    @staticmethod
    def _detect_writer() -> str:
        try:
            import imageio_ffmpeg  # noqa: F401
            import imageio  # noqa: F401
            return "imageio-ffmpeg"
        except Exception:
            return "npz"

    def _on_step(self) -> bool:
        # Episode boundaries come through locals["dones"] when SB3's
        # vec-env wrapper reports a done. Only count *true* episode
        # completions, not every-step noise. SB3 sets done[i]=True when
        # env.step returns terminated OR truncated.
        dones = np.asarray(self.locals.get("dones", [False])).astype(bool)
        # Count the number of *new* episode completions this step
        prev_count = self._ep_count
        new_completions = int(np.sum(dones))
        if new_completions == 0:
            return True
        self._ep_count += new_completions
        # Only record every Nth completion
        completed_idx = self._ep_count - 1
        if completed_idx % self.every_n != 0:
            return True
        # Record a fresh roll-out with the *current* policy
        try:
            self._record_one()
        except Exception as exc:
            if self.verbose:
                print(f"[video] record failed: {exc}")
        return True

    def _record_one(self) -> None:
        # Unwrap the SB3 chain to reach the bare gym env.
        # The chain is typically:
        #   VecTransposeImage
        #     -> DummyVecEnv / SubprocVecEnv
        #        -> envs[0] (a Monitor)
        #           -> env (a TimeLimit)
        #              -> env (RecordedRolloutEnv / LastInchInsertEnv)
        env = self._env
        for _ in range(12):
            if hasattr(env, "envs") and len(env.envs) > 0:
                env = env.envs[0]
            elif hasattr(env, "venv"):
                env = env.venv
            elif hasattr(env, "env"):
                env = env.env
            else:
                break

        try:
            obs, _ = env.reset()
        except Exception as exc:
            if self.verbose:
                print(f"[video] reset failed: {exc}")
            return

        frames = []
        for _ in range(self.max_steps):
            try:
                action, _ = self.model.predict(obs, deterministic=True)
            except Exception as exc:
                if self.verbose:
                    print(f"[video] predict failed: {exc}")
                break
            try:
                frame = env.render()
            except Exception:
                frame = None
            if frame is not None:
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                frames.append(frame)
            try:
                obs, _rew, term, trunc, _info = env.step(action)
            except Exception as exc:
                if self.verbose:
                    print(f"[video] step failed: {exc}")
                break
            if term or trunc:
                break
        if not frames:
            if self.verbose:
                print(f"[video] ep {self._ep_count}: no frames captured")
            return

        out_path = self.video_dir / f"ep_{self._ep_count:06d}.{self._ext}"
        if self._writer_kind == "imageio-ffmpeg":
            import imageio.v2 as imageio
            imageio.mimsave(str(out_path), frames, fps=self.fps, codec="libx264")
        else:
            np.savez_compressed(str(out_path), frames=np.stack(frames, axis=0))
        if self.verbose:
            print(f"[video] wrote {out_path} ({len(frames)} frames)")


class WandbVideoRecorder(BaseCallback):
    """Record evaluation rollouts into W&B as media, without touching training env.

    `make_eval_env` must return `(vec_env, render_env)`: a VecEnv matching the
    model observation preprocessing, plus the underlying renderable env.
    """

    def __init__(self, make_eval_env, every_n_episodes: int = 25,
                 max_steps: int = 200, fps: int = 20,
                 key: str = "eval/rollout_video",
                 record_on_training_end: bool = True,
                 deterministic: bool = True,
                 enabled: bool = True,
                 verbose: int = 1):
        super().__init__(verbose=verbose)
        self.make_eval_env = make_eval_env
        self.every_n = max(int(every_n_episodes), 1)
        self.max_steps = int(max_steps)
        self.fps = int(fps)
        self.key = key
        self.record_on_training_end = bool(record_on_training_end)
        self.deterministic = bool(deterministic)
        self.enabled = bool(enabled)
        self._ep_count = 0
        self._recorded_steps: set[int] = set()

    def _on_step(self) -> bool:
        if not self.enabled:
            return True
        dones = np.asarray(self.locals.get("dones", [False])).astype(bool)
        new_completions = int(np.sum(dones))
        if new_completions == 0:
            return True
        prev = self._ep_count
        self._ep_count += new_completions
        if prev == 0 or (self._ep_count - 1) % self.every_n == 0:
            self._record_one(tag=f"episode_{self._ep_count:06d}")
        return True

    def _on_training_end(self) -> None:
        if self.enabled and self.record_on_training_end:
            self._record_one(tag="final")

    @staticmethod
    def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return frame
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except Exception:
            font = None
        width = max([int(draw.textlength(s, font=font)) for s in lines] or [0]) + 12
        height = 6 + 18 * len(lines) + 6
        draw.rectangle([0, 0, width, height], fill=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((6, 6 + i * 18), line, fill=(255, 255, 255), font=font)
        return np.asarray(im)

    def _record_one(self, tag: str) -> None:
        if int(self.num_timesteps) in self._recorded_steps and tag != "final":
            return
        try:
            import wandb
        except Exception as exc:
            if self.verbose:
                print(f"[wandb-video] wandb import failed: {exc}", flush=True)
            return
        if wandb.run is None:
            if self.verbose:
                print("[wandb-video] no live wandb run yet; skipping", flush=True)
            return

        vec_env = None
        render_env = None
        frames: list[np.ndarray] = []
        last_info: dict[str, Any] = {}
        last_reward = 0.0
        term_status = None
        try:
            vec_env, render_env = self.make_eval_env()
            obs = vec_env.reset()
            for step in range(self.max_steps):
                try:
                    frame = render_env.render()
                except Exception:
                    frame = None
                if frame is not None:
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    if frame.ndim == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                    axis_deg = np.degrees(float(last_info.get("plug_axis_error_rad", 0.0)))
                    excess_mm = 1000.0 * float(last_info.get("plug_port_penetration_excess_m", 0.0))
                    depth = float(last_info.get("depth_norm", 0.0))
                    lines = [
                        f"{tag} train_step={int(self.num_timesteps)} eval_step={step}",
                        f"term={term_status} reward={last_reward:.2f}",
                        f"depth={depth:.3f} axis={axis_deg:.1f}deg excess={excess_mm:.2f}mm",
                    ]
                    frames.append(self._overlay(frame[:, :, :3], lines))

                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, infos = vec_env.step(action)
                last_reward = float(np.asarray(reward).reshape(-1)[0])
                last_info = dict(infos[0]) if infos else {}
                term_status = last_info.get("term_status", term_status)
                if bool(np.asarray(done).reshape(-1)[0]):
                    break
        except Exception as exc:
            if self.verbose:
                print(f"[wandb-video] record failed: {exc}", flush=True)
            return
        finally:
            if vec_env is not None:
                try:
                    vec_env.close()
                except Exception:
                    pass

        if not frames:
            if self.verbose:
                print("[wandb-video] no frames captured", flush=True)
            return
        try:
            import tempfile
            import imageio.v2 as imageio
            with tempfile.TemporaryDirectory(prefix="aic_wandb_video_") as td:
                path = Path(td) / f"{tag}.mp4"
                imageio.mimsave(str(path), frames, fps=self.fps, codec="libx264", quality=7)
                payload = {
                    self.key: wandb.Video(str(path), fps=self.fps, format="mp4"),
                    "eval/video_frames": len(frames),
                    "eval/video_term_status": str(term_status),
                }
                wandb.log(payload, step=int(self.num_timesteps))
        except Exception as exc:
            if self.verbose:
                print(f"[wandb-video] upload failed: {exc}", flush=True)
            return
        self._recorded_steps.add(int(self.num_timesteps))
        if self.verbose:
            print(f"[wandb-video] logged {tag} ({len(frames)} frames) to {self.key}",
                  flush=True)


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #


def plot_dashboard(metrics_path: Path, out_path: Optional[Path] = None) -> Optional[Path]:
    """Read a metrics.jsonl and produce a 2x2 dashboard PNG.

    Plots:
        (0,0) episode return over training step
        (0,1) term-status histogram (success / timeout / force_abort / off_limit)
        (1,0) image-L1 over training step
        (1,1) per-component reward mean over the last 50 episodes
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    import pandas as pd

    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return None
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(df["step"], df["return"], alpha=0.4, lw=0.8)
    if len(df) > 20:
        df["return_smooth"] = df["return"].rolling(20, min_periods=1).mean()
        ax.plot(df["step"], df["return_smooth"], color="C0", lw=1.5, label="20-ep MA")
        ax.legend()
    ax.set_xlabel("env step")
    ax.set_ylabel("episode return")
    ax.set_title("return over training")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    term_counts = df["term"].value_counts()
    ax.bar(term_counts.index, term_counts.values)
    ax.set_title(f"termination ({len(df)} episodes)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.plot(df["step"], df["image_l1_final"], alpha=0.4, lw=0.8, label="image-L1 (final)")
    if "f_z_mean" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["step"], df["f_z_mean"], alpha=0.3, lw=0.6, color="C1",
                 label="f_z mean")
        ax2.set_ylabel("Fz (N)")
    ax.set_xlabel("env step")
    ax.set_ylabel("image L1 (norm)")
    ax.set_title("image-L1 (and F/T) over training")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    components = ["image", "force", "depth", "xy", "action", "lateral",
                  "axis", "collision", "done"]
    means = [df[c].mean() if c in df.columns else 0.0 for c in components]
    ax.bar(components, means)
    ax.set_title("mean per-component reward")
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if out_path is None:
        out_path = metrics_path.with_name("dashboard.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# curriculum scheduler
# --------------------------------------------------------------------------- #


class CurriculumScheduler(BaseCallback):
    """Reverse-curriculum start-pose scheduler (Florensa 2018 style).

    Level 0   = near the goal (small jitter, easy).
    Level 1   = full last-inch envelope (hard).
    Advance when the success rate over the last `--curriculum-eval-window`
    episodes exceeds `--curriculum-advance-threshold`. Retreat when it
    falls below `--curriculum-retreat-threshold`. The level is also
    written to a file so `--resume` keeps it.
    """

    def __init__(self, env=None, eval_window: int = 100,
                 advance_threshold: float = 0.6,
                 retreat_threshold: float = 0.15,
                 force_abort_guard: float = 0.25,
                 force_abort_retreat: float = 0.50,
                 step_size: float = 0.05,
                 level_path: Optional[Path] = None):
        super().__init__()
        self._env = env  # may be None when running with SubprocVecEnv
        self.eval_window = int(eval_window)
        self.advance_thr = float(advance_threshold)
        self.retreat_thr = float(retreat_threshold)
        self.force_abort_guard = float(force_abort_guard)
        self.force_abort_retreat = float(force_abort_retreat)
        self.step_size = float(step_size)
        self.level_path = Path(level_path) if level_path is not None else None
        self._current_level: float = 0.0
        self._recent: list[str] = []
        # restore prior level if present
        if self.level_path is not None and self.level_path.exists():
            try:
                lvl = float(self.level_path.read_text().strip())
                self._current_level = lvl
                if self._env is not None and hasattr(self._env, "set_curriculum_level"):
                    self._env.set_curriculum_level(lvl)
                if self.verbose:
                    print(f"[curriculum] resumed level={lvl:.3f}", flush=True)
            except Exception:
                pass

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            # update running stats via a sliding window of recent
            # term statuses (we don't have direct access to ep_info_buffer
            # history from here without scanning model state)
            self._record(info)
            self._maybe_update()
        return True

    def _record(self, info: dict) -> None:
        self._recent.append(str(info.get("term_status", "unknown")))
        if len(self._recent) > self.eval_window:
            self._recent = self._recent[-self.eval_window:]

    def _maybe_update(self) -> None:
        if len(self._recent) < self.eval_window:
            return
        success_rate = sum(1 for s in self._recent if s == "success") / len(self._recent)
        force_abort_rate = sum(1 for s in self._recent if s == "force_abort") / len(self._recent)
        bad_collision_rate = sum(1 for s in self._recent if s == "bad_collision") / len(self._recent)
        unsafe_rate = force_abort_rate + bad_collision_rate
        cur = self._current_level
        new = cur
        if unsafe_rate >= self.force_abort_retreat:
            new = max(0.0, cur - self.step_size)
        elif success_rate >= self.advance_thr and unsafe_rate <= self.force_abort_guard:
            new = min(1.0, cur + self.step_size)
        elif success_rate <= self.retreat_thr:
            new = max(0.0, cur - self.step_size)
        if new != cur:
            self._current_level = new
            # best-effort: tell the (in-process) env if we have one
            if self._env is not None and hasattr(self._env, "set_curriculum_level"):
                self._env.set_curriculum_level(new)
            # persistent: write the level to a file. The env reads it
            # at every reset() so this works with SubprocVecEnv too.
            if self.level_path is not None:
                tmp = self.level_path.with_suffix(self.level_path.suffix + ".tmp")
                tmp.write_text(f"{new:.4f}\n")
                os.replace(tmp, self.level_path)
            self._recent.clear()
            if self.verbose:
                print(f"[curriculum] level {cur:.3f} -> {new:.3f} "
                      f"(success_rate={success_rate:.2f}, "
                      f"force_abort_rate={force_abort_rate:.2f}, "
                      f"bad_collision_rate={bad_collision_rate:.2f} "
                      f"over fresh window={self.eval_window})",
                      flush=True)

    def get_current_level(self) -> float:
        return float(self._current_level)


# --------------------------------------------------------------------------- #
# checkpoint manager (crash-safe, resumable)
# --------------------------------------------------------------------------- #


class CheckpointManager(BaseCallback):
    """Periodically persist a *single canonical* resume point so a killed run
    never starts over.

    Writes (all atomically — tmp file + os.replace):
        <out>/model.zip            every `model_every_steps` env steps
                                   (SB3 zip = weights + all optimizers +
                                    num_timesteps + ent_coef state)
        <out>/replay_buffer.pkl    every `buffer_every_steps` env steps
                                   (big — SAC is off-policy so a slightly
                                    stale buffer on resume is fine)

    The curriculum level is persisted separately by CurriculumScheduler into
    <out>/curriculum_level.txt (the env re-reads it on every reset). Together
    these three files are everything `train.py --resume` needs.

    `save_freq` counters are in ENV steps (not vec steps); we convert using
    num_timesteps which SB3 advances by n_envs per vec step.
    """

    def __init__(self, out_dir: Path, model_every_steps: int = 20_000,
                 buffer_every_steps: int = 100_000, save_buffer: bool = True,
                 verbose: int = 1):
        super().__init__(verbose=verbose)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_every = max(int(model_every_steps), 1)
        self.buffer_every = max(int(buffer_every_steps), 1)
        self.save_buffer = bool(save_buffer)
        self._next_model = self.model_every
        self._next_buffer = self.buffer_every
        self._last_model_step = -1
        self._last_buffer_step = -1

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        if t >= self._next_model:
            self._next_model = t + self.model_every
            self.save_model()
        if self.save_buffer and t >= self._next_buffer:
            self._next_buffer = t + self.buffer_every
            self.save_replay_buffer()
        return True

    def save_model(self) -> None:
        """Atomically write <out>/model.zip (skips a redundant same-step save)."""
        t = int(self.num_timesteps)
        if t == self._last_model_step:
            return
        self._last_model_step = t
        try:
            tmp = self.out_dir / ".model.tmp.zip"
            self.model.save(str(tmp))
            os.replace(tmp, self.out_dir / "model.zip")
            if self.verbose:
                print(f"[ckpt] model.zip @ step {int(self.num_timesteps)}", flush=True)
        except Exception as exc:
            print(f"[ckpt] model save failed: {exc}", flush=True)

    def save_replay_buffer(self) -> None:
        """Atomically write <out>/replay_buffer.pkl (skips a redundant same-step save)."""
        t = int(self.num_timesteps)
        if t == self._last_buffer_step:
            return
        self._last_buffer_step = t
        try:
            tmp = self.out_dir / ".rb.tmp.pkl"
            self.model.save_replay_buffer(str(tmp))
            os.replace(tmp, self.out_dir / "replay_buffer.pkl")
            if self.verbose:
                sz = (self.out_dir / "replay_buffer.pkl").stat().st_size / 1e9
                print(f"[ckpt] replay_buffer.pkl @ step {int(self.num_timesteps)} "
                      f"({sz:.1f} GB)", flush=True)
        except Exception as exc:
            print(f"[ckpt] buffer save failed: {exc}", flush=True)


# --------------------------------------------------------------------------- #
# progress printer (stdout logging for headless / interactive use)
# --------------------------------------------------------------------------- #


class ProgressPrinter(BaseCallback):
    """Print periodic training stats to stdout.

    Every `--log-every` env steps, emits a single line with:
        step, ep, mean_return(50), success_rate(50), curriculum_level,
        q_loss, alpha, image_l1_mean
    """

    def __init__(self, log_every_steps: int = 500):
        super().__init__()
        self.log_every = max(int(log_every_steps), 50)
        self._recent_returns: list[float] = []
        self._recent_terms: list[str] = []
        self._recent_l1: list[float] = []
        self._next_print_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self._recent_returns.append(float(ep["r"]))
                self._recent_terms.append(str(info.get("term_status", "?")))
                self._recent_l1.append(float(info.get("image_l1_norm", float("nan"))))
        if len(self._recent_returns) > 50:
            self._recent_returns = self._recent_returns[-50:]
            self._recent_terms = self._recent_terms[-50:]
            self._recent_l1 = self._recent_l1[-50:]

        if self.num_timesteps < self._next_print_step:
            return True
        self._next_print_step = self.num_timesteps + self.log_every

        ret = self._recent_returns
        terms = self._recent_terms
        l1 = self._recent_l1
        n = max(len(ret), 1)
        succ = sum(1 for s in terms if s == "success") / len(terms) if terms else 0.0
        # pull Q-loss / alpha from SB3's logger if present.
        # SB3 SAC logs the combined critic loss as "train/critic_loss" and
        # the entropy coefficient as "train/ent_coef" (NOT qf1_loss /
        # entropy_coef — those keys never exist and always read nan).
        qloss = float(self.logger.name_to_value.get("train/critic_loss", float("nan")))
        alpha = float(self.logger.name_to_value.get("train/ent_coef", float("nan")))
        l1_mean = float(np.nanmean([x for x in l1 if x == x])) if l1 else float("nan")
        print(
            f"[step {self.num_timesteps:>8d}] "
            f"ret(50)={sum(ret)/n:+7.1f}  "
            f"succ(50)={succ:5.1%}  "
            f"img_l1={l1_mean:.3f}  "
            f"q_loss={qloss:7.2f}  "
            f"alpha={alpha:.3f}",
            flush=True,
        )
        return True


# --------------------------------------------------------------------------- #
# Weights & Biases logger
# --------------------------------------------------------------------------- #

# Default target entity/project. `train.py` resolves CLI/env/wandb/info.txt and
# passes explicit values; these are just safe fallbacks.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "1-inch-intrinsic-policy")


class WandbLogger(BaseCallback):
    """Optional Weights & Biases logger for the residual-SAC trainer.

    Logs grouped metrics matching the rest of this repo's conventions:

        train/global_step          (int)
        train/episode_length       (int)
        train/learning_rate        (float)
        eval/success_rate          (float, rolling over --wandb-success-window)
        reward/episode             (float)
        reward/mean                (float, rolling mean of episode returns)
        loss/policy                (float, SAC actor_loss)
        loss/critic                (float, mean of qf1_loss + qf2_loss)
        loss/critic_q1             (float)
        loss/critic_q2             (float)
        loss/entropy_coef          (float, SAC alpha)
        rollout/image_l1_norm      (float)
        rollout/f_z                (float)

    Does NOT upload checkpoints, videos, replay buffers, or model
    artifacts by default (free W&B tier has limited storage). See
    `_maybe_log_artifact` for an opt-in hook — disabled by default.

    Requirements:
        * `wandb` importable (already pulled in transitively by ultralytics
          in this repo's pixi env; pinned explicitly in pixi.toml).
        * WANDB_API_KEY env var unless WANDB_MODE=offline.

    Usage from train.py:
        cb = WandbLogger(
            run_name=args.wandb_run_name,
            config={"steps": args.steps, ...},
            log_every_steps=args.wandb_log_every,
            success_window=args.wandb_success_window,
            out_dir=args.out,
        )
        callbacks.append(cb)
        try:
            model.learn(..., callback=callbacks)
        finally:
            cb.finish()  # ensures wandb.finish() on early exit / errors
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        log_every_steps: int = 500,
        success_window: int = 100,
        out_dir: Optional[Path] = None,
        entity: Optional[str] = WANDB_ENTITY,
        project: Optional[str] = WANDB_PROJECT,
        enabled: bool = True,
    ):
        super().__init__()
        self.run_name = run_name
        self.config = dict(config) if config else {}
        self.log_every_steps = max(int(log_every_steps), 1)
        self.success_window = max(int(success_window), 1)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.entity = entity
        self.project = project
        # `enabled=False` is a hard kill-switch — no wandb calls at all.
        self._enabled = bool(enabled)
        self._wandb = None  # lazy import
        self._run = None
        self._initialised = False
        self._init_error: Optional[str] = None
        # rolling stats for episode-end metrics
        self._recent_returns: list[float] = []
        self._recent_terms: list[str] = []
        self._next_log_step = 0
        # optional checkpoint-artifact hook (kept OFF by default; see method)
        self._upload_checkpoints = False

    # ------------------------------------------------------------------ #
    # init / shutdown
    # ------------------------------------------------------------------ #

    def _try_init(self) -> bool:
        """Initialise wandb once. Returns True if a run is live."""
        if not self._enabled or self._initialised:
            return self._initialised
        try:
            import wandb  # noqa: F401  (kept as self._wandb for later use)
        except ImportError as exc:
            self._init_error = (
                f"wandb is enabled but the package is not importable: {exc}. "
                "Install wandb or pass --no-wandb to train.py."
            )
            print(f"[wandb] {self._init_error}", flush=True)
            return False

        import wandb as _wandb
        self._wandb = _wandb

        mode_env = os.environ.get("WANDB_MODE", "").strip().lower()
        api_key = os.environ.get("WANDB_API_KEY", "").strip()

        # offline mode is self-sufficient (no API key needed)
        if mode_env == "offline":
            effective_mode = "offline"
        elif mode_env in ("disabled", "false", "0"):
            self._enabled = False
            return False
        else:
            effective_mode = mode_env or "online"
            if not api_key and effective_mode == "online":
                self._init_error = (
                    "WANDB_API_KEY is not set but W&B is enabled. "
                    "Set WANDB_API_KEY in the environment, or run with "
                    "WANDB_MODE=offline, or pass --no-wandb to train.py. "
                    "Refusing to log to the wandb default entity "
                    "to avoid silently landing on the wrong account."
                )
                print(f"[wandb] {self._init_error}", flush=True)
                return False

        try:
            init_kwargs = dict(
                project=self.project or WANDB_PROJECT,
                name=self.run_name or None,
                config=self.config,
                mode=effective_mode,
                reinit=True,
            )
            if self.entity:
                init_kwargs["entity"] = self.entity
            self._run = _wandb.init(
                **init_kwargs,
            )
        except Exception as exc:
            self._init_error = f"wandb.init failed: {exc}"
            print(f"[wandb] {self._init_error}", flush=True)
            return False

        self._initialised = True
        # explicit reminder so the run URL is obvious in stdout
        try:
            url = getattr(self._run, "url", None) or _wandb.run.url
            print(f"[wandb] run live at {url} "
                  f"(entity={self.entity or '<wandb-default>'}, "
                  f"project={self.project or WANDB_PROJECT})",
                  flush=True)
        except Exception:
            pass
        return True

    def finish(self) -> None:
        """Idempotent shutdown — safe to call from finally: blocks."""
        if not self._initialised:
            return
        try:
            if self._wandb is not None:
                self._wandb.finish()
        except Exception as exc:
            print(f"[wandb] finish() raised: {exc}", flush=True)
        finally:
            self._initialised = False
            self._run = None

    def _on_training_end(self) -> None:
        # train.py owns finalization so other callbacks can still log media at
        # training_end before W&B is closed.
        return None

    # ------------------------------------------------------------------ #
    # per-step logging
    # ------------------------------------------------------------------ #

    def _on_step(self) -> bool:
        if not self._enabled:
            return True
        if not self._initialised and not self._try_init():
            # init failed (missing key, etc.) — silently skip without breaking training
            self._enabled = False
            return True

        w = self._wandb
        log_payload: dict[str, Any] = {}

        # --- episode-end metrics (from Monitor wrapper) ---
        for info in self.locals.get("infos", []) or []:
            ep = info.get("episode")
            if ep is None:
                continue
            ret = float(ep["r"])
            length = int(ep["l"])
            term = str(info.get("term_status", "unknown"))
            self._recent_returns.append(ret)
            self._recent_terms.append(term)
            if len(self._recent_returns) > self.success_window:
                self._recent_returns = self._recent_returns[-self.success_window:]
                self._recent_terms = self._recent_terms[-self.success_window:]

            log_payload["reward/episode"] = ret
            log_payload["train/episode_length"] = length
            if "image_l1_norm" in info:
                log_payload["rollout/image_l1_norm"] = float(info["image_l1_norm"])
            if "f_z" in info:
                log_payload["rollout/f_z"] = float(info["f_z"])
            if "plug_axis_error_rad" in info:
                log_payload["geometry/plug_axis_error_deg"] = float(
                    np.degrees(info["plug_axis_error_rad"]))
            if "plug_port_penetration_m" in info:
                log_payload["geometry/plug_port_penetration_mm"] = float(
                    1000.0 * info["plug_port_penetration_m"])
            if "plug_port_penetration_excess_m" in info:
                log_payload["geometry/plug_port_penetration_excess_mm"] = float(
                    1000.0 * info["plug_port_penetration_excess_m"])

        # --- rolling aggregates ---
        if self._recent_returns:
            log_payload["reward/mean"] = float(
                sum(self._recent_returns) / len(self._recent_returns)
            )
        if self._recent_terms:
            succ = sum(1 for s in self._recent_terms if s == "success")
            log_payload["eval/success_rate"] = float(succ / len(self._recent_terms))

        # --- per-N-step SB3 logger passthrough (losses, alpha, lr) ---
        if self.num_timesteps >= self._next_log_step:
            self._next_log_step = self.num_timesteps + self.log_every_steps
            n2v = self.model.logger.name_to_value

            actor = n2v.get("train/actor_loss")
            if actor is not None:
                log_payload["loss/policy"] = float(actor)

            # SB3 SAC logs a single combined critic loss ("train/critic_loss"),
            # not separate qf1/qf2 losses. Use the real key.
            crit = n2v.get("train/critic_loss")
            if crit is not None:
                log_payload["loss/critic"] = float(crit)

            alpha = n2v.get("train/ent_coef")
            if alpha is not None:
                log_payload["loss/entropy_coef"] = float(alpha)

            lr = n2v.get("train/learning_rate")
            if lr is not None:
                log_payload["train/learning_rate"] = float(lr)

        # always emit global_step so wandb has a monotonic x-axis
        log_payload["train/global_step"] = int(self.num_timesteps)

        if len(log_payload) > 1:  # at least global_step plus something
            try:
                w.log(log_payload, step=int(self.num_timesteps))
            except Exception as exc:
                # never break training on a wandb network blip
                if not hasattr(self, "_log_warned"):
                    print(f"[wandb] log() raised: {exc} "
                          "(further errors suppressed)", flush=True)
                    self._log_warned = True

        return True

    # ------------------------------------------------------------------ #
    # optional artifact hook (DISABLED BY DEFAULT — free-tier storage)
    # ------------------------------------------------------------------ #
    #
    # To turn on checkpoint uploads later:
    #   1. Set self._upload_checkpoints = True (e.g. via a CLI flag).
    #   2. Save SB3 .zip somewhere accessible (model.save(...) already does
    #      this into args.out / "model.zip" in train.py).
    #   3. Call _maybe_log_artifact(model_zip_path, artifact_name, metadata).
    #
    # We deliberately do NOT call this anywhere by default.

    def _maybe_log_artifact(self, path: Path, name: str,
                            metadata: Optional[dict] = None) -> None:
        """Opt-in: upload a local file as a wandb artifact.

        Disabled by default. Free W&B tiers have limited storage; do not
        enable this for routine runs.
        """
        if not self._enabled or not self._initialised or not self._upload_checkpoints:
            return
        try:
            art = self._wandb.Artifact(name=name, type="model",
                                       metadata=metadata or {})
            art.add_file(str(path))
            self._wandb.log_artifact(art)
        except Exception as exc:
            print(f"[wandb] artifact upload failed: {exc}", flush=True)


__all__ = [
    "MetricsLogger",
    "VideoRecorder",
    "WandbVideoRecorder",
    "CurriculumScheduler",
    "CheckpointManager",
    "ProgressPrinter",
    "plot_dashboard",
    "WandbLogger",
]
