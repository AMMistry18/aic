"""Image-SAC trainer for last-inch insertion on the real MuJoCo scene.

MVP plumbing (per user, 2026-07-02): get the reverse-curriculum initial-position
setting training, connected to W&B, writing rollout videos to RL/outputs/ so the
pipeline can be verified. Reward is the placeholder image-distance in scene_env.py;
reward shaping comes next.

Run (short smoke):
    pixi run python RL/train_sac.py --timesteps 2000 --run-name smoke
W&B: set WANDB_API_KEY for online logging; otherwise runs offline (synced later with
`wandb sync RL/outputs/wandb/offline-*`).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from RL.scene_env import SceneEnvConfig, SceneInsertEnv

OUT_DIR = Path(__file__).resolve().parent / "outputs"
CURRICULUM_FILE = OUT_DIR / "curriculum_level.txt"


class CurriculumCallback(BaseCallback):
    """Reverse curriculum: level 0 (inserted) -> 1 (full last inch).

    Placeholder schedule (linear over training) pending success-based advance
    (advance on success-rate>=0.6, retreat on <=0.15) once reward shaping lands.
    Shares the level via a file so vec-env workers can read it.
    """
    def __init__(self, total_timesteps, level_file=CURRICULUM_FILE, verbose=0):
        super().__init__(verbose)
        self.total = max(int(total_timesteps), 1)
        self.level_file = Path(level_file)
        self.level_file.parent.mkdir(parents=True, exist_ok=True)

    def _on_training_start(self):
        self._write(0.0)

    def _write(self, lvl):
        self.level_file.write_text(f"{lvl:.4f}")
        for env in self.training_env.envs:
            env.unwrapped.set_curriculum_level(lvl)

    def _on_step(self) -> bool:
        lvl = min(1.0, self.num_timesteps / self.total)
        self._write(lvl)
        self.logger.record("curriculum/level", lvl)
        return True


class VideoCallback(BaseCallback):
    """Every `freq` steps, roll out one greedy episode and save an mp4 to RL/outputs/."""
    def __init__(self, eval_env, freq=1000, max_len=120, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.freq = freq
        self.max_len = max_len
        self._last = 0

    def _save(self, frames, tag):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"rollout_{tag}.mp4"
        try:
            import imageio
            imageio.mimwrite(path, frames, fps=20, macro_block_size=1)
        except Exception as e:
            path = OUT_DIR / f"rollout_{tag}.gif"
            try:
                import imageio
                imageio.mimwrite(path, frames, fps=20)
            except Exception as e2:
                print(f"[video] could not write video ({e}; {e2})"); return None
        return path

    def _rollout(self):
        env = self.eval_env
        obs, _ = env.reset()
        frames = []
        for _ in range(self.max_len):
            a, _ = self.model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            f = env.render()
            if f is not None:
                frames.append(np.asarray(f, dtype=np.uint8))
            if term or trunc:
                break
        return frames

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.freq:
            self._last = self.num_timesteps
            self.eval_env.unwrapped.set_curriculum_level(
                self.training_env.envs[0].unwrapped.get_curriculum_level())
            frames = self._rollout()
            path = self._save(frames, f"step{self.num_timesteps}")
            if path:
                print(f"[video] wrote {path} ({len(frames)} frames)")
                try:
                    import wandb
                    if wandb.run is not None and str(path).endswith(".mp4"):
                        wandb.log({"rollout": wandb.Video(str(path), fps=20, format="mp4")},
                                  step=self.num_timesteps)
                except Exception:
                    pass
        return True


class WandbCallback(BaseCallback):
    """Minimal, offline-safe W&B logger (reward, curriculum, FT). The richer
    RL/logging_utils.py:WandbLogger can be swapped in later."""
    def __init__(self, run_name, config, project="1-inch-intrinsic-policy",
                 entity="satya_anandh", log_every=200, verbose=0):
        super().__init__(verbose)
        self.run_name, self.config = run_name, config
        self.project, self.entity, self.log_every = project, entity, log_every
        self._wandb = None

    def _on_training_start(self):
        if "WANDB_API_KEY" not in os.environ:
            os.environ["WANDB_MODE"] = "offline"
            print("[wandb] no WANDB_API_KEY -> offline mode (sync later)")
        try:
            import wandb
            self._wandb = wandb
            wandb.init(project=self.project, entity=self.entity, name=self.run_name,
                       config=self.config, dir=str(OUT_DIR), reinit=True)
            print(f"[wandb] run started: {wandb.run.name} (mode={os.environ.get('WANDB_MODE','online')})")
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing without wandb")
            self._wandb = None

    def _on_step(self) -> bool:
        if self._wandb is None or self.num_timesteps % self.log_every != 0:
            return True
        env = self.training_env.envs[0].unwrapped
        logd = {"curriculum/level": env.get_curriculum_level()}
        infos = self.locals.get("infos")
        if infos and "tcp_pos" in infos[0]:
            logd["tcp/z"] = float(infos[0]["tcp_pos"][2])
        rew = self.locals.get("rewards")
        if rew is not None:
            logd["train/reward"] = float(np.mean(rew))
        self._wandb.log(logd, step=self.num_timesteps)
        return True

    def _on_training_end(self):
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.finish()


def make_env(port="sfp", images=True):
    cams = ("center_camera",)  # single cam -> 3ch image -> clean SB3 CNN
    target = "sfp_module_link" if port == "sfp" else "sc_tip_link"
    cfg = SceneEnvConfig(include_images=images, cameras=cams, insert_target_body=target)
    return SceneInsertEnv(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2000)
    ap.add_argument("--port", choices=["sfp", "sc"], default="sfp")
    ap.add_argument("--run-name", default="scene_sac_smoke")
    ap.add_argument("--video-freq", type=int, default=1000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_env = DummyVecEnv([lambda: make_env(args.port)])
    eval_env = make_env(args.port)

    model = SAC(
        "MultiInputPolicy", train_env,
        buffer_size=5000, learning_starts=200, batch_size=128,
        train_freq=1, gradient_steps=1, verbose=1,
        tensorboard_log=None,
    )
    cbs = CallbackList([
        CurriculumCallback(args.timesteps),
        WandbCallback(args.run_name, config=vars(args)),
        VideoCallback(eval_env, freq=args.video_freq),
    ])
    model.learn(total_timesteps=args.timesteps, callback=cbs, progress_bar=False)
    model.save(str(OUT_DIR / f"sac_{args.port}_{args.run_name}"))
    print(f"[done] model + videos in {OUT_DIR}")
    eval_env.close()


if __name__ == "__main__":
    main()
