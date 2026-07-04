"""
Train image-SAC for the last-inch insertion policy.

Usage (full run):
    MUJOCO_GL=egl pixi run python RL/train.py --port-type sc --steps 500000

This script:
    1. Builds the MuJoCo env(s) (RL/env.py) as a SubprocVecEnv.
    2. Wraps a Dict obs space with a CombinedImageState extractor (image CNN
       + state MLP fused to 256-d).
    3. Trains SAC (stable-baselines3, batch 4096) with the reward in
       RL/reward.py and a reverse curriculum.
    4. Checkpoints a single canonical resume point (model.zip + replay_buffer
       + curriculum_level.txt) so a killed run continues instead of starting
       over — re-run the SAME command and it auto-resumes.
    5. Exports a TorchScript policy for deployment.

Batch-4096 SAC scaling (see the tuning notes in the header table below):
    num_envs=16, train_freq=1 (vec step = 16 env steps), gradient_steps=2
    → UTD≈0.125 (512 samples consumed per env step, 2× SAC default). Do NOT
    use gradient_steps=-1 (that is UTD=1 at batch 4096 → critic overfit +
    ~16× slower). buffer_size=500k (~9.5 GB, uint8 image obs is mandatory).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np

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


def _read_local_wandb_info(path: Path) -> dict[str, str]:
    """Load local W&B credentials/config without echoing secrets.

    Supported keys in `wandb/info.txt`:
      API key, username, team name, project name
    Colons and equals are both accepted.
    """
    path = Path(path)
    if not path.is_absolute():
        path = _REPO / path
    if not path.exists():
        return {}

    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" in s:
            k, v = s.split(":", 1)
        elif "=" in s:
            k, v = s.split("=", 1)
        else:
            continue
        key = " ".join(k.strip().lower().replace("_", " ").split())
        raw[key] = v.strip()

    out: dict[str, str] = {}
    api_key = raw.get("api key") or raw.get("apikey") or raw.get("wandb api key")
    if api_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = api_key
        out["api_key_loaded"] = "true"

    entity = raw.get("team name") or raw.get("team") or raw.get("entity") or raw.get("username")
    if entity and not os.environ.get("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = entity
    if entity:
        out["entity"] = entity

    project = raw.get("project name") or raw.get("project")
    if project and not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = project
    if project:
        out["project"] = project
    return out


def main():
    p = argparse.ArgumentParser(description="Train image-SAC for last-inch insertion")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=500_000,
                   help="total env steps (default 500k; ~45-90 min on a 5090)")
    p.add_argument("--port-type", choices=["sc", "sfp"], default="sc")
    p.add_argument("--seed", type=int, default=42)
    # ---- SAC / batch-4096 hyperparameters ----
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--warmup-steps", type=int, default=20_000,
                   help="learning_starts (random-action steps before training)")
    p.add_argument("--tau", type=float, default=0.01,
                   help="target-net Polyak rate (raised to compensate for the "
                        "lower update-to-data ratio at batch 4096)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=str, default="auto",
                   help="SAC entropy coef ('auto', 'auto_0.1', or a float)")
    p.add_argument("--target-entropy", type=str, default="auto",
                   help="'auto' (=-dim(A)=-6) or a float")
    p.add_argument("--train-freq", type=int, default=1,
                   help="vec steps between training phases (1 = every vec step)")
    p.add_argument("--gradient-steps", type=int, default=2,
                   help="grad steps per training phase. NEVER -1 at batch 4096.")
    p.add_argument("--num-envs", type=int, default=16,
                   help="SubprocVecEnv workers (= physical cores). 1 = DummyVecEnv.")
    # ---- checkpointing / resume ----
    p.add_argument("--ckpt-every", type=int, default=20_000,
                   help="env steps between model.zip checkpoints")
    p.add_argument("--buffer-every", type=int, default=100_000,
                   help="env steps between replay_buffer.pkl checkpoints (big). "
                        "0 to never save the buffer.")
    p.add_argument("--resume", action="store_true",
                   help="force-resume from <out>/model.zip. NOTE: an incomplete "
                        "run auto-resumes even without this flag.")
    p.add_argument("--force", action="store_true",
                   help="ignore any existing checkpoint and start fresh")
    p.add_argument("--no-skip-if-cached", dest="skip_if_cached", action="store_false",
                   help="re-train even if a COMPLETED run with the same key exists")
    # ---- logging ----
    p.add_argument("--log-every", type=int, default=2_000)
    p.add_argument("--no-torchscript", action="store_true")
    p.add_argument("--video-every", type=int, default=0,
                   help="record a rollout video every N episodes (0 = off; "
                        "video is expensive during long runs)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--recorded", action="store_true")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--last-inch-start-frame", type=int, default=80)
    # ---- REAL scene (scene_env.py on the exported scene.xml with real ports) ----
    p.add_argument("--scene", action="store_true",
                   help="train on the REAL AIC scene (RL/scene_env.py: UR5e + "
                        "gripper + cable + task board + NIC/SFP/SC ports) instead "
                        "of the procedural box env. This is the deployment-faithful path.")
    p.add_argument("--image-size", type=int, default=256,
                   help="[--scene] wrist-cam obs resolution (HxW, square)")
    p.add_argument("--reward-image-res", type=int, default=0,
                   help="[--scene] image-distance reward render res (0 = native "
                        "1152x1024 center cam; e.g. 256 for faster smoke runs)")
    # ---- curriculum ----
    p.add_argument("--reset-mode", choices=["curriculum", "random", "near_goal"],
                   default="curriculum")
    p.add_argument("--curriculum-eval-window", type=int, default=100)
    p.add_argument("--curriculum-advance-threshold", type=float, default=0.6)
    p.add_argument("--curriculum-retreat-threshold", type=float, default=0.15)
    p.add_argument("--curriculum-force-abort-guard", type=float, default=0.25,
                   help="do not advance curriculum if recent force-abort rate exceeds this")
    p.add_argument("--curriculum-force-abort-retreat", type=float, default=0.50,
                   help="retreat curriculum if recent force-abort rate reaches this")
    p.add_argument("--curriculum-step", type=float, default=0.05)
    # ---- W&B ----
    p.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-log-every", type=int, default=1_000)
    p.add_argument("--wandb-success-window", type=int, default=100)
    p.add_argument("--wandb-eval-every", type=int, default=0,
                   help="[--scene] run a no-video W&B score eval every N env steps "
                        "(0 = off; uses --wandb-video-level)")
    p.add_argument("--wandb-eval-steps", type=int, default=200,
                   help="[--scene] max steps for no-video W&B score eval")
    p.add_argument("--wandb-video-every", type=int, default=0,
                   help="log a W&B rollout video every N completed episodes (0 = off)")
    p.add_argument("--wandb-video-steps", type=int, default=120,
                   help="max eval steps per W&B rollout video")
    p.add_argument("--wandb-video-fps", type=int, default=20)
    p.add_argument("--wandb-video-level", type=float, default=1.0,
                   help="[--scene] curriculum level used for W&B eval videos. "
                        "Use a negative value to mirror the training reset mode.")
    p.add_argument("--wandb-video-camera", type=str, default="center_camera",
                   help="[--scene] camera used for W&B videos")
    p.add_argument("--wandb-video-width", type=int, default=768,
                   help="[--scene] W&B video render width; independent of policy image size")
    p.add_argument("--wandb-video-height", type=int, default=688,
                   help="[--scene] W&B video render height; independent of policy image size")
    p.add_argument("--wandb-video-min-frames", type=int, default=12,
                   help="pad short W&B clips to at least this many frames")
    p.add_argument("--wandb-entity", type=str, default=None,
                   help="W&B entity/team. Defaults to WANDB_ENTITY or wandb/info.txt.")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="W&B project. Defaults to WANDB_PROJECT, wandb/info.txt, or the RL project.")
    p.add_argument("--wandb-info-file", type=Path, default=Path("wandb/info.txt"),
                   help="local W&B credential/config file; API key is never logged")
    args = p.parse_args()

    if args.out is None:
        run_name = f"residual_sac_{args.port_type}_{args.reset_mode}"
        if args.recorded:
            run_name += "_recorded"
        args.out = Path(__file__).resolve().parent / "output" / run_name
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[train] saving to {args.out.resolve()}", flush=True)

    wandb_local = _read_local_wandb_info(args.wandb_info_file)
    if args.wandb_entity is None:
        args.wandb_entity = os.environ.get("WANDB_ENTITY") or wandb_local.get("entity")
    if args.wandb_project is None:
        args.wandb_project = (
            os.environ.get("WANDB_PROJECT")
            or wandb_local.get("project")
            or "1-inch-intrinsic-policy"
        )
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if wandb_local.get("api_key_loaded") == "true":
        print("[wandb] loaded API key from local info file", flush=True)
    if args.wandb_entity or args.wandb_project:
        print(f"[wandb] target entity={args.wandb_entity or '<default>'} "
              f"project={args.wandb_project or '<default>'}", flush=True)

    # ------------------------------------------------------------------ #
    # cache / resume decision
    # ------------------------------------------------------------------ #
    from RL.cache import make_cache_key, write_config

    cfg_dict = dict(vars(args))
    cfg_dict.pop("force", None)
    cfg_dict.pop("resume", None)
    cache_key = make_cache_key(cfg_dict)
    print(f"[train] cache_key={cache_key}", flush=True)

    completed_marker = args.out / "COMPLETED"
    model_zip = args.out / "model.zip"
    rb_pkl = args.out / "replay_buffer.pkl"
    key_file = args.out / "cache_key.txt"

    prior_key = key_file.read_text().strip() if key_file.exists() else None
    is_completed = completed_marker.exists() and prior_key == cache_key

    if args.force:
        # wipe resumable state so we truly start fresh
        for f in (completed_marker, model_zip, rb_pkl):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

    if args.skip_if_cached and is_completed and not args.resume and not args.force:
        print(f"[train] CACHE HIT: COMPLETED run with cache_key={cache_key} exists.",
              flush=True)
        print("[train] nothing to do. Use --force to retrain or --no-skip-if-cached.",
              flush=True)
        return

    # auto-resume when an *incomplete* checkpoint with the SAME key exists
    resume = (args.resume or
              (model_zip.exists() and not is_completed and not args.force
               and prior_key == cache_key))
    if model_zip.exists() and prior_key != cache_key and not args.force and not args.resume:
        print(f"[train] WARNING: existing checkpoint has a different cache_key "
              f"({prior_key}); starting fresh. Use --resume to force-continue it.",
              flush=True)

    write_config(args.out, cfg_dict, cache_key)

    # ------------------------------------------------------------------ #
    # heavy imports (after sys.path fix; before CUDA touch → keep VecEnv first)
    # ------------------------------------------------------------------ #
    import torch
    import torch.nn as nn
    from stable_baselines3 import SAC
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, SubprocVecEnv, VecTransposeImage)
    from gymnasium.wrappers import TimeLimit

    from RL.env import EnvConfig, LastInchInsertEnv

    env_cfg = EnvConfig()
    n_envs = int(args.num_envs)

    # ------------------------------------------------------------------ #
    # env factory
    # ------------------------------------------------------------------ #
    use_recorded = bool(args.recorded)
    use_scene = bool(args.scene)
    make_wandb_video_env = None
    if use_scene:
        from RL.scene_env import SceneInsertEnv, SceneEnvConfig

        def _make_env(rank: int = 0):
            def _thunk():
                e = SceneInsertEnv(SceneEnvConfig(
                    image_h=args.image_size, image_w=args.image_size,
                    reward_image_res=args.reward_image_res,
                    max_episode_steps=env_cfg.term.max_steps))
                e.set_reset_mode(args.reset_mode)
                if args.reset_mode == "curriculum":
                    e.set_level_file(str(args.out / "curriculum_level.txt"))
                e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
                return Monitor(e)
            return _thunk

        def _make_scene_video_env():
            holder = {}

            def _thunk():
                e = SceneInsertEnv(SceneEnvConfig(
                    image_h=args.image_size, image_w=args.image_size,
                    reward_image_res=args.reward_image_res,
                    max_episode_steps=env_cfg.term.max_steps))
                if args.wandb_video_level >= 0.0:
                    e.set_reset_mode("curriculum")
                    e.set_curriculum_level(float(args.wandb_video_level))
                else:
                    e.set_reset_mode(args.reset_mode)
                if args.reset_mode == "curriculum" and args.wandb_video_level < 0.0:
                    e.set_level_file(str(args.out / "curriculum_level.txt"))
                holder["render_env"] = e
                return Monitor(TimeLimit(e, max_episode_steps=env_cfg.term.max_steps))

            venv = DummyVecEnv([_thunk])
            venv = VecTransposeImage(venv)
            return venv, holder["render_env"]

        make_wandb_video_env = _make_scene_video_env
    elif use_recorded:
        from RL.recorded_env import (
            RecordedEnvConfig, RecordedRolloutEnv, discover_dataset)
        if args.dataset_dir is None or not Path(args.dataset_dir).exists():
            raise SystemExit(
                f"--recorded requires --dataset-dir (got {args.dataset_dir})")
        paths = discover_dataset(str(args.dataset_dir), port_type=args.port_type)
        if not paths:
            raise SystemExit(f"no rollouts under {args.dataset_dir} "
                             f"for port_type={args.port_type!r}")
        print(f"[train] recorded env: {len(paths)} rollouts", flush=True)
        rec_cfg = RecordedEnvConfig(
            obs=env_cfg.obs, reward=env_cfg.reward, term=env_cfg.term,
            port_type=args.port_type,
            last_inch_start_frame=int(args.last_inch_start_frame))

        def _make_env(rank: int = 0):
            def _thunk():
                shard = paths[rank::n_envs]
                e = RecordedRolloutEnv(shard, rec_cfg)
                e.set_reset_mode(args.reset_mode)
                if args.reset_mode == "curriculum":
                    e.set_level_file(str(args.out / "curriculum_level.txt"))
                e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
                return Monitor(e)
            return _thunk
    else:
        def _make_env(rank: int = 0):
            def _thunk():
                e = LastInchInsertEnv(env_cfg, port_type=args.port_type)
                e.set_reset_mode(args.reset_mode)
                if args.reset_mode == "curriculum":
                    e.set_level_file(str(args.out / "curriculum_level.txt"))
                e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
                return Monitor(e)
            return _thunk

    if n_envs == 1:
        env = DummyVecEnv([_make_env(0)])
    else:
        env = SubprocVecEnv([_make_env(i) for i in range(n_envs)],
                            start_method="fork" if os.name == "posix" else "spawn")
    env = VecTransposeImage(env)   # image Dict subspace → channels-first (C,H,W)
    print(f"[train] obs_space={list(env.observation_space.spaces)} "
          f"num_envs={n_envs} reset_mode={args.reset_mode} resume={resume}", flush=True)

    # ------------------------------------------------------------------ #
    # features extractor (image CNN + state MLP)  — channels-first
    # ------------------------------------------------------------------ #
    class SceneImageStateExtractor(BaseFeaturesExtractor):
        """CNN on wrist images + normalised MLP over policy-visible state.

        VecTransposeImage has already put the image channels-first, so the
        image tensor arrives as (B, C, H, W) — no permute here.
        """

        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            img_shape = observation_space["image"].shape   # (C, H, W)
            n_channels = img_shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_channels, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *img_shape)
                cnn_out = self.cnn(dummy).shape[1]

            # generic: every non-image obs key feeds the state MLP (works for both
            # RL/env.py and RL/scene_env.py obs schemas).
            state_keys = tuple(sorted(k for k in observation_space.spaces if k != "image"))
            state_dim = sum(int(np.prod(observation_space[k].shape)) for k in state_keys)
            self._state_keys = state_keys
            scale = []
            for key in state_keys:
                dim = int(np.prod(observation_space[key].shape))
                if key == "arm_qpos":
                    scale.append(np.full(dim, np.pi, dtype=np.float32))
                elif key == "arm_qvel":
                    scale.append(np.full(dim, 10.0, dtype=np.float32))
                elif key == "ft":
                    scale.append(np.full(dim, 50.0, dtype=np.float32))
                elif key == "tcp_pose" and dim == 7:
                    scale.append(np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                                          dtype=np.float32))
                else:
                    scale.append(np.ones(dim, dtype=np.float32))
            state_scale = np.concatenate(scale) if scale else np.ones(1, dtype=np.float32)
            self.register_buffer("_state_scale",
                                 torch.as_tensor(state_scale, dtype=torch.float32).view(1, -1))

            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
            )
            self.fuse = nn.Sequential(
                nn.Linear(cnn_out + 128, features_dim), nn.LayerNorm(features_dim), nn.ReLU(),
            )

        def forward(self, observations):
            img = observations["image"].float() / 255.0   # (B, C, H, W)
            z_img = self.cnn(img)
            state = torch.cat([
                observations[k].float().view(observations[k].shape[0], -1)
                for k in self._state_keys
            ], dim=1)
            z_state = self.state_mlp(state / self._state_scale.clamp_min(1e-6))
            return self.fuse(torch.cat([z_img, z_state], dim=1))

    policy_kwargs = dict(
        features_extractor_class=SceneImageStateExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
    )

    class ComponentRewardLogger(BaseCallback):
        def __init__(self, log_every: int = 100):
            super().__init__()
            self.log_every = log_every
            self._i = 0

        def _on_step(self) -> bool:
            self._i += 1
            if self._i % self.log_every == 0 and self.model.ep_info_buffer:
                last = self.model.ep_info_buffer[-1]
                self.logger.record("rollout/term_status", str(last.get("term_status", "")))
                self.logger.record("rollout/image_l1_norm", float(last.get("image_l1_norm", 0.0)))
                self.logger.record("rollout/f_z", float(last.get("f_z", 0.0)))
            return True

    # ------------------------------------------------------------------ #
    # model (fresh or resumed)
    # ------------------------------------------------------------------ #
    ent_coef = args.ent_coef
    try:
        ent_coef = float(ent_coef)   # allow a fixed numeric coefficient
    except ValueError:
        pass
    target_entropy = args.target_entropy
    try:
        target_entropy = float(target_entropy)
    except ValueError:
        pass

    def _build_model():
        return SAC(
            "MultiInputPolicy", env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.warmup_steps,
            tau=args.tau,
            gamma=args.gamma,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            seed=args.seed,
            verbose=1,
            device="auto",
            tensorboard_log=str(args.out / "tb") if _has_tensorboard() else None,
        )

    if resume and model_zip.exists():
        print(f"[train] resuming from {model_zip}", flush=True)
        model = SAC.load(str(model_zip), env=env, device="auto")
        # re-apply the CLI learning rate (in case it changed on resume)
        from stable_baselines3.common.utils import get_schedule_fn
        model.learning_rate = args.lr
        model.lr_schedule = get_schedule_fn(args.lr)
        if rb_pkl.exists():
            model.load_replay_buffer(str(rb_pkl))
            print(f"[train] loaded replay buffer "
                  f"({model.replay_buffer.size()} transitions)", flush=True)
        else:
            # no buffer to resume from → re-warm before training on empty buffer
            model.learning_starts = model.num_timesteps + args.warmup_steps
            print("[train] no replay_buffer.pkl — re-warming "
                  f"(learning_starts={model.learning_starts})", flush=True)
        remaining = max(int(args.steps) - int(model.num_timesteps), 0)
        print(f"[train] resumed at {model.num_timesteps} steps; "
              f"{remaining} remaining of {args.steps}", flush=True)
    else:
        model = _build_model()
        remaining = int(args.steps)

    if remaining <= 0:
        print("[train] target step count already reached; marking COMPLETED.", flush=True)
        completed_marker.write_text(f"steps={model.num_timesteps}\n")
        return

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    from RL.logging_utils import (
        MetricsLogger, VideoRecorder, CurriculumScheduler, CheckpointManager,
        ProgressPrinter, WandbLogger, WandbVideoRecorder, WandbScoreEvaluator)

    ckpt = CheckpointManager(
        args.out,
        model_every_steps=args.ckpt_every,
        buffer_every_steps=max(args.buffer_every, 1),
        save_buffer=args.buffer_every > 0,
    )
    callbacks = [
        ComponentRewardLogger(),
        MetricsLogger(args.out / "metrics.jsonl", append=resume),
        ProgressPrinter(log_every_steps=args.log_every),
        ckpt,
    ]

    # W&B enable decision
    wandb_env_off = os.environ.get("WANDB_MODE", "").strip().lower() in (
        "disabled", "false", "0")
    wandb_api_present = bool(os.environ.get("WANDB_API_KEY", "").strip())
    if wandb_env_off or args.wandb is False:
        wandb_enabled = False
    elif args.wandb is True or wandb_api_present:
        wandb_enabled = True
    else:
        wandb_enabled = False

    wandb_config = dict(cfg_dict)
    try:
        wandb_config["reward"] = dataclasses.asdict(env_cfg.reward)
        wandb_config["termination"] = dataclasses.asdict(env_cfg.term)
    except Exception:
        pass
    wandb_config.update({
        "env/port_type": args.port_type,
        "env/pos_scale": list(env_cfg.pos_scale),
        "env/rot_scale": list(env_cfg.rot_scale),
        "algorithm": "SAC", "batch_size": args.batch_size,
        "train_freq": args.train_freq, "gradient_steps": args.gradient_steps,
    })
    run_name = args.wandb_run_name or f"{args.port_type}-sac-b{args.batch_size}-seed{args.seed}"
    wandb_cb = WandbLogger(
        run_name=run_name if wandb_enabled else None,
        config=wandb_config, log_every_steps=args.wandb_log_every,
        success_window=args.wandb_success_window, out_dir=args.out,
        entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
        project=args.wandb_project or os.environ.get("WANDB_PROJECT"),
        enabled=wandb_enabled)
    callbacks.append(wandb_cb)
    if wandb_enabled and args.wandb_eval_every > 0 and make_wandb_video_env is not None:
        callbacks.append(WandbScoreEvaluator(
            make_eval_env=make_wandb_video_env,
            every_steps=args.wandb_eval_every,
            max_steps=args.wandb_eval_steps,
            deterministic=True,
            enabled=True))
    if wandb_enabled and args.wandb_video_every > 0 and make_wandb_video_env is not None:
        callbacks.append(WandbVideoRecorder(
            make_eval_env=make_wandb_video_env,
            every_n_episodes=args.wandb_video_every,
            max_steps=args.wandb_video_steps,
            fps=args.wandb_video_fps,
            render_camera=args.wandb_video_camera,
            render_width=args.wandb_video_width,
            render_height=args.wandb_video_height,
            min_frames=args.wandb_video_min_frames,
            key="eval/rollout_video",
            record_on_training_end=True,
            enabled=True))

    if args.reset_mode == "curriculum":
        callbacks.append(CurriculumScheduler(
            env=None,
            eval_window=args.curriculum_eval_window,
            advance_threshold=args.curriculum_advance_threshold,
            retreat_threshold=args.curriculum_retreat_threshold,
            force_abort_guard=args.curriculum_force_abort_guard,
            force_abort_retreat=args.curriculum_force_abort_retreat,
            step_size=args.curriculum_step,
            level_path=args.out / "curriculum_level.txt"))
    if not args.no_video and args.video_every > 0:
        callbacks.append(VideoRecorder(env, args.out / "videos",
                                       every_n_episodes=args.video_every,
                                       max_steps=env_cfg.term.max_steps))

    # ------------------------------------------------------------------ #
    # learn — with crash-safe final checkpoint on ANY exit
    # ------------------------------------------------------------------ #
    t0 = time.time()
    interrupted = False
    try:
        model.learn(total_timesteps=remaining, callback=callbacks,
                    reset_num_timesteps=not resume)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[train] interrupted — saving checkpoint before exit", flush=True)
    finally:
        # always persist a resumable checkpoint (model + buffer)
        ckpt.save_model()
        if args.buffer_every > 0:
            ckpt.save_replay_buffer()
        wandb_cb.finish()
        try:
            env.close()
        except Exception as exc:
            print(f"[train] WARN: env close failed: {exc}", flush=True)
    elapsed = time.time() - t0
    done_steps = int(model.num_timesteps)
    print(f"[train] {'stopped' if interrupted else 'done'} at {done_steps} steps "
          f"in {elapsed:.1f}s ({remaining/max(elapsed,1e-9):.0f} steps/s)", flush=True)

    if interrupted:
        print("[train] re-run the same command to auto-resume.", flush=True)
        return

    # completed the target
    completed_marker.write_text(f"steps={done_steps}\ncache_key={cache_key}\n")

    # export TorchScript for deployment
    if not args.no_torchscript:
        try:
            ts_path = args.out / "policy.pt"
            policy = model.policy
            policy.eval()
            dummy = {}
            for k, sp in env.observation_space.spaces.items():
                dtype = torch.uint8 if "image" in k else torch.float32
                dummy[k] = torch.zeros(1, *sp.shape, dtype=dtype)
            with torch.no_grad():
                traced = torch.jit.trace(policy, dummy, strict=False)
            traced.save(str(ts_path))
            print(f"[train] exported TorchScript policy to {ts_path}", flush=True)
        except Exception as exc:
            print(f"[train] WARN: TorchScript export failed: {exc}", flush=True)

    (args.out / "train_meta.txt").write_text(
        f"port_type={args.port_type}\nsteps={done_steps}\nseed={args.seed}\n"
        f"lr={args.lr}\nbatch_size={args.batch_size}\nbuffer_size={args.buffer_size}\n"
        f"warmup_steps={args.warmup_steps}\ntrain_freq={args.train_freq}\n"
        f"gradient_steps={args.gradient_steps}\nnum_envs={n_envs}\n"
        f"elapsed_s={elapsed:.1f}\ncache_key={cache_key}\n"
        f"policy_class=aic_example_policies.ros.LastInchInsert\n")
    print(f"[train] wrote {args.out / 'train_meta.txt'}", flush=True)

    if args.plot:
        try:
            from RL.logging_utils import plot_dashboard
            png = plot_dashboard(args.out / "metrics.jsonl")
            print(f"[train] dashboard: {png}", flush=True)
        except Exception as exc:
            print(f"[train] WARN: dashboard failed: {exc}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["main"]
