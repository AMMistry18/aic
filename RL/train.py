"""
Train residual SAC for the last-inch insertion policy.

Usage (residual pipeline, real scene, state obs — the default path):
    MUJOCO_GL=egl pixi run python RL/train.py --scene --port-type sfp --steps 500000

This script:
    1. Builds the MuJoCo env(s) as a SubprocVecEnv. With --scene (recommended):
       RL/scene_env.py on the real exported scene, cartesian_residual action
       mode + scripted base advance, privileged flat-state obs, RSI-mix starts
       (~1 inch out + in-port reference-state inits). Without --scene: the
       legacy procedural env (RL/env.py, legacy reward).
    2. Trains SAC (stable-baselines3) with the redesigned residual-insertion
       reward in RL/reward.py (potential-based aligned approach, predicted-miss
       penalty, >20 N force cost, smoothness terms).
    3. Checkpoints a single canonical resume point (model.zip + replay_buffer)
       so a killed run continues instead of starting over — re-run the SAME
       command and it auto-resumes.
    4. Exports a TorchScript policy for deployment.

SAC scaling defaults (retuned 2026-07-03 for exploration + adaptation speed):
    batch_size=1024, num_envs=16, train_freq=1, gradient_steps=4
    → UTD=0.25 (4096 samples consumed per vec step — same data throughput as
    the old batch-4096/grad-2 config, but 4 distinct smaller updates: noisier
    gradients + faster policy adaptation). Do NOT use gradient_steps=-1
    (UTD=1 → critic overfit + ~16× slower). target_entropy defaults to -3.0
    (SAC's auto = -6 collapses the exploration noise early on a 6-D residual
    action). With state obs the replay buffer is tiny — raise --buffer-size
    (e.g. 300k) for image-free runs.
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
    p = argparse.ArgumentParser(
        description="Train residual SAC for last-inch insertion "
                    "(scripted base advance + Cartesian residual, RSI-mix starts)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=500_000,
                   help="total env steps (default 500k; ~45-90 min on a 5090)")
    p.add_argument("--port-type", choices=["sc", "sfp"], default="sc")
    p.add_argument("--seed", type=int, default=42)
    # ---- SAC hyperparameters (retuned: batch 1024 / UTD 0.25 / target-H -3) ----
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=1024,
                   help="SAC minibatch. 1024 x 4 grad steps consumes the same "
                        "4096 samples/vec-step as the old batch-4096 x 2, but "
                        "adapts the policy 2x more often with noisier gradients "
                        "(better exploration)")
    p.add_argument("--buffer-size", type=int, default=50_000,
                   help="replay transitions. RAM = buffer * channels*H*W * 2 "
                        "(obs + next_obs, uint8): 256^2x9 = 1.18 MB/transition "
                        "(50k = 59 GB!), 128^2x9 = 295 KB (50k = 14.7 GB). "
                        "Size to fit RAM — the old 500k default needed 590 GB.")
    p.add_argument("--warmup-steps", type=int, default=10_000,
                   help="learning_starts (random-action steps before training). "
                        "Curriculum level-0 starts are seated, so long random "
                        "warmups mostly log 1-step force aborts — keep it short.")
    p.add_argument("--tau", type=float, default=0.005,
                   help="target-net Polyak rate (SAC standard; the old 0.01 "
                        "compensated for the lower UTD of the batch-4096 config)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=str, default="auto",
                   help="SAC entropy coef ('auto', 'auto_0.1', or a float)")
    p.add_argument("--target-entropy", type=str, default="-3.0",
                   help="target policy entropy for auto alpha. SAC's 'auto' is "
                        "-dim(A)=-6, which drives alpha (exploration noise) to "
                        "near-zero early on this task; -3.0 keeps exploring. "
                        "Pass 'auto' to restore the SB3 default.")
    p.add_argument("--success-mix", type=float, default=0.25,
                   help="fraction of every SAC minibatch drawn from a side "
                        "buffer of SUCCESSFUL-episode transitions (SIL/DDPGfD-"
                        "style exploitation; keeps wins alive after the main "
                        "ring overwrites them). 0 = plain uniform replay.")
    p.add_argument("--success-buffer-size", type=int, default=4000,
                   help="capacity (transitions) of the success side buffer "
                        "(~295 KB each at 128^2x9 obs -> 4000 = ~1.2 GB)")
    p.add_argument("--train-freq", type=int, default=1,
                   help="vec steps between training phases (1 = every vec step)")
    p.add_argument("--gradient-steps", type=int, default=4,
                   help="grad steps per training phase (UTD = grad_steps/num_envs "
                        "= 0.25 at defaults). NEVER -1 (UTD=1 -> critic overfit).")
    p.add_argument("--num-envs", type=int, default=16,
                   help="SubprocVecEnv workers (= physical cores). 1 = DummyVecEnv.")
    p.add_argument("--max-episode-steps", type=int, default=300,
                   help="episode cap. Median success takes ~18 steps; 600-step "
                        "wandering timeouts were 30x that in wasted env budget.")
    # ---- checkpointing / resume ----
    p.add_argument("--ckpt-every", type=int, default=20_000,
                   help="env steps between model.zip checkpoints")
    p.add_argument("--buffer-every", type=int, default=100_000,
                   help="env steps between replay_buffer.pkl checkpoints (big). "
                        "0 to never save the buffer.")
    p.add_argument("--resume", action="store_true",
                   help="force-resume from <out>/model.zip. NOTE: an incomplete "
                        "run auto-resumes even without this flag.")
    p.add_argument("--no-resume-replay", action="store_true",
                   help="when resuming, load model.zip but skip replay_buffer.pkl "
                        "and re-warm under the current reward. Useful after reward "
                        "shaping changes or when the buffer came from a bad run.")
    p.add_argument("--actor-only-resume", action="store_true",
                   help="when resuming, copy only the actor from model.zip into a "
                        "fresh SAC model, resetting critic/target critic/optimizers "
                        "and rebuilding replay under the current reward.")
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
                   help="[--scene, images only] wrist-cam obs resolution (HxW, square)")
    p.add_argument("--no-state-obs", dest="state_obs", action="store_false",
                   default=True,
                   help="[--scene] use wrist-camera image observations instead "
                        "of the default privileged flat-state obs (state obs "
                        "trains ~10x faster and matches the perception-estimated "
                        "state available to the deployed LastInchInsert hook)")
    # ---- action mode (default: the residual pipeline) ----
    p.add_argument("--action-mode", choices=["joint_residual", "cartesian_residual"],
                   default="cartesian_residual",
                   help="[--scene] 'cartesian_residual' (default): SAC action is "
                        "a small Cartesian TCP residual in the port frame, "
                        "tracked by a Jacobian-transpose Cartesian impedance "
                        "controller that mirrors aic_controller's "
                        "CartesianImpedanceAction. 'joint_residual': legacy 6-D "
                        "joint residual on the joint PD target.")
    p.add_argument("--cart-action-dims", type=int, choices=[5, 6], default=6,
                   help="[cartesian_residual] 6 = [dx,dy,dz,droll,dpitch,dyaw]; "
                        "5 drops dyaw (no correction about the insertion axis)")
    p.add_argument("--cart-trans-scale-mm", type=float, default=1.0,
                   help="[cartesian_residual] translation residual per policy "
                        "step at |action|=1, in mm")
    p.add_argument("--cart-rot-scale-deg", type=float, default=1.0,
                   help="[cartesian_residual] rotation residual per policy "
                        "step at |action|=1, in degrees")
    p.add_argument("--no-base-script", dest="base_script", action="store_false",
                   default=True,
                   help="[cartesian_residual] disable the scripted base advance "
                        "(default ON: the base target pose slowly pushes along "
                        "the port insertion axis, clamped at the seated goal; "
                        "the SAC residual rides on top with a tighter envelope, "
                        "learning the correction, not the transport).")
    p.add_argument("--base-script-step-mm", type=float, default=0.5,
                   help="[--base-script] base target advance per policy step, mm")
    # ---- start-state distribution (fixed; replaces the level curriculum) ----
    p.add_argument("--reset-mode", choices=["mix", "random", "near_goal"],
                   default="mix",
                   help="'mix' = RSI-mix (default), 'random' = free 1-inch "
                        "starts only, 'near_goal' = RSI in-port starts only")
    p.add_argument("--rsi-frac", type=float, default=0.30,
                   help="fraction of episodes starting INSIDE the port "
                        "(reference-state initialization)")
    p.add_argument("--start-gap-in", type=float, default=1.0,
                   help="mean start distance outside the entrance plane, inches")
    p.add_argument("--start-gap-jitter-in", type=float, default=0.5,
                   help="+- axial start-distance jitter, inches")
    p.add_argument("--start-lat-jitter-in", type=float, default=0.5,
                   help="+- lateral start jitter per port axis, inches")
    p.add_argument("--no-start-curriculum", dest="start_curriculum",
                   action="store_false", default=True,
                   help="[--scene] disable the lightweight start distribution ramp")
    p.add_argument("--start-curriculum-window", type=int, default=50,
                   help="[--scene] completed episodes per reset-difficulty window")
    p.add_argument("--start-curriculum-advance-threshold", type=float, default=0.45,
                   help="[--scene] score-success rate needed to move starts back")
    p.add_argument("--start-curriculum-retreat-threshold", type=float, default=0.10,
                   help="[--scene] low success rate that eases starts slightly")
    p.add_argument("--start-curriculum-step", type=float, default=0.10,
                   help="[--scene] reset-difficulty level step in [0,1]")
    # ---- W&B ----
    p.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-log-every", type=int, default=1_000)
    p.add_argument("--wandb-success-window", type=int, default=100)
    p.add_argument("--wandb-eval-every", type=int, default=10_000,
                   help="[--scene] run a no-video W&B score eval every N env steps "
                        "(0 = off; uses --wandb-video-level)")
    p.add_argument("--wandb-eval-steps", type=int, default=200,
                   help="[--scene] max steps for no-video W&B score eval")
    p.add_argument("--wandb-eval-episodes", type=int, default=16,
                   help="[--scene] episodes averaged per W&B score eval")
    p.add_argument("--wandb-video-every", type=int, default=0,
                   help="DEPRECATED (episode-based gating fired ~once with vec "
                        "envs). Any value > 0 just enables the step-based "
                        "recorder below.")
    p.add_argument("--wandb-video-every-steps", type=int, default=5_000,
                   help="log a W&B rollout video every N env steps (0 = off)")
    p.add_argument("--wandb-video-episodes", type=int, default=2,
                   help="eval episodes stitched into each W&B video clip")
    p.add_argument("--wandb-video-steps", type=int, default=200,
                   help="max eval steps per episode in a W&B rollout video")
    p.add_argument("--wandb-video-fps", type=int, default=20)
    p.add_argument("--wandb-video-level", type=float, default=-1.0,
                   help="DEPRECATED (curriculum levels are gone): videos always "
                        "use the training reset mode. Kept for CLI compat.")
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

    if args.action_mode != "joint_residual" and not args.scene:
        raise SystemExit("--action-mode cartesian_residual requires --scene "
                         "(pass --scene: the residual pipeline trains on the "
                         "real exported scene)")
    if args.base_script and args.action_mode != "cartesian_residual":
        args.base_script = False   # base script only exists for cartesian mode

    if args.out is None:
        run_name = f"residual_sac_{args.port_type}_{args.reset_mode}"
        if args.action_mode == "cartesian_residual":
            run_name += "_cart_basescript" if args.base_script else "_cart"
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
    cfg_dict.pop("no_resume_replay", None)
    cfg_dict.pop("actor_only_resume", None)
    if args.action_mode == "joint_residual":
        # keep cache keys byte-identical to pre-cartesian-mode runs so existing
        # checkpoints still auto-resume (the new keys only hash when used)
        for k in ("action_mode", "cart_action_dims", "cart_trans_scale_mm",
                  "cart_rot_scale_deg", "base_script", "base_script_step_mm"):
            cfg_dict.pop(k, None)
    cache_key = make_cache_key(cfg_dict)
    print(f"[train] cache_key={cache_key}", flush=True)

    completed_marker = args.out / "COMPLETED"
    model_zip = args.out / "model.zip"
    rb_pkl = args.out / "replay_buffer.pkl"
    key_file = args.out / "cache_key.txt"
    start_level_file = args.out / "start_curriculum_level.txt"

    prior_key = key_file.read_text().strip() if key_file.exists() else None
    is_completed = completed_marker.exists() and prior_key == cache_key

    if args.force:
        # wipe resumable state so we truly start fresh
        for f in (completed_marker, model_zip, rb_pkl, start_level_file):
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

    # procedural/recorded envs keep the legacy reward; the scene env uses the
    # redesigned residual-insertion reward (RL/reward.py) below
    env_cfg = EnvConfig()
    if args.max_episode_steps > 0:
        env_cfg = dataclasses.replace(
            env_cfg, term=dataclasses.replace(env_cfg.term,
                                              max_steps=int(args.max_episode_steps)))
    n_envs = int(args.num_envs)

    # ------------------------------------------------------------------ #
    # env factory
    # ------------------------------------------------------------------ #
    use_recorded = bool(args.recorded)
    use_scene = bool(args.scene)
    make_wandb_video_env = None
    if use_scene:
        from RL.reward import RewardConfig as SceneRewardConfig
        from RL.scene_env import SceneInsertEnv, SceneEnvConfig

        inch = 0.0254
        scene_kwargs = dict(
            image_h=args.image_size, image_w=args.image_size,
            include_images=not args.state_obs,
            privileged_obs=bool(args.state_obs),
            reward=SceneRewardConfig(),
            max_episode_steps=env_cfg.term.max_steps,
            action_mode=args.action_mode,
            cart_action_dims=args.cart_action_dims,
            cart_trans_scale_m=args.cart_trans_scale_mm * 1e-3,
            cart_rot_scale_rad=float(np.radians(args.cart_rot_scale_deg)),
            base_script_enabled=bool(args.base_script),
            base_script_step_m=args.base_script_step_mm * 1e-3,
            rsi_frac=float(args.rsi_frac),
            start_gap_m=float(args.start_gap_in) * inch,
            start_gap_jitter_m=float(args.start_gap_jitter_in) * inch,
            start_lat_jitter_m=float(args.start_lat_jitter_in) * inch,
            start_curriculum_enabled=bool(args.start_curriculum),
            start_curriculum_level_file=str(args.out / "start_curriculum_level.txt"),
        )

        def _make_env(rank: int = 0):
            def _thunk():
                e = SceneInsertEnv(SceneEnvConfig(**scene_kwargs))
                e.set_reset_mode(args.reset_mode)
                e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
                return Monitor(e)
            return _thunk

        def _make_scene_video_env():
            holder = {}

            def _thunk():
                # videos need camera rendering regardless of the policy obs
                e = SceneInsertEnv(SceneEnvConfig(**scene_kwargs))
                e.set_reset_mode(args.reset_mode)
                holder["render_env"] = e
                return Monitor(TimeLimit(e, max_episode_steps=env_cfg.term.max_steps))

            venv = DummyVecEnv([_thunk])
            if not args.state_obs:
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
                # legacy env: 'mix' maps onto its 'curriculum' reset mode
                e.set_reset_mode(
                    "curriculum" if args.reset_mode == "mix" else args.reset_mode)
                e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
                return Monitor(e)
            return _thunk
    else:
        def _make_env(rank: int = 0):
            def _thunk():
                e = LastInchInsertEnv(env_cfg, port_type=args.port_type)
                # legacy env: 'mix' maps onto its 'curriculum' reset mode
                e.set_reset_mode(
                    "curriculum" if args.reset_mode == "mix" else args.reset_mode)
                e = TimeLimit(e, max_episode_steps=env_cfg.term.max_steps)
                return Monitor(e)
            return _thunk

    if n_envs == 1:
        env = DummyVecEnv([_make_env(0)])
    else:
        env = SubprocVecEnv([_make_env(i) for i in range(n_envs)],
                            start_method="fork" if os.name == "posix" else "spawn")
    flat_obs = not hasattr(env.observation_space, "spaces")
    if not flat_obs:
        env = VecTransposeImage(env)   # image Dict subspace → channels-first (C,H,W)
    obs_desc = (env.observation_space.shape if flat_obs
                else list(env.observation_space.spaces))
    print(f"[train] obs_space={obs_desc} "
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

    if flat_obs:
        policy_name = "MlpPolicy"
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    else:
        policy_name = "MultiInputPolicy"
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
                self.logger.record("rollout/f_z", float(last.get("f_z", 0.0)))
                self.logger.record("rollout/f_peak", float(last.get("f_peak", 0.0)))
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

    replay_buffer_class = None
    replay_buffer_kwargs = None
    if args.success_mix > 0.0:
        from RL.success_buffer import (
            SuccessMixDictReplayBuffer, SuccessMixReplayBuffer)
        replay_buffer_class = (
            SuccessMixReplayBuffer if flat_obs else SuccessMixDictReplayBuffer)
        replay_buffer_kwargs = dict(
            success_mix=args.success_mix,
            success_buffer_size=args.success_buffer_size,
        )
        print(f"[train] success-mix replay: {args.success_mix:.0%} of each "
              f"minibatch from the success side buffer "
              f"(cap {args.success_buffer_size})", flush=True)

    def _build_model():
        return SAC(
            policy_name, env,
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
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            seed=args.seed,
            verbose=1,
            device="auto",
            tensorboard_log=str(args.out / "tb") if _has_tensorboard() else None,
        )

    if resume and model_zip.exists() and args.actor_only_resume:
        print(f"[train] actor-only resume from {model_zip}", flush=True)
        teacher = SAC.load(str(model_zip), env=env, device="auto")
        teacher_steps = int(teacher.num_timesteps)
        model = _build_model()
        model.policy.actor.load_state_dict(teacher.policy.actor.state_dict())
        model.num_timesteps = teacher_steps
        if hasattr(teacher, "_episode_num"):
            model._episode_num = int(getattr(teacher, "_episode_num", 0))
        del teacher
        model.learning_starts = model.num_timesteps + args.warmup_steps
        print("[train] copied actor only; critic/replay/optimizers are fresh "
              f"(learning_starts={model.learning_starts})", flush=True)
        remaining = max(int(args.steps) - int(model.num_timesteps), 0)
        print(f"[train] resumed actor at {model.num_timesteps} steps; "
              f"{remaining} remaining of {args.steps}", flush=True)
    elif resume and model_zip.exists():
        print(f"[train] resuming from {model_zip}", flush=True)
        model = SAC.load(str(model_zip), env=env, device="auto")
        # re-apply the CLI learning rate (in case it changed on resume)
        from stable_baselines3.common.utils import get_schedule_fn
        model.learning_rate = args.lr
        model.lr_schedule = get_schedule_fn(args.lr)
        if rb_pkl.exists() and not args.no_resume_replay:
            model.load_replay_buffer(str(rb_pkl))
            print(f"[train] loaded replay buffer "
                  f"({model.replay_buffer.size()} transitions)", flush=True)
        else:
            # no buffer to resume from → re-warm before training on empty buffer
            model.learning_starts = model.num_timesteps + args.warmup_steps
            reason = "--no-resume-replay" if args.no_resume_replay else "no replay_buffer.pkl"
            print(f"[train] {reason} — re-warming "
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
        MetricsLogger, VideoRecorder, CheckpointManager,
        ProgressPrinter, StartCurriculumScheduler, WandbLogger,
        WandbVideoRecorder, WandbScoreEvaluator)

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
    if use_scene and args.start_curriculum:
        callbacks.append(StartCurriculumScheduler(
            level_path=args.out / "start_curriculum_level.txt",
            eval_window=args.start_curriculum_window,
            advance_threshold=args.start_curriculum_advance_threshold,
            retreat_threshold=args.start_curriculum_retreat_threshold,
            step_size=args.start_curriculum_step,
        ))

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
        if use_scene:
            wandb_config["reward"] = dataclasses.asdict(scene_kwargs["reward"])
        else:
            wandb_config["reward"] = dataclasses.asdict(env_cfg.reward)
            wandb_config["termination"] = dataclasses.asdict(env_cfg.term)
    except Exception:
        pass
    wandb_config.update({
        "env/port_type": args.port_type,
        "algorithm": "SAC", "batch_size": args.batch_size,
        "train_freq": args.train_freq, "gradient_steps": args.gradient_steps,
    })
    _mode_tag = ""
    if args.action_mode == "cartesian_residual":
        _mode_tag = "-cart-basescript" if args.base_script else "-cart"
    run_name = args.wandb_run_name or (
        f"{args.port_type}-sac{_mode_tag}-b{args.batch_size}-seed{args.seed}")
    wandb_cb = WandbLogger(
        run_name=run_name if wandb_enabled else None,
        config=wandb_config, log_every_steps=args.wandb_log_every,
        success_window=args.wandb_success_window, out_dir=args.out,
        entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
        project=args.wandb_project or os.environ.get("WANDB_PROJECT"),
        enabled=wandb_enabled)
    callbacks.append(wandb_cb)
    mirror_video = args.wandb_video_level < 0.0
    if wandb_enabled and args.wandb_eval_every > 0 and make_wandb_video_env is not None:
        callbacks.append(WandbScoreEvaluator(
            make_eval_env=make_wandb_video_env,
            every_steps=args.wandb_eval_every,
            max_steps=args.wandb_eval_steps,
            n_episodes=args.wandb_eval_episodes,
            fixed_level=1.0 if mirror_video else None,
            deterministic=True,
            enabled=True))
    video_every_steps = int(args.wandb_video_every_steps)
    if video_every_steps <= 0 and args.wandb_video_every > 0:
        video_every_steps = 20_000   # legacy episode flag -> step-based recorder
    step_video_enabled = (
        wandb_enabled and video_every_steps > 0 and make_wandb_video_env is not None
    )
    wandb_video_recorder = None
    if step_video_enabled:
        wandb_video_recorder = WandbVideoRecorder(
            make_eval_env=make_wandb_video_env,
            every_steps=video_every_steps,
            max_steps=args.wandb_video_steps,
            episodes_per_video=args.wandb_video_episodes,
            episode_levels=(None,) if mirror_video else None,
            fps=args.wandb_video_fps,
            render_camera=args.wandb_video_camera,
            render_width=args.wandb_video_width,
            render_height=args.wandb_video_height,
            min_frames=args.wandb_video_min_frames,
            key="eval/rollout_video",
            record_on_training_end=step_video_enabled,
            enabled=True)
        callbacks.append(wandb_video_recorder)

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
            device = policy.device
            if flat_obs:
                dummy = torch.zeros(1, *env.observation_space.shape,
                                    dtype=torch.float32, device=device)
            else:
                dummy = {}
                for k, sp in env.observation_space.spaces.items():
                    dtype = torch.uint8 if "image" in k else torch.float32
                    dummy[k] = torch.zeros(1, *sp.shape, dtype=dtype, device=device)
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
