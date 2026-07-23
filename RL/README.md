# RL

This directory contains two SAC training tracks for the real exported AIC MuJoCo scene. Both train the welded SFP connector against `sfp_port_1_link_entrance`.

## Tracks

| Path | Purpose | Reset behavior | Output root |
| --- | --- | --- | --- |
| `RL/sb3_sac/trainer.py` | Compact SB3 SAC smoke and experimentation loop. | Linear reverse curriculum. | `RL/output/sb3_sac/` |
| `RL/residual_sac/trainer.py` | Full residual-SAC trainer with checkpoints, success replay, metrics, and W&B. | Random by default. Pass `--reset-mode curriculum` to enable adaptive reverse curriculum. | `RL/output/residual_sac/` |

Shared modules remain at the `RL/` root:

- `scene_env.py`: real MuJoCo environment and residual action controller.
- `reward.py`: reward and termination logic.
- `cache.py`: configuration and scene-aware cache keys.
- `success_buffer.py`: success-biased replay buffer for the full trainer.
- `logging_utils.py`: metrics, checkpoints, video, curriculum, and W&B callbacks.

## Commands

Run from the repository root in the configured Pixi environment:

```bash
# Compact reverse-curriculum SB3 SAC run
pixi run python RL/sb3_sac/trainer.py --timesteps 2000 --run-name smoke

# Full residual-SAC run without reverse curriculum
pixi run python RL/residual_sac/trainer.py --steps 500000

# Full residual-SAC run with adaptive reverse curriculum
pixi run python RL/residual_sac/trainer.py --steps 500000 --reset-mode curriculum
```

The real scene is the only supported environment. Historical procedural, recorded-rollout, and placeholder wrappers were removed.

The GPU-vectorized port of this same task lives in
`aic_utils/aic_isaac_sim/`. Read its `STATUS.md` before generating USD assets
or launching Isaac SAC; it records the physics-parity gates that cannot be
validated on a machine without Isaac Sim and an NVIDIA GPU.
