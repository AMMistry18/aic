# RL — AIC Last-Inch Policy

This folder holds everything we're adding on top of the qualifying-phase
`aic_example_policies` for **residual-RL on the last inch** of cable
insertion.

## Layout

| File / dir | Status | Purpose |
|---|---|---|
| `models.toml` | ✅ populated | Model registry. Switch active model with `AIC_MODEL=<name>` env var. |
| `load_model.py` | ✅ done | Resolves the active model + its weight path. CLI helpers below. |
| `REWARD_SPEC.md` | ✅ done | Reward function design + hyperparameters. |
| `reward.py` | ✅ done | Framework-agnostic reward function (SAC-safe, no torch at import). |
| `observation.py` | 🔲 TODO | ROS `Observation` → policy-state dict per `REWARD_SPEC.md` §4. |
| `env.py` | 🔲 TODO | gym wrapper around the AIC sim that consumes `(observation, reward)`. |
| `train.py` | 🔲 TODO | Stable-Baselines3 + custom feature extractor runner; uses `reward.py` and `env.py`. |
| `LastInchInsert.py` | 🔲 TODO | New `Policy` subclass for Phase-1 deployment. |
| `outputs/<run-name>/` | will appear | Checkpoints + logs from `train.py`. |

## Daily commands

```bash
# 1. List all registered models (mark the active one with *)
cd /home/Anshul/AIC_Phase_1/aic_0/aic
pixi run python RL/load_model.py --list

# 2. Print the active model's policy class (for aic_model --policy:= arg)
pixi run python RL/load_model.py --class
#   → outputs e.g.: aic_example_policies.ros.PerceptionInsert

# 3. Switch to a different model
export AIC_MODEL=residual_sac_v0
pixi run python RL/load_model.py --name        # → residual_sac_v0
pixi run python RL/load_model.py --class       # → aic_example_policies.ros.LastInchInsert

# 4. Show full metadata for one model
pixi run python RL/load_model.py --show perception_insert_baseline

# 5. Check the weights path
pixi run python RL/load_model.py --weights
#   → /home/Anshul/AIC_Phase_1/aic_0/aic/outputs/residual_sac_v0/  (or "" if not trained yet)
```

## Running a model against the live sim

```bash
# Terminal A — keep the distrobox container running
/home/Anshul/.local/bin/distrobox list                    # aic_eval-latest should be "Up"
/home/Anshul/.local/bin/distrobox enter aic_eval-latest -- bash -c \
  '/entrypoint.sh ground_truth:=false start_aic_engine:=false gazebo_gui:=false launch_rviz:=false'
# (or just leave the one already running)

# Terminal B — run any policy from your pixi env
cd /home/Anshul/AIC_Phase_1/aic_0/aic
export PATH="$HOME/.pixi/bin:$PATH"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/127.0.0.1:7447"];transport/shared_memory/enabled=false'
export AIC_MODEL=perception_insert_baseline       # or residual_sac_v0 once trained
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=$(pixi run python RL/load_model.py --class)
```

## Adding a new model checkpoint

1. Train (after `env.py` and `train.py` exist):
   ```bash
   pixi run python RL/train.py --out outputs/residual_sac_v1 --steps 200_000
   ```
2. Edit `models.toml`, add a new `[models.residual_sac_v1]` block. Copy the
   schema from `residual_sac_v0` and update the `weight_path` to point at
   the new directory. Update `last_score` after Phase-1 submissions.
3. Sanity-check:
   ```bash
   pixi run python RL/load_model.py --show residual_sac_v1
   ```

## Notes on the registry

- `[defaults].active` is what runs when `AIC_MODEL` is unset.
- `[defaults].fallback_to_baseline_on_missing_weights` — when `true`, a
  model whose `weight_path` doesn't exist on disk falls back to the
  baseline's policy class. Useful for Phase-1 submissions: if the bundle
  zip got corrupted, the evaluator still runs the hand-coded controller.
- `tomllib` is the Python 3.11+ standard library. Pixi ships Python 3.12,
  so no extra dep is needed.
