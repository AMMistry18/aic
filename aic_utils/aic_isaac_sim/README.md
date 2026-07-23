# AIC Isaac Sim Last-Inch Port

This is the maintained Isaac Lab port of the MuJoCo SFP last-inch task. It is
a GPU-vectorized `DirectRLEnv` with skrl SAC. It does not restore the removed
Isaac PPO stack.

The source of truth remains:

- `aic_utils/aic_mujoco/mjcf/scene.xml` for geometry and assets
- `RL/scene_env.py` for controller timing, reset distribution, observations,
  and terminal conditions
- `RL/reward.py` for reward weights

The compatibility target inherited from the NVIDIA integration is Isaac Lab
`2.3.2` with Isaac Sim `5.0/5.1`. The environment uses that release's incoming
joint-wrench API. The converter is called through Isaac Lab rather than directly
calling an importer command. Do not mix arbitrary Isaac Lab and Isaac Sim
versions; port the wrench accessor deliberately before moving to Isaac Lab 3.x.

## Implemented Contract

- 500 Hz physics/controller (`dt=0.002`)
- 20 Hz policy (25 simulation steps per action)
- UR5e joint gains `[100, 100, 100, 50, 50, 50]`
- damping `[40, 40, 40, 15, 15, 15]`
- effort limits `[150, 150, 150, 28, 28, 28]`
- gravity-disabled robot links to match AIC/MuJoCo gravity compensation, while
  cable and plug gravity remain enabled
- six incremental joint-residual actions at `0.01 rad/action`
- `0.35 rad` action envelope around each reset
- 31-D state observation: arm position/velocity, TCP pose, 6-D tool wrench,
  and previous action
- the MuJoCo reverse curriculum, transferred as a generated reset bank
- geometry-first reward and success/abort thresholds from the active MuJoCo task
- separate MJCF-to-USD import, smoke, and skrl SAC launch scripts

## Build Assets

From the repository root, first produce the reset bank in the existing MuJoCo
Pixi environment:

```bash
pixi run python aic_utils/aic_isaac_sim/scripts/export_reset_bank.py
```

Then use the Python launcher shipped with the installed Isaac Sim/Isaac Lab:

```bash
./isaaclab.sh -p -m pip install -e aic_utils/aic_isaac_sim
./isaaclab.sh -p aic_utils/aic_isaac_sim/scripts/import_mjcf_to_usd.py \
  --mjcf aic_utils/aic_mujoco/mjcf/scene.xml \
  --usd-dir aic_utils/aic_isaac_sim/usd
```

The importer writes `usd/asset_manifest.json`. Generated USD and reset-bank
files are intentionally ignored by Git. Set `AIC_ISAAC_ASSET_MANIFEST` or
`AIC_ISAAC_RESET_BANK` to use generated files outside the repository.

## Validate Before Training

Start with one environment and rendering enabled so prim and collision errors
are visible:

```bash
./isaaclab.sh -p aic_utils/aic_isaac_sim/scripts/smoke_task.py \
  --num_envs 1 --steps 100
```

Then validate cloning and GPU state paths:

```bash
./isaaclab.sh -p aic_utils/aic_isaac_sim/scripts/smoke_task.py \
  --num_envs 64 --steps 200 --headless
```

Do not start a long run until the parity gates in `STATUS.md` pass.

## Train SAC

```bash
./isaaclab.sh -p aic_utils/aic_isaac_sim/scripts/train_sac.py \
  --num_envs 1024 --headless
```

Use fewer environments if the imported cable articulation exhausts GPU memory.
The trainer config is `aic_isaac_sim/agents/skrl_sac_cfg.yaml`.

## Policy Handoff

The training policy is state based and outputs the same six joint-residual
actions as the maintained MuJoCo SB3 task. Export only after the Isaac and
MuJoCo observation order, normalization statistics, action scale, and 20 Hz
cadence match. A separate teacher/student distillation step is needed only if
the deployed AIC policy must consume a smaller or different observation set,
such as camera features instead of simulator state. Distillation does not fix
simulation-to-reality physics differences; those require parity validation and
domain randomization.
