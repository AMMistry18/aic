# Isaac Sim Port Handoff

## Objective

Train the maintained last-inch SFP insertion task with GPU-vectorized Isaac
Lab environments, then export a policy for the Gazebo/AIC engine path. The old
Isaac PPO task and `PerceptionInsert` are intentionally gone and must not be
restored.

## Current Repository State

The previous cleanup commit was `dd0faae`. It removed the old manager-based
Isaac PPO package and broken policy wrappers. The current uncommitted work has
two parts:

1. MuJoCo timing/controller corrections in `RL/scene_env.py`, checkpoint source
   hashing in `RL/cache.py`, and setup/smoke command fixes.
2. A new `aic_utils/aic_isaac_sim` package implementing the Isaac port.

Nothing required by the active MuJoCo scene, meshes, robot model, reset IK, SAC
trainer, or residual trainer was deleted in the cleanup commit. The deleted
Isaac code used different timing, task geometry, rewards, and PPO.

The deleted NVIDIA integration documented Isaac Lab `2.3.2` as its tested
baseline (Isaac Sim `5.x`). Keep that pairing for first bring-up. The new asset
script uses Isaac Lab's `MjcfConverter` wrapper so importer implementation
differences stay inside the installed Lab version.

## Implemented Isaac Components

- `scripts/import_mjcf_to_usd.py`: uses Isaac Sim's maintained MJCF importer,
  strips MuJoCo-only plugin/equality declarations, preserves absolute mesh and
  texture paths, imports robot and world USD files, discovers required prims,
  and writes a fail-fast asset manifest.
- `scripts/export_reset_bank.py`: samples the exact MuJoCo reverse curriculum
  after IK and settling, writing arm states, cable root poses, and diagnostics.
- `aic_isaac_sim/envs/last_inch_env.py`: `DirectRLEnv` with 500 Hz physics,
  20 Hz actions, matching gains/limits, reset-bank sampling, GPU reward and
  terminal logic, frame tracking, plug contact force, and the tool-to-plug fixed
  joint reconstructed from the MuJoCo equality relpose.
- `aic_isaac_sim/envs/task_core.py`: pure Torch geometry/reward/termination.
- `aic_isaac_sim/agents/skrl_sac_cfg.yaml`: separate actor/twin critics and
  target critics for skrl SAC.
- `scripts/smoke_task.py` and `scripts/train_sac.py`: custom launchers that
  import the external Gym registration before environment creation.

## Exact Task Contract

- Physics: `dt=0.002 s`; action decimation 25; policy period `0.05 s`.
- Home joints: `[-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]`.
- Joint stiffness: `[100, 100, 100, 50, 50, 50]`.
- Joint damping: `[40, 40, 40, 15, 15, 15]`.
- Torque limits: `[150, 150, 150, 28, 28, 28] Nm`.
- Robot-link gravity is disabled to mirror AIC/MuJoCo bias-force compensation;
  cable and plug gravity remain enabled.
- Action: six incremental joint targets, `0.01 rad` at magnitude one, clipped to
  a `0.35 rad` envelope around the episode reset.
- Target: `sfp_port_1_link_entrance`, local +Z insertion axis.
- Seated depth: `0.0458 m`; curriculum span: `0.090 m`.
- Frontier band: `0.25`; easy replay fraction: `0.2`.
- In-port jitter: XY `0.0008 m`, yaw `0.03 rad`, tilt `0.01 rad`.
- Full jitter: XY `0.006 m`, yaw `0.12 rad`, tilt `0.04 rad`.
- Success: depth `>=0.99`, axial `<0.003 m`, lateral `<0.005 m`, axis
  `<=0.035 rad`, roll `<=0.15 rad`, over-insertion `<=0.001 m`, with contact.
- Force abort: `60 N` for three policy steps or `120 N` immediately.
- Episode: 200 policy actions / 10 seconds.

Reward mirrors `RL/reward.py`: depth progress 30; lateral position 0.35 at a
6 mm reference; max axis/roll 0.15 after a 0.05 rad free region; axial force
0.20 after 12 N; lateral force 0.20 after 3 N; action delta 0.02; success +50;
bad collision -25; force abort -25; timeout -15.

## Required GPU-Machine Validation

This development computer has no Isaac Sim, PyTorch, NVIDIA training GPU, or
generated USD, so syntax was checked but runtime physics could not be executed.
The next agent must run these gates in order:

1. Generate `assets/reset_bank.npz`; inspect diagnostic maxima and reject reset
   samples outside the MuJoCo tolerances.
2. Run the MJCF importer. Confirm every manifest prim resolves and that the USD
   stage contains one UR5e articulation, the static board/ports, the cable joint
   chain, plug/module/tip bodies, and collision meshes.
3. Run `smoke_task.py --num_envs 1` with rendering. Verify the fixed joint does
   not merge or invalidate the robot/cable articulation and the cable follows a
   reset without an impulse explosion.
4. Confirm the imported cable ball joints retain `0.2` damping. The removed
   `mujoco.elasticity.cable` plugin used twist `1e2` and bend `4e1`; there is no
   automatic PhysX equivalent. Tune PhysX joint drives/stiffness against a
   MuJoCo deflection trajectory before training.
5. Compare level 0, 0.25, 0.5, 0.75, and 1 reset snapshots. Measure tip pose,
   axis, keyed roll, penetration, contact force, and cable shape in both engines.
6. Apply identical zero and single-joint pulse actions for 100 policy steps.
   Joint and TCP motion must have matching sign, scale, and 50 ms cadence.
7. Verify `plug_contact` reports port contact during insertion. Reward/abort
   uses this baseline-free PhysX normal-contact approximation, while the policy
   observation uses the 2.3.2 articulation's 6-D incoming tool-joint wrench. If
   forces diverge, calibrate a per-reset baseline before changing thresholds.
8. Reproduce MuJoCo's penetration-excess and off-limit collision checks using
   filtered PhysX contact sensors. The current Isaac port has over-insertion,
   alignment, and force aborts but not MuJoCo's calibrated penetration curve or
   enclosure-specific off-limit termination.
9. Run 64 environments for 200 steps, then scale to 1024 while watching GPU
   memory and simulation throughput.

Items 4, 7, and 8 are release blockers for claiming exact physics parity. The
environment is suitable for bring-up before those gates, but a long training run
should wait.

## Known Design Decisions

- The policy observation is state based and excludes cameras for throughput.
  MuJoCo's image reward is disabled by default, so this matches the active
  reward. Camera assets may still import with the robot, but they are not policy
  inputs.
- The reset bank currently applies arm joint states. Cable root poses are stored
  for parity diagnostics but are not written into the cloned world because the
  imported full-world USD owns the cable internally. If fixed-joint resets are
  unstable, split the generated world into static-world and cable USD assets,
  register the cable as an Isaac `Articulation`, and apply the exported cable
  root pose during `_reset_idx`.
- The current curriculum increases globally over 500,000 environment steps.
  MuJoCo SB3 uses evaluation-driven reverse-curriculum updates. Replace the
  schedule with a success-rate gate if exact trainer behavior is required.
- SAC is used here, not PPO. The YAML follows skrl's separate Gaussian actor,
  twin deterministic critic, and target critic convention.

## Distillation And Deployment

Distillation is not inherently required to move an SAC actor between simulators
when deployment has the same 31-D observation, preprocessing, six actions, and
20 Hz interface. It is required when the Isaac teacher uses privileged simulator
state unavailable to AIC/Intrinsic Flowstate, or when deployment uses camera or
force history with a different encoder.

For a privileged-state teacher, collect synchronized trajectories containing
teacher observation/action and deployable AIC observation. Train a student with
behavior cloning on teacher actions, optionally fine-tune with DAgger, export the
student plus observation-normalizer statistics, and validate it first in MuJoCo,
then Gazebo, then on hardware with force limits. Do not try to load an skrl
checkpoint directly into the old TorchScript policy wrapper.

## Commands

```bash
pixi run python aic_utils/aic_isaac_sim/scripts/export_reset_bank.py
./isaaclab.sh -p -m pip install -e aic_utils/aic_isaac_sim
./isaaclab.sh -p aic_utils/aic_isaac_sim/scripts/import_mjcf_to_usd.py \
  --mjcf aic_utils/aic_mujoco/mjcf/scene.xml \
  --usd-dir aic_utils/aic_isaac_sim/usd
./isaaclab.sh -p aic_utils/aic_isaac_sim/scripts/smoke_task.py \
  --num_envs 1 --steps 100
./isaaclab.sh -p aic_utils/aic_isaac_sim/scripts/smoke_task.py \
  --num_envs 64 --steps 200 --headless
./isaaclab.sh -p aic_utils/aic_isaac_sim/scripts/train_sac.py \
  --num_envs 1024 --headless
```

## Verification Already Completed

- All new and modified Python files compile with `py_compile`.
- `git diff --check` passes.
- MuJoCo timing derives a 20 Hz policy rate from actual XML `dt` and 25
  substeps, and cache identity now includes task/reward source hashes.
- The generated-world importer now resolves meshes/textures relative to the
  committed source MJCF rather than its intermediate output directory.
