# RL — AIC Last-Inch Insertion (Image-SAC + Reverse Curriculum)

Residual reinforcement learning for the **final ~1 inch** of fiber-optic cable
insertion into the AIC task board. The learned policy takes over the contact-rich
seating that the hand-coded macro controller does worst.

> **STATUS (2026-07-03).** Training now runs on the **REAL AIC MuJoCo scene**
> (`RL/scene_env.py`) — UR5e + Robotiq Hand-E + welded LC/SFP plug + elastic
> cable + task board with the **actual receptacle ports** (NIC-card SFP cage +
> SC ports), re-exported from Gazebo. The older procedural box scene
> (`RL/env.py`) is a fast unit-test toy only. **Everything runs inside the
> `aic_eval-latest` distrobox container** (its ROS/Gazebo/GL match eval), using
> the repo's `pixi` env for the Python/RL stack.
>
> **CURRENT GEOMETRY:** the scene env targets the exported
> `sfp_port_1_link_entrance` frame, not the NIC-card mount origin. The seated
> target is a full-depth `0.046 m` tip pose along the port inward axis, with
> finite-difference reset IK on both SFP-tip position and plug-axis alignment,
> plus plug-port penetration rejection. Success additionally requires
> `depth_norm >= 0.97` and bounded plug-port penetration, so a near-but-not-
> bottomed or collision-tunneled smoke rollout does **not** count as inserted.
> See [§7 Geometry diagnostics](#7-geometry-diagnostics).

---

## 0. Quickstart — run in the container

The container's *system* python has no mujoco/torch/sb3; the RL stack lives in
the repo **pixi** env, which works inside the container because distrobox
bind-mounts the host home (`~/.pixi`, the repo, `/tmp`). Reliable recipe — a tiny
wrapper invoked through `distrobox enter`, capturing output on the **host** side:

```bash
# crun.sh  (activates pixi + EGL inside the container, then runs a python file)
#   export PATH="$HOME/.pixi/bin:$PATH"; export MUJOCO_GL=egl
#   cd /home/Anshul/AIC_Phase_1/aic_0/aic; exec pixi run python "$@"

distrobox enter aic_eval-latest -- bash /path/to/crun.sh RL/train.py --scene ... > run.log 2>&1
```

Notes:
- `distrobox list` → `aic_eval-latest` (image `ghcr.io/intrinsic-dev/aic/aic_eval:latest`).
  First `distrobox enter` auto-starts it.
- **Do NOT** use `distrobox enter -- python -c "..."` — inline quoting mangles and
  pixi activation dumps `declare -x` noise. Always run a **script file**.
- **GPU:** MuJoCo EGL is HW-accelerated in-container (RTX 5090, ~13.5k fps @256²)
  via `/dev/nvidia*`. Gazebo's *mesa* perception rendering hits
  `/dev/dri/renderD128: Permission denied`; only needed if you re-run Gazebo
  perception. Fix (needs your password): `sudo setfacl -m u:1001:rw /dev/dri/renderD128 /dev/dri/card1`.

---

## 1. Data flow

```
  scene.xml (real assets, with ports)
        │  loads
        ▼
  RL/scene_env.py : SceneInsertEnv ──▶ obs Dict ──▶ RL/reward.py : compute_reward ──▶ r
   (UR5e+gripper+cable+board+ports)     (image+ft+                (image L1 + force +
   IK reverse curriculum, FT sensor,     tcp_pose+…)               potential depth + xy +
   pose+force success)                                             action + terminal)
        ▲                                                                 │
        └───────── action (6,) joint residual ◀───────────────────────────┘
                                     │
             RL/train.py --scene : SAC(MultiInputPolicy, batch 4096)
             CombinedImageState (image CNN + generic state MLP)
             CurriculumScheduler + CheckpointManager (atomic, auto-resume)
                                     │
                        outputs/<run>/{model.zip, replay_buffer.pkl,
                                       curriculum_level.txt, metrics.jsonl}
```

---

## 2. Environment assets — where everything lives

All paths relative to the repo root `/home/Anshul/AIC_Phase_1/aic_0/aic/`.

### 2.1 The MuJoCo scene (what training loads)
| Path | What |
|---|---|
| `aic_utils/aic_mujoco/mjcf/scene.xml` | **Top-level scene** — `<include>`s the two below. 63 bodies, 3 wrist cams, FT sensors, cable plugin. This is what `SceneInsertEnv` loads. |
| `aic_utils/aic_mujoco/mjcf/aic_robot.xml` | UR5e + Robotiq Hand-E + wrist cameras + FT sensor + actuators. |
| `aic_utils/aic_mujoco/mjcf/aic_world.xml` | Enclosure, floor, task board, **the receptacle ports**, and the cable. |
| `aic_utils/aic_mujoco/mjcf/*.obj` (64) `*.png` (8) | Converted meshes+textures (hash-named). Must stay co-located with the XMLs. |
| `aic_utils/aic_mujoco/mjcf_backup_preports/` | The **old** port-less scene (backup from the re-export). |

### 2.2 Key bodies / sites (query by name in MuJoCo)
| Name | id | Role |
|---|---|---|
| `nic_card_mount_2::nic_card_mount_link` | 27 | **SFP insertion target** (NIC card mount on `nic_rail_2`). |
| `nic_card_link` | 28 | The NIC card itself (has the SFP cage slots). |
| `sc_port_0::sc_port_link`, `sc_port_1::sc_port_link` | — | The two SC ports (`sc_port_base_link_entrance` = SC entrance frame). |
| `sfp_tip_link` (60), `sfp_module_link` (59), `lc_plug_link` (58) | — | The **cable's** SFP plug end (the moving plug). `lc_plug_link` is welded to the gripper. |
| `gripper_tcp` (site 1) | — | Gripper TCP — IK target frame. |
| `center/left/right_camera` | — | Wrist cams, **native 1152×1024**, fovy 45. |
| `AtiForceTorqueSensor_force/_torque` | — | 6-axis wrist FT sensor. |

### 2.3 Source assets (used by the re-export, not loaded directly)
| Path | What |
|---|---|
| `aic_assets/models/NIC Card/nic_card_visual.glb` | SFP-cage NIC card mesh. |
| `aic_assets/models/SC Port/sc_port_visual.glb` | SC port mesh. |
| `aic_assets/models/SFP Mount`, `SC Mount`, `NIC Card Mount`, `LC Mount` | Rail mounts. |
| `aic_assets/models/{SFP Module, SC Plug, LC Plug}` | Cable connector meshes. |
| `aic_assets/models/Task Board Base`, `Enclosure`, `Enclosure Walls`, `Floor` | Static world. |
| `aic_assets/models/sfp_sc_cable` | The cable model (SFP+LC one end, SC other end). |
| `aic_description/urdf/task_board.urdf.xacro` | Task-board assembly (port `*_present` / `*_translation` / `*_yaw` xacro args). |
| `aic_engine/config/sfp_validation_config.yaml` | Eval board config. Trial 1: SFP target = `nic_card_mount_2` at `nic_rail_2` (translation −0.0077, yaw 0.0842); board pose (0.1445, −0.0602, yaw 3.1895). |

---

## 3. The RL contracts (what `scene_env.py` produces)

### 3.1 Observation (Dict)
| key | shape | meaning |
|---|---|---|
| `image` | `(256, 256, 9)` uint8 | 3 wrist cams (L/C/R) × RGB, channel-stacked |
| `ft` | `(6,)` f32 | raw wrist FT wrench (force xyz + torque xyz), N / N·m |
| `tcp_pose` | `(7,)` f32 | gripper TCP pose `[xyz, wxyz]` |
| `arm_qpos` | `(6,)` f32 | UR5e joint angles |
| `arm_qvel` | `(6,)` f32 | UR5e joint velocities |
| `last_action` | `(6,)` f32 | previous residual action |

### 3.2 Action
`Box(-1, 1, (6,))` → a **6-DoF UR5e joint residual** (`action * action_joint_scale`,
default 0.03 rad) added to a gravity-compensated PD hold of the reset pose.

### 3.3 Reward (`RL/reward.py:compute_reward`, weighted sum)
| term | intent |
|---|---|
| `r_image` (w=1.0) | dense image-L1 to the seated goal image, computed at **native 1152×1024** (center cam) for last-inch pixel accuracy |
| `r_depth` (w=2.0) | **potential-based** insertion-depth progress (Ng 1999); telescopes to +2 over a full seat |
| `r_force` (w=0.05) | seating-force shaping, calibrated band; positive branch gated on `depth_norm>0.3` |
| `r_xy` / `r_lateral` / `r_action` | centring nudge / side-load penalty / jerk penalty |
| `r_axis` / `r_collision` | straight-insertion penalty / excess plug-port penetration penalty |
| `r_done` | **+50 success / −25 bad_collision / −10 force_abort / −10 off_limit / 0 timeout** |

Contact force = **raw FT − baseline** (the ~−10.5 N gripped-plug weight, captured
at reset). `f_z` = force along the insertion axis; `f_xy` = lateral.

### 3.4 Success / termination
- **success** = plug tip near the seated pose (`success_axial_tol_m` 6 mm,
  `success_lateral_tol_m` 8 mm), `depth_norm >= success_depth_norm` (0.97),
  plug axis within ~3 deg of the port axis, plug-port penetration within 1 mm
  of the calibrated seated baseline, **AND**
  `|axial contact force| > success_force_n` (2 N).
- **bad_collision** = excess plug-port penetration beyond 3 mm, or a bent plug
  axis beyond ~11 deg once the plug has entered the last-inch corridor.
- **force_abort** = `|force| > force_abort_n` (safety).
- **timeout** = `max_episode_steps` (200). SB3 `TimeLimit` bootstraps Q on truncation.

### 3.5 Reverse curriculum (last-inch, relative to the port)
Level 0 = plug **seated** at the SFP entrance frame; as level→1, reset samples
retract up to `last_inch_m` (4 cm) along the port-frame insertion axis plus
port-frame lateral/yaw/tilt jitter. `CurriculumScheduler` advances the level
when success rate crosses the threshold and the recent force-abort rate stays
below the guard; after every level change it clears the window so the next
decision uses fresh episodes. The level is written to
`outputs/<run>/curriculum_level.txt` (env re-reads on reset → works across
`SubprocVecEnv` workers).

---

## 4. Training

```bash
# via crun.sh inside the container:
pixi run python RL/train.py --scene --port-type sfp --steps 500000
```
Defaults (batch-4096 Image-SAC, reasoned for a single 5090):
`--num-envs 16 --batch-size 4096 --train-freq 1 --gradient-steps 2`
(UTD≈0.125; never `-1`), `--buffer-size 500000` (uint8 image obs is mandatory),
`--warmup-steps 20000 --tau 0.01 --gamma 0.99 --ent-coef auto`,
`--image-size 256 --reward-image-res 0` (0 = native reward distance).

> **Scene env is heavy to construct/reset** (IK + settle). For big `--num-envs`,
> each `SubprocVecEnv` worker builds its own `SceneInsertEnv`; start with
> `--num-envs 8` and scale up once geometry is tuned. Use `--reward-image-res 256`
> for fast iteration (native 1152×1024 is slower).

**Smoke** (fast, 1 env):
```bash
pixi run python RL/train.py --scene --out RL/output/smoke \
  --steps 2000 --num-envs 1 --batch-size 128 --warmup-steps 50 \
  --image-size 128 --reward-image-res 128 --buffer-size 5000 \
  --gradient-steps 1 --no-video --no-torchscript --reset-mode curriculum
```

### 4.1 Crash-safe caching / auto-resume
`train.py` writes a single canonical resume point so a killed run never restarts:
- `<out>/model.zip` every `--ckpt-every` (20k) steps (atomic),
- `<out>/replay_buffer.pkl` every `--buffer-every` (100k) steps,
- `<out>/curriculum_level.txt` on every curriculum change,
- a final checkpoint on Ctrl-C / exception (via `finally`).

**Re-run the exact same command** → an incomplete run **auto-resumes** (loads
model+buffer, continues with `reset_num_timesteps=False`, appends metrics). A
finished run writes `<out>/COMPLETED` and is cache-skipped. `--force` = fresh;
`--resume` = force-continue.

---

## 5. Code map

| File | Role |
|---|---|
| `RL/scene_env.py` | **PRIMARY env** — `SceneInsertEnv` on the real `scene.xml` (ports, IK reverse curriculum, FT, reward, pose+force success). Edit `SceneEnvConfig` to tune. |
| `RL/reward.py` | Framework-agnostic reward (numpy only). `compute_reward`, `RewardConfig`, `check_termination`. |
| `RL/train.py` | SB3 SAC runner. `--scene` selects the real env. `CombinedImageState` extractor (image CNN + **generic** state MLP over all non-image keys). Curriculum + atomic checkpoint/auto-resume. |
| `RL/logging_utils.py` | `MetricsLogger`, `VideoRecorder`, `CurriculumScheduler`, `CheckpointManager`, `ProgressPrinter`, `WandbLogger`, `plot_dashboard`. |
| `RL/env.py` | **Legacy toy** — procedural box plug+port. Fast unit tests only; NOT deployment-faithful. |
| `RL/observation.py` | Obs builders/schema for `env.py` + the ROS deploy path. |
| `RL/cache.py`, `RL/load_model.py`, `RL/models.toml` | Run cache-key, model registry/resolver. |
| `RL/mujoco_env.py` | Older stub loader for `scene.xml` (superseded by `scene_env.py`). |
| `RL/tests/*.png` | Rendered verification images of the scene + ports. |

---

## 6. How the ported scene was re-exported (to change ports / board)

The Gazebo→MuJoCo export drops receptacle ports by default (the world-save plugin
`WorldSdfGeneratorPlugin` fires at ~8 s, *before* the `aic_engine` spawns per-trial
ports; the launch defaults all port rails to `present=false`). To get ports you
**bake them into the static task board** and re-export. All in-container:

```bash
source /opt/ros/kilted/setup.bash && source /ws_aic/install/setup.bash
# 1. Bake ports into the task board (values from sfp_validation_config trial_1)
xacro <install>/share/aic_description/urdf/task_board.urdf.xacro \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=-0.0077 nic_card_mount_2_yaw:=0.0842 \
  sc_port_0_present:=true sc_port_0_translation:=-0.0391 sc_port_0_yaw:=-0.0534 \
  sc_port_1_present:=true sc_port_1_translation:=0.0378  sc_port_1_yaw:=0.1078 \
  > /tmp/task_board_ports.urdf
# 2. Launch Gazebo → WorldSdfGeneratorPlugin auto-saves /tmp/aic.sdf (poll then kill)
ros2 launch aic_bringup aic_gz_bringup.launch.py spawn_task_board:=true \
  task_board_description_file:=/tmp/task_board_ports.urdf \
  task_board_x:=0.1445 task_board_y:=-0.0602 task_board_yaw:=3.1895 \
  spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
  ground_truth:=true gazebo_gui:=false launch_rviz:=false start_aic_engine:=true
# 3. Convert (mjcf_venv has mujoco/trimesh/pycollada; sdf2mjcf_aic aliases sdformat15/gz.math8)
/ws_aic/mjcf_venv/bin/python aic_utils/aic_mujoco/scripts/sdf2mjcf_aic.py /tmp/aic.sdf /tmp/out/aic_world.xml
# 4. Split + cable plugin (pixi mujoco 3.5)
pixi run python aic_utils/aic_mujoco/scripts/add_cable_plugin.py \
  --input /tmp/out/aic_world.xml --output /tmp/out/aic_world.xml \
  --robot_output /tmp/out/aic_robot.xml --scene_output /tmp/out/scene.xml
# 5. Copy /tmp/out/* into aic_utils/aic_mujoco/mjcf/
```
No SDF sed-patching was needed (mesh URIs came out as valid absolute `file:///…`).

---

## 7. Geometry diagnostics

The prior blocker was the seated goal using
`nic_card_mount_2::nic_card_mount_link`, which is inside the solid NIC card. The
env now uses `sfp_port_1_link_entrance` plus `seated_depth_m=0.046` along that
body's local +Z insertion axis. Reset IK is two-stage: TCP pose/position seed,
then finite-difference SFP-tip **and plug-axis** correction through the welded
plug/cable transform. The clean seated reset is also used to calibrate the
exported cage's unavoidable soft-contact overlap; reward/termination uses
`plug_port_penetration_excess_m` beyond that baseline. Diagnostics include
`plug_axis_error_deg`, `plug_port_penetration_m`,
`plug_port_penetration_excess_m`, total contact count, and worst contact pair.

Run:

```bash
MUJOCO_GL=egl AIC_REWARD_RES=128 pixi run python RL/scene_env.py
```

The diagnostic prints entrance/seated pose, insertion axis, reset tip error,
plug-axis angle, plug-port penetration/excess penetration, axial/lateral errors,
depth, contact force, and termination for levels 0/0.5/1.
Set `AIC_WRITE_TEST_IMAGES=1` to refresh only `RL/tests/new_*` verification
renders.

### 7.1 W&B
`wandb` is installed in the pixi env. `RL/train.py` auto-loads `wandb/info.txt`
when present, sets `WANDB_API_KEY` for the process, and resolves the target as:
CLI `--wandb-entity/--wandb-project` → env `WANDB_ENTITY/WANDB_PROJECT` →
`wandb/info.txt`. The API key is not written to run configs.

---

## 8. Measured perf (this box: i7-13700K, RTX 5090, in-container)
| stage | cost | note |
|---|---|---|
| MuJoCo EGL render (256²) | ~0.07 ms | HW-accelerated on the 5090 |
| `scene_env` reset | ~0.1–0.9 s | IK + settle (dominant reset cost) |
| SAC grad step (batch 4096) | ~15–30 ms | the training wall (GPU) |

Expect env throughput to jump once geometry is tuned (episodes stop aborting at
step 1, so far fewer heavy resets).
