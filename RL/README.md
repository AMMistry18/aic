# RL — AIC Last-Inch Insertion (Image-SAC + Reverse Curriculum)

> **UPDATE (2026-07-03, curriculum/reward/video rework).** Curriculum span now
> exits the cage (`last_inch_m` 0.072: seated → entrance → 26 mm approach) with
> frontier-band sampling + two-phase x/y/z jitter (§3.5); reward is now simple
> geometry-first shaping with image distance default-off (§3.3); W&B videos are
> step-scheduled and repeat reliably (§7.1); SAC defaults retuned to batch 1024
> / UTD 0.25 / target-entropy −3 (§4).

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
             RL/train.py --scene : SAC(MultiInputPolicy, batch 1024, UTD 0.25)
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

### 3.3 Reward (`RL/reward.py:compute_reward`, simple geometry-first shaping)
| term | intent |
|---|---|
| `r_depth` (w=20) | main dense term: `depth_norm - prev_depth_norm`; positive for inserting deeper, negative for backing out, zero for just sitting still |
| `r_done` | sparse task result: `+50` success, `-10` bad collision / force abort, `0` timeout |
| `r_xy` / `r_axis` | tiny per-step alignment costs for being off-center or angled |
| `r_force` / `r_lateral` / `r_collision` | safety costs only; there is no positive force reward to farm |
| `r_action` | tiny smoothness cost on residual action changes |
| `r_events` | one-shot non-terminal safety costs for off-limit contact or sustained high force |
| `r_image` | optional image-L1 term; default weight is `0`, so the reward renderer is skipped unless `--image-reward-weight > 0` |

Contact force = **raw FT − baseline** (the ~−10.5 N gripped-plug weight, captured
at reset). `f_z` = force along the insertion axis; `f_xy` = lateral.

**Penetration baseline is depth-dependent** (2026-07-04): the exported cage's
soft-contact overlap varies with depth (~5.4 mm mid-cage vs 3.4 mm seated), so
`SceneInsertEnv._calibrate_pen_baseline()` sweeps 9 jitter-free resets along
the axis at construction and `plug_port_penetration_excess_m` charges only the
excess over that curve (+0.5 mm margin). Before this, a single seated baseline
made the collision term tax the whole mid-cage traversal ~−1.7/step (and up to
−12/step under the pre-2026-07-04 weights) for merely existing there.

The published-style score diagnostics are still logged separately from
`SceneInsertEnv._score_diag`; they are not mirrored term-by-term in the training
reward anymore. Validation: `RL/scripts/validate_reward_geometry.py` (22 unit +
in-sim probes, all PASS 2026-07-04: depth telescopes exactly, nothing farmable,
scripted outside-in insertion nets +65.5 with success, sit-still ≈ −0.03/step).

### 3.4 Success / termination
- **success** = plug tip near the seated pose (`success_axial_tol_m` 6 mm,
  `success_lateral_tol_m` 8 mm), `depth_norm >= success_depth_norm` (0.97),
  plug axis within ~3 deg of the port axis, plug-port penetration within 1 mm
  of the calibrated seated baseline, **AND**
  `|axial contact force| > success_force_n` (2 N).
- **bad_collision** = excess plug-port penetration beyond 3 mm, or a bent plug
  axis beyond ~11 deg once the plug has entered the last-inch corridor.
- **force_abort** = contact force over `force_abort_n` (60 N) for
  `force_abort_dwell_steps` (3) **consecutive** steps, or instantly past
  `force_abort_hard_n` (120 N). Single-step PD contact transients no longer end
  episodes — instantaneous 60 N aborts were 43% of level-0.1 terminations in
  the 80k run and pinned the curriculum at the 0.05↔0.1 boundary.
- **reset acceptance**: in-cage starts additionally require the settled lateral
  error ≤ `reset_inport_lateral_tol_m` (3 mm), so the 6 mm reset-IK tolerance
  cannot seed wedged starts the residual policy can't recover from.
- **timeout** = `max_episode_steps` (200). SB3 `TimeLimit` bootstraps Q on truncation.

### 3.5 Reverse curriculum (last-inch, relative to the port)
Retract span `last_inch_m = 0.072 m`: **seated (0) → cage entrance (0.046) →
~26 mm of free-space approach**. (The old 0.04 span never exited the cage —
seated depth 0.046 > 0.04 — so "level 1" still started 6 mm *inside* the
entrance and the policy never saw the approach.)

Start sampling (`_sample_start_tcp`) is **frontier-band**: retraction is drawn
from `[(level − curriculum_band) · span, level · span]` (band 0.25) so the
start distribution tracks the level instead of staying half-trivial, with a
`curriculum_easy_frac` (0.2) chance of an easy replay start in `[0, level]`
against forgetting. Jitter is **two-phase**: while the sampled tip is inside
the cage, lateral/yaw/tilt stay at sub-mm `*_inport` values (SFP clearance);
once outside the entrance they widen linearly to the full level-1 values
(±6 mm x/y, ±0.12 rad yaw, ±0.04 rad tilt). Net effect: level growth reads as
*seated → slides out of the port → approach pose varies in x/y/z*.

`CurriculumScheduler` advances when the success rate over a **fresh**
`eval_window` crosses the threshold and the recent force-abort+bad-collision
rate stays below the guard; it retreats on low success or high unsafe rate,
and clears the window after every change. The level is written to
`outputs/<run>/curriculum_level.txt` (env re-reads on reset → works across
`SubprocVecEnv` workers). `train.py` deletes a stale level file on any fresh
(non-resume) start — previously a leftover file silently started a "fresh"
run mid-curriculum.

---

## 4. Training

```bash
# via crun.sh inside the container:
pixi run python RL/train.py --scene --port-type sfp --steps 500000
```
Defaults (retuned 2026-07-03 for exploration; reasoned for a single 5090):
`--num-envs 16 --batch-size 1024 --train-freq 1 --gradient-steps 4`
(UTD=0.25 — same 4096 samples consumed per vec step as the old batch-4096
config, but 4 distinct smaller updates: noisier gradients + 2× faster policy
adaptation as the curriculum shifts the start distribution; never `-1`),
`--buffer-size 50000` (uint8 image obs is mandatory; RAM = buffer ×
channels·H·W × 2 for obs+next_obs → 128²×9 ≈ 295 KB/transition, 256²×9 ≈
1.18 MB/transition. **On this 30 GB box use `--image-size 128
--buffer-size 40000`** — 256² obs + any useful buffer does not fit; the
reward image stays native-res either way since it is never stored),
`--warmup-steps 10000` (level-0 starts are seated, long random warmups just
log 1-step aborts), `--tau 0.005 --gamma 0.99 --ent-coef auto`,
`--target-entropy -3.0` (SB3's auto = −dim(A) = −6 collapses exploration noise
early on this 6-D residual task; pass `auto` to restore),
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

**Videos** (`eval/rollout_video`) are recorded every
`--wandb-video-every-steps` env steps (default **5000**, on whenever W&B is),
each clip stitching `--wandb-video-episodes` (2) deterministic eval episodes of
up to `--wandb-video-steps` (200) steps with a metric overlay, on a persistent
eval env. With the default `--wandb-video-level -1`, **every episode mirrors
the CURRENT training curriculum level** (clips show the actual training
distribution); full-task competence is tracked numerically by the score eval,
which stays pinned at level 1.0. Pass `--wandb-video-level 1.0` to pin video
episodes to the full task instead. The old `--wandb-video-every` (episodes)
gate is deprecated: with a vec env the episode counter jumps in batches, so its
`% N == 0` check fired once at episode 1 and then ~never — that was the "one
4-second video" symptom.
Score curves (`eval/*`) come from `--wandb-eval-every` (default 10000), each
point averaging `--wandb-eval-episodes` (2) no-video eval episodes at level 1.0.

### 7.2 Image-distance reward validation
`RL/scripts/validate_image_reward.py` sweeps controlled port-frame placements
(axial 0→7.2 cm, lateral ±6 mm) and checks the image-L1 signal:
2026-07-03 result — axial Spearman **+0.989** (native) / **+0.984** (256²),
dynamic range 1.0, PASS; but the curve is basin-shaped (0→0.23 within the
first 6 mm, then nearly flat) and the lateral gradient at 10 mm standoff is
slightly *misleading* (wrist-cam parallax). Consequence: the image term is the
sharp near-goal basin; `r_depth`/`r_xy` (true port-frame geometry) own the
approach and centring gradients. Plot: `RL/tests/new_image_reward_validation.png`.

---

## 8. Measured perf (this box: i7-13700K, RTX 5090, in-container)
| stage | cost | note |
|---|---|---|
| MuJoCo EGL render (256²) | ~0.07 ms | HW-accelerated on the 5090 |
| `scene_env` reset | ~0.1–0.9 s | IK + settle (dominant reset cost) |
| SAC grad step (batch 4096) | ~15–30 ms | the training wall (GPU) |

Expect env throughput to jump once geometry is tuned (episodes stop aborting at
step 1, so far fewer heavy resets).
