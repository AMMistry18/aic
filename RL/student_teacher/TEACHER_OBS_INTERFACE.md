# Teacher Observation / Information Interface (Phase-3)

Privileged-teacher information audit for the SFP insertion task, so later
student distillation knows exactly what to withhold. All keys/numbers below are
read directly from `RL/scene_env.py` (`_obs`, `observation_space`, port-frame
helpers, `_reward_and_term`) — nothing is guessed. Line refs are approximate.

## 1. Env observation — what *any* policy here receives (`_obs`, ~L1517; `observation_space`, ~L338)

The env returns a `Dict` obs. Every policy (teacher or student) trained against
this env sees exactly these keys:

| key | shape | source | note |
|-----|-------|--------|------|
| `arm_qpos`   | (6,)  | `data.qpos[arm]` | joint angles — measurable on a real arm |
| `arm_qvel`   | (6,)  | `data.qvel[arm]` | joint velocities — measurable on a real arm |
| `tcp_pose`   | (7,)  | `site_xpos` (3) + `site_xmat`→quat (4) | **PRIVILEGED**: exact, noise-free sim TCP pose |
| `ft`         | (6,)  | `_raw_ft()` = force(3)+torque(3) sensors | **FULL 6-axis wrench** (see note) |
| `last_action`| (action_dim,) | cached | previous commanded action |
| `image`      | (256, 256, 3·N_cams) uint8 | 3 wrist cams stacked (`center/left/right_camera`) | low-res RGB |

Idealizations already baked into this obs relative to real deployment:
- **`tcp_pose` is EXACT sim pose** — it comes straight from the simulator site
  transform with **no perception noise, no estimation, no latency**. A real
  robot only knows TCP pose through forward kinematics + calibration + (for the
  part) a noisy perception estimate. This is a privileged, un-deployable signal
  as given.
- **`ft` is the full 6-axis wrench** (3 force + 3 torque). This is *physically
  measurable* by a real 6-axis FT sensor, so it is idealized (noise-free) but
  **not un-deployable** in principle.
- (Config note: `cfg.image_h/w = 256`, `cameras = (center, left, right)` → 3
  cams stacked on the channel axis = 9 channels. Earlier notes citing
  "128×128×3" are stale; the code says 256×256×9.)

## 2. Ground truth the privileged SCRIPTED teacher reads directly (NOT in the obs dict)

The scripted teacher/env computes the port frame and goal analytically from sim
state (`_configure_port_frame`, ~L475, and helpers). None of these are in
`_obs`; they are the teacher's true information set:

- `_port_pos` — world position of the port mouth / insertion entrance.
- `_insert_axis` — unit insertion direction in world (port +Z).
- `_lat_x`, `_lat_y` — port-frame lateral basis (orthonormal to insert axis).
- `_inserted_tip` — seated tip target = `_port_pos + seated_depth_m·_insert_axis` (`seated_depth_m = 0.0458 m`).
- `_goal_tcp` — the exact TCP pose that seats the plug (`_tcp_for_tip`).
- `_goal_quat` — the aligned seated orientation (`_aligned_goal_quat`).
- `_plug_axis_error()` — exact plug-axis vs insert-axis misalignment (rad).
- `_plug_roll_error()` — exact keyed-cross-section roll error (rad).
- `_insertion_depth_m()` / `_depth_norm()` — exact signed tip depth (m) and its
  normalized fraction in [0,1] (0 = mouth, 1 = seated).
- `_overinsert_m()` — exact overshoot past the seated frame (m).
- `plug_port_penetration_m` / `_excess_m` (`_contact_summary`, ~L1206) — exact
  geometric interpenetration of plug and port from MuJoCo contact distances.
- contact classification/counts — `plug_port_contacts`, `port_stop_contacts`,
  `off_limit_contacts` (enclosure/task_board), with worst-pair names.
- wrench decomposition — the same 6-axis wrench split into `f_axial` (along
  `_insert_axis`) and `f_lat` for reward/termination.

## 3. The deployable STUDENT gap — what must be WITHHELD / replaced by perception

Must be **withheld** from the student (or replaced by a noisy perception estimate):
- Exact object/port pose and the whole **GT port frame**: `_port_pos`,
  `_insert_axis`, `_lat_x/_lat_y`, `_inserted_tip`, `_goal_tcp`, `_goal_quat`.
  A real robot must *estimate* these from vision → noisy, biased, possibly
  intermittent.
- The **exact noise-free `tcp_pose`** as an oracle part-relative pose (real pose
  is FK + calibration + a perception estimate of the port, not ground truth).
- Exact geometric **penetration / penetration-excess / overinsert** and
  **contact-normal / contact classification** signals — the sim knows these from
  collision geometry; a real robot cannot directly perceive interpenetration.
- Exact `_depth_norm`, `_plug_axis_error`, `_plug_roll_error` — these are
  functions of the GT port frame, so they inherit its privilege.

What a real robot CAN still measure (NOT privileged in the un-deployable sense —
keep for the student, add realistic noise):
- Joint state: `arm_qpos`, `arm_qvel` (encoders).
- The **6-axis FT wrench** `ft` — a real wrist FT sensor provides force+torque.
- A **low-res camera image** — the student can use `image` (with realistic
  noise/lighting) as its perception input in place of the GT port frame.

## 4. Success / termination definition (`_reward_and_term`, ~L1390; cfg ~L155)

`term_status` ∈ {`success`, `bad_collision`, `force_abort`, `timeout`}; step marks
`terminated` for {success, force_abort, bad_collision, off_limit} and `truncated`
for timeout. `off_limit` is otherwise a one-shot scoring event
(enclosure/task_board contact), non-terminal in reward.

**Success** requires ALL of:
- `depth_norm >= 0.99` (`success_depth_norm`) and
  `insertion_depth_m >= seated_depth_m − 0.003` (`success_axial_tol_m`),
- `|axial_err| < 0.003 m`, `lateral_err < 0.005 m` (`success_lateral_tol_m`),
- `axis_error <= 0.035 rad` (`success_axis_tol_rad`),
  `roll_error <= 0.15 rad` (`success_roll_tol_rad`),
- `overinsert <= 0.001 m` (`success_max_overinsert_m`),
- `penetration_excess <= 0.001 m` (`success_max_plug_port_penetration_excess_m`),
- at least one plug↔port contact (`success_require_port_contact = True`),
- optional seating-force gate disabled (`success_force_n = 0.0`).

**bad_collision**: `pen_excess > 0.0015 m` OR `overinsert > 0.002 m` OR
(deep-enough `depth_norm` AND axis/roll past `bad_collision_axis/roll` bounds).
**force_abort**: peak wrench `> 120 N` (`force_abort_hard_n`, instant) OR `> 60 N`
(`force_abort_n`) sustained for `>= 3` steps (`force_abort_dwell_steps`).
**timeout**: `step_count >= max_episode_steps`.

## 5. Note — the earlier "deployable SAC baseline" assumption was wrong

An earlier project assumption held that the existing SAC baseline was already
"deployable." It was not. That baseline was trained on **this same idealized obs
dict** — in particular the **exact, noise-free `tcp_pose`** (an oracle sim
transform, not a perception estimate) and the **full 6-axis wrench**. The exact
pose is a privileged, un-deployable signal a real robot cannot obtain without
perception, so the SAC baseline was already **partially privileged**, not a true
deployable policy. Student distillation must replace that oracle pose (and all
GT port-frame quantities in §2) with vision-derived, noisy estimates, keeping
only the genuinely measurable channels (joint state, 6-axis FT, low-res image).
