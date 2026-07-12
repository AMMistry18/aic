# SC-port 1-inch teacher — environment + training handoff

Created 2026-07-12. Audience: the engineer building the SC (fiber duplex-LC / SC)
port insertion **teacher** policy, mirroring the existing SFP teacher. Scope:
**environment setup + physics + teacher training only.** Student distillation,
perception, and Flowstate deploy are downstream (follow the SFP path in
`docs/FLOWSTATE_STATUS.md` and `RL/student_teacher/` once the teacher exists).

The SFP work is the template. Copy its structure; do NOT copy its mistake — the
Gate-0 contact randomization drove MuJoCo into `QACC` numerical instability
(9.8 kN forces, plug ejected 50–342 mm). Root cause + required fixes are in
`RL/student_teacher/STUDENT_V3_PILOT_ROOT_CAUSE_20260712.md`. **Physics stability
is gate 0 for SC too — read the "Good physics" section before anything else.**

---

## 0. What already exists for SC (do not rebuild these)

| Thing | Where | Status |
| --- | --- | --- |
| SC port VISUAL mesh + task-board bodies | `aic_utils/aic_mujoco/mjcf/aic_world.xml` (`sc_port_0::sc_port_link`, `sc_port_1::…`) | exists, **visual only** (`contype=0 conaffinity=0`) |
| SC entrance frame | `sc_port_base_link_entrance` (pos z = **-0.01564 m**) | exists |
| SC plug mesh | `sc_plug_visual_*` in same MJCF | exists, **visual only** |
| SC port keypoints / dimensions | `aic_example_policies/.../DataCollectorScPoseGT.py` | exists |
| SC pose perception weights | `best_sc_pose.pt` (bundled) | exists (for later student, not teacher) |

### SC geometry (measured from the repo — use these as starting values)

- **Port mouth:** half-width `SC_HALF_WIDTH_M = 0.0044` → **8.8 mm wide**;
  half-height `SC_HALF_HEIGHT_M = 0.0030` → **6.0 mm tall**. (Much smaller than
  SFP.)
- **Insertion depth:** entrance frame is **~15.6 mm** in front of the port link
  (`sc_port_base_link_entrance pos z=-0.01564`). **SC is a short insertion**
  vs SFP's ~45.8 mm. Your seated depth / curriculum retract must use ~15.6 mm,
  NOT 45 mm.
- **Duplex, angled:** there are two ports (`sc_port_0`, `sc_port_1`) at different
  task-board positions and each has a non-trivial mounting quaternion — they are
  NOT axis-aligned. Read the real quat from the MJCF; do not assume flat/vertical.

**MUST-VERIFY FIRST:** the env loads `aic_utils/aic_mujoco/mjcf/scene.xml`
(`RL/scene_env.py:64` `_DEFAULT_SCENE`). That file is a 678-byte wrapper that
`<include>`s the world. Confirm which included file actually carries the
*collision* geometry the SFP teacher trained against, and where SC bodies live in
the compiled model. `mujoco.MjModel.from_xml_path(scene_path)` then
`mj_name2id(..., "sc_port_0::sc_port_link")` to confirm the SC body compiles.

---

## 1. Build the SC collision model (SFP collision is the reference)

The SC port has no collision geometry yet. The SFP teacher did not collide against
the raw visual mesh — it used **programmatically-added primitive "contact ridge"
box geoms** at the entrance frame. Replicate that approach for SC.

Reference implementation: `RL/scene_env.py` `_compile_scene_model()` (~L305–368).
It adds four box "ridge" geoms around the SFP entrance:

```python
# SFP reference (scene_env.py ~L335):
geom = entrance.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[ridge_half_width, 0.0065, 0.0015],       # 3 mm axial contact band
    pos=[sign*(ridge_inner_x+ridge_half_width), 0, ridge_depth])
geom.contype = 1; geom.conaffinity = 1
geom.priority = 10
geom.friction = [2.0, 0.01, 0.001]
```

For SC, mirror this but with SC dimensions and a **round/duplex** lead-in:

- **Do NOT collide the raw visual mesh.** Mesh-mesh contact is the fastest route
  to `QACC` blow-ups. Use primitive geoms (BOX ridges like SFP, or a CYLINDER
  bore + chamfer for the round SC ferrule).
- Size the collision bore to the SC mouth (8.8 × 6.0 mm) minus the plug's own
  collision half-extents, leaving a clearance in the SFP-style range
  (`compiled_contact_ridge_clearance_range_m`).
- Put the contact band a few mm inside the ~15.6 mm bore (SFP used a 3 mm axial
  band centered in the 6–7.5 mm region; scale to SC's shorter depth).
- Give the plug its own primitive collision geoms (the SFP plug had
  `contact_collision_1`, `non_contact_collision`); build the analogous SC plug
  collision (a short cylinder/box for the ferrule).
- Keep `priority`, `contype/conaffinity`, and `friction` matching the SFP ridge
  as the starting point, then tune.

---

## 2. Good physics (READ THIS — it is why the SFP pilot failed)

The SFP Gate-0 randomization ejected the plug because contact went numerically
unstable (`QACC` warnings, kN forces). The required fixes (from the root-cause
note) — apply from the start for SC:

1. **Keep contact solver time constant SAFE vs the timestep.** MuJoCo REF becomes
   unsafe below `2*timestep`. This scene's timestep makes 4 ms the floor
   (`scene_env.py:252`). The SFP randomization went as low as
   `random_contact_timeconst_range_s=(0.004, 0.040)` — the low end sits right at
   the unsafe boundary. For SC, **start `solref` timeconst well above `2*dt`**
   (e.g. ≥ 5–8 ms) and only widen the range after guided-only episodes are stable.
2. **Bound the randomization ranges conservatively first, widen later.** The SFP
   ranges that blew up (`scene_env.py:249–259`):
   - `random_friction_scale_range=(0.35, 2.5)` (log-uniform)
   - `random_contact_timeconst_range_s=(0.004, 0.040)`
   - `random_contact_dampratio_range=(0.35, 2.0)`
   - `random_policy_hz_range=(2.5, 20.0)`, `random_controller_scale_range=(0.75,1.25)`
   Start SC with **narrower, stable** versions (higher timeconst floor, dampratio
   nearer 1.0, friction nearer 1.0) and expand only after the stability gate.
3. **Add an immediate off-limits termination.** If the plug leaves a sane spatial
   envelope (e.g. >30–50 mm lateral from the port, or force > a physical ceiling
   like 200 N), terminate the episode as invalid. A diverging sim must not keep
   running and feed garbage transitions to the learner.
4. **`solimp`/`solref` on the new SC geoms:** set explicit, stable values rather
   than inheriting defaults; keep them consistent between plug and port.

### Stability gate 0 for SC (MUST pass before any teacher training)

Run **guided/scripted control only (no learning)** under the SC randomization and
confirm:
- **Zero `QACC` warnings** across the randomized episodes.
- **No one-step lateral jumps** > a few mm (the SFP failure showed 50–342 mm
  single-step ejections).
- Peak force stays in a **realistic contact range (tens of N, not kN)**.
- Nominal (un-randomized) success is high (SFP nominal guided was 10/10).

Reuse `RL/student_teacher/gate0_contact_jam.py` as the probe (it takes
`--controller guided`); point it at the SC target frame. Only when guided-only is
stable does a *realistic jam* (short stall at a few N–tens of N), not an
explosion, exist to train against.

---

## 3. Environment wiring (SC swaps in scene_env)

`RL/scene_env.py` `SceneEnvConfig` is the config object. The SC env is mostly the
SFP env with these fields repointed:

- `insert_target_body`: `"sfp_port_1_link_entrance"` → the SC entrance frame
  (e.g. `"sc_port_0::…/sc_port_base_link_entrance"` — use the exact compiled body
  name).
- `plug_axis_tail_body`: `"sfp_module_link"` → the SC plug body.
- Seated / insertion depth: **~15.6 mm** (SC), not the SFP ~45.8 mm. Update
  `seated_depth_m` and any curriculum `last_inch_m` accordingly (SC's "1 inch"
  envelope is relative to a 15.6 mm bore).
- Contact-ridge config (`compiled_contact_ridge_*`): re-derive depth/clearance
  ranges for the SC bore.
- Jitter / handoff envelope (`jitter_xy_m`, `jitter_yaw_rad`, `jitter_tilt_rad`):
  keep SFP magnitudes as a start, but SC's smaller mouth (8.8×6 mm) means the
  same absolute lateral jitter is a larger *fraction* of the mouth — expect to
  reduce it.
- Plug/tip calibration constants (SFP had `SFP_TIP_IN_TCP_POS/QUAT` in
  `aic_model/aic_model/rl_insert_contract.py`): the SC teacher needs the SC
  plug-tip-in-TCP transform. For a **privileged teacher** you can read plug/port
  ground truth directly from the sim (no perception), so this is only needed for
  the later student/deploy.

---

## 4. Teacher training

The teacher is a **privileged-observation policy** (sees true plug/port/contact
geometry from the sim), trained with SAC in MuJoCo — same pattern as
`teacher_level1.zip`.

- **Entry points:** `RL/residual_sac/trainer.py` (primary) and
  `RL/sb3_sac/trainer.py` (`RL/__init__.py`). The trainer builds the MuJoCo scene
  as a `SubprocVecEnv` and runs SAC (stable-baselines3).
- **Reward:** `RL/reward.py`. Keep success + depth-progress as primary; penalize
  lateral, force, collision, stalled pushing. **Watch reward signs** — the SFP
  Student-v3 had a sign bug that refunded 70% of the lateral penalty during
  contact (root-cause note §secondary). Verify each penalty is actually negative.
- **Observation:** privileged state for the teacher (true tip/port pose, contact,
  physics). See `RL/student_teacher/TEACHER_OBS_INTERFACE.md` for the SFP
  teacher's observation/success definition to mirror.
- **Curriculum:** reverse last-inch curriculum (start seated, retract toward the
  full envelope) — see `scene_env.py:118–135`. Scale distances to SC's 15.6 mm.
- **Do not freeze/modify** the SFP `teacher_level1.zip`. The SC teacher is a new,
  separate checkpoint.

### Log these throughout training (the SFP pilot did NOT, and was undebuggable)

Return, **peak lateral, peak force, collision rate, force-abort rate, stall
steps, outcome breakdown, and QACC-warning count** vs steps. Select checkpoints
by held-out success + collision + force, NOT training reward.

---

## 5. TACC (training compute)

Same environment as the SFP work (`RL/student_teacher/TACC_NEXT_AGENT_HANDOFF.md`):

- `ssh satya_a@stampede3.tacc.utexas.edu` (account password + 6-digit MFA).
- `WORK=/work2/11590/satya_a/stampede3`, `SCRATCH=/scratch/11590/satya_a`.
- Pixi directly on TACC (`export PATH="$HOME/.pixi/bin:$PATH"`), headless MuJoCo
  `MUJOCO_GL=egl`. **No Distrobox on TACC.**
- Do NOT modify the frozen SFP teacher, snapshots, datasets, or the shared Pixi
  env. Put SC outputs under a new `SCRATCH` subdir.
- Run 2–3 pilot seeds (~250–300k) FIRST; continue only the best to full scale,
  and only after the SC stability gate (§2) passes with guided-only control.

---

## 6. Recommended order of work

1. Confirm the SC bodies compile in the loaded scene; find where collision
   geometry lives (§0 MUST-VERIFY).
2. Build SC primitive collision geometry (plug + port bore/ridge), SFP as
   reference (§1).
3. Set stable `solref`/`solimp`/timestep and conservative randomization (§2).
4. **Pass the guided-only SC stability gate** — zero QACC, no ejections,
   realistic forces (§2). Do not train until this passes.
5. Wire the SC env config (§3), verify reward signs (§4).
6. Pilot the teacher (2–3 seeds, 300k) on TACC (§5); evaluate; then scale.

---

## Key references

- SFP collision reference: `RL/scene_env.py` `_compile_scene_model()` ~L305–368.
- Physics failure + required fixes: `RL/student_teacher/STUDENT_V3_PILOT_ROOT_CAUSE_20260712.md`.
- Teacher obs/success interface: `RL/student_teacher/TEACHER_OBS_INTERFACE.md`.
- Trainers: `RL/residual_sac/trainer.py`, `RL/sb3_sac/trainer.py`; reward `RL/reward.py`.
- SC geometry/keypoints: `aic_example_policies/.../DataCollectorScPoseGT.py`.
- TACC access: `RL/student_teacher/TACC_NEXT_AGENT_HANDOFF.md`.
