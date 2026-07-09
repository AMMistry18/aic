# AIC Phase 1 — Project Overview

*Last updated: 2026-07-03. Top-level map of this repo and the overall strategy.
Deep dives live in `RL/README.md` and the official docs in
`../aic_1/aic-phase-1/docs/`.*

---

## 1. What this is

Entry for the **AI for Industry Challenge (AIC), Phase 1** — Intrinsic's
(Alphabet) open robotics competition. The task: a **UR5e + Robotiq Hand-E**
in simulation must pick up fiber-optic cables and insert **both ends** (SFP
plug + SC plug) into their target ports on a task board.

Evaluation format:

- One **Flowstate process** (`AIC Phase 1 Submission`) is run **5× sequentially**
  — one cable per run — on a **randomized** taskboard layout (global pose +
  per-component rail offsets).
- Scores accumulate across the 5 runs.
- **If any insertion fails, the evaluation terminates** — later runs never
  execute. Per-run reliability compounds; a 90%-reliable run gives ~59% chance
  of even reaching run 5.
- Score is printed to the `aic_engine` log at end of trial
  (Flowstate → Window → Open Text Logs Viewer → `aic_engine`).

### Scoring rubric (per run)

| Tier | What | Points (local estimate) |
| :--- | :--- | :--- |
| Tier 1 | Flowstate process/workflow validity | gate |
| Tier 2 | jerk smoothness (6), duration (12), path efficiency (6), wrist force ≤ 20 N (−12 penalty), off-limit contacts (−24 penalty) | ~24 max |
| Tier 3 | port alignment / partial-insertion proximity (25) / full insertion (75); **averaged over the two cable ends** | 75 max |

Tier-2 bonuses are only awarded if **both** ends reach proximity of their
ports. Exact Tier-2 math is in `aic_scoring/src/ScoringTier2.cc` (qualification
scorer — Phase 1 is "derived from" it); our local mirror of it is
`_score_diag()` in `RL/scene_env.py` (`score_*` keys in metrics/W&B).

---

## 2. Strategy

Hybrid pipeline, per the suggested architecture (policy is *required* for
last-inch by the rules):

```
Flowstate (cloud portal)                          this repo
┌─────────────────────────────────────┐   ┌──────────────────────────────┐
│ perceive board → grasp cable end →  │ → │ RL policy: last-inch          │
│ motion-plan to ~inches from port    │   │ contact-rich insertion        │
└─────────────────────────────────────┘   └──────────────────────────────┘
        macro (built-in skills)                micro (trained SAC)
```

1. **Macro (Flowstate)** — built-in perception/grasp/motion-plan skill blocks
   get the plug near the port. Custom perception (YOLO-pose port detection,
   below) supports adapting to the randomized layout.
2. **Micro (this repo, `RL/`)** — image-based **SAC residual policy** takes
   over for the final ~1 inch: wrist-camera images + F/T + proprio in, 6-DoF
   pose deltas out, on top of the hand controller's impedance command.

---

## 3. Flowstate process anatomy (macro side)

*Sources: `../aic_1/aic-phase-1/docs/flowstate_tutorials.md`,
`flowstate_capabilities.md`, `explore_template_solution.md`.*

A Flowstate **Process** is a behavior tree of **Skills** (action nodes) calling
**Services** (containerized runtimes). Three world states matter:

- **Init** — persistent digital twin (solution config).
- **Belief** — the robot's runtime estimate; what motion planners use.
- **Sim** — Gazebo ground truth; **what gets scored**.

`spawn_taskboard_skill` places the randomized board in **SIM only** — the
Belief world starts blind, which is why perception is mandatory.

### Per-run skill sequence (from the template solution)

```
start_engine_skill                    # begin scored trial
  └─ spawn_taskboard_skill            # SIM-only randomized layout (disabled at eval)
  └─ tare_force_torque_sensor         # zero wrist F/T
  └─ estimate_pose_ivm_cloud          # IVM: CAD-prompted pose est., 3 wrist cams
       └─ filter_estimates → create_object   # populate Belief world
  └─ grasp + motion-plan skills       # macro: bring plug near port
  └─ switch_to_aic_controller_skill   # enable aic_controller/pose_commands streaming
  └─ lifecycle_transition_skill       # configure + activate aic_model
  └─ insert_cable_skill               # InsertCable action → our policy
       # params: cable_type/name, plug_type/name, port_type/name,
       #         target_module_name, time_limit  (wired from process inputs)
  └─ switch_to_default_controller_skill
  └─ (repeat macro→policy for the second cable end)
stop_engine_skill                     # finished=true on last run → total score
```

### Notes

- **Deployment boundary**: the trained policy must live behind the
  `aic_model` ROS 2 lifecycle action server (`InsertCable` action), consuming
  `/observations` from `aic_adapter` and commanding
  `aic_controller/pose_commands`. That is the exact interface the RL policy
  exports to.
- **IVM (Intrinsic Vision Model)**: cloud pose estimation supports exactly the
  AIC assets (`aic_nic_card`, `aic_sc_plug`, `aic_sc_port`, `aic_sfp_module`)
  via `cloud_ml_proxy_service`; recommended with all 3 wrist cameras. The
  custom YOLO-pose pipeline (`train_sc.py`, `perception_core.py`) is a
  fallback/refinement alongside it.
- **Controller switching is stateful**: after `insert_cable_skill` completes
  you must call `switch_to_default_controller_skill` before any cataloged
  move skill, and `switch_to_aic_controller_skill` again before the next
  policy invocation.
- Public Flowstate docs are mostly behind the beta login; the asset catalog is
  at <https://flowstate.intrinsic.ai/docs/assets/assets_catalog/overview/> and
  the SDK at <https://github.com/intrinsic-ai/sdk>. The local docs above are
  the authoritative reference for the AIC-specific skills.

---

## 4. Repo layout

Fork of the official `intrinsic-dev/aic` toolkit. **Bold = our additions.**

| Path | What |
| :--- | :--- |
| **`RL/`** | Last-inch insertion RL: real MuJoCo scene, reward, compact SB3 SAC, and full residual-SAC training. See `RL/README.md`. |
| **`train_sc.py`** | YOLO-pose training for **SC port** detection (4 keypoints; NIC recipe tuned for tiny ports). Weights → `~/bestSC.pt`. |
| **`perception_core.py`** | Standalone (no-ROS) perception helpers: NIC/SC detection + multi-camera triangulation. |
| **`eval_sc_pose_model.py`, `eval_color_sc.py`, `sc_pose_sanity_check.py`, `sc_policy_eval_summary.py`** | Perception/policy eval + sanity-check scripts. |
| `aic_model/` | Policy container (ROS 2 service) — where the trained policy gets deployed for submission. |
| `aic_controller/` | Impedance/robot controller the policy commands. |
| `aic_adapter/` | Publishes `/observations` (images, F/T) to the policy. |
| `aic_engine/` | Trial orchestration/recording (stripped down in Phase 1 — scoring + metrics only). |
| `aic_scoring/` | **Official Tier-1/2 scoring implementation (C++)** — ground truth for what's graded. |
| `aic_assets/`, `aic_description/`, `aic_gazebo/` | Task board / robot models, Gazebo world. |
| `aic_utils/` | Incl. `aic_mujoco`, teleoperation, training utils. |
| `docs/` | Official qualification-phase docs (scoring, policy integration, interfaces). |

Related, outside this repo:

- `~/aic_ws/src/gz-mujoco` — **SDF→MJCF converter** used to re-export the real
  Gazebo scene (ports, cable, board) into MuJoCo for fast RL training
  (~13.5k fps EGL on RTX 5090 vs. slow Gazebo). Recent fixes: light conversion,
  submesh/material consolidation, joint pose resolution.
- `~/AIC_Phase_1/aic_1/aic-phase-1` — official Phase-1 docs repo
  (task description, scoring & submissions, Flowstate architecture/tutorials).

---

## 5. RL pipeline (summary — details in `RL/README.md`)

- **Env**: `RL/scene_env.py` `SceneInsertEnv` — real exported AIC scene
  (UR5e + Hand-E + welded LC/SFP plug + elastic cable + NIC/SC ports),
  targeting `sfp_port_1_link_entrance`.
- **Reward**: `RL/reward.py` provides geometry-first depth progress, success,
  alignment, force, collision, and smoothness terms.
- **Curriculum**: reverse curriculum (Florensa-style) — start seated
  (level 0), back the start pose out toward the full last-inch envelope
  (level 1) as the success rate over a window clears a threshold.
  `CurriculumScheduler` in `RL/logging_utils.py`.
- **Score diagnostics**: every episode logs `score_*` keys (local rubric
  estimate: tiers, duration/efficiency/smoothness points, force/off-limit
  penalties) so training progress is legible in competition points, not just
  reward. Periodic W&B score evals + rollout videos.
- **Runs**: `RL/output/<run>/` — `config.json`, `metrics.jsonl`, `run.log`,
  `model.zip`, `curriculum_level.txt`. W&B: entity `tar2`, project `intrinsic`.
- **Environment**: everything runs inside the `aic_eval-latest` distrobox
  container using the repo **pixi** env (`crun.sh` wrapper recipe in
  `RL/README.md` §0).

Known gaps between training env and the real eval (see git history / W&B for
current numbers):

- Training is **single-ended** (SFP end); real Tier 3 averages **two** ends
  and gates Tier-2 bonuses on both ends reaching proximity.
- Fixed exported scene vs. randomized taskboard at eval time.
- Local force check uses plug contact force; official uses wrist F/T
  (`/fts_broadcaster/wrench`). Threshold (20 N) matches.

---

## 6. Quickstart pointers

- **Train / resume RL**: `RL/README.md` §0 (container + pixi recipe) — canonical
  command lines live there and in each run's `RL/output/<run>/config.json`.
- **Train SC port detector**: `pixi run python3 train_sc.py` (see docstring).
- **Submission flow**: `../aic_1/aic-phase-1/docs/scoring_and_submissions.md` —
  copy the Flowstate solution, submit Solution ID + version via the Google
  Form; don't commit to the submitted copy afterwards.
