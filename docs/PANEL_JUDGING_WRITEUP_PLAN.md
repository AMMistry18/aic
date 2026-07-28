# Phase 1 panel writeup plan: AIC model and perception

Target length: 500–1000 words. Recommended final length: 750–850 words.

## Core positioning

Describe the submitted system as a **hybrid perception-and-control pipeline**, not
as an end-to-end RL policy. The deployed Docker image explicitly selects
`aic_model.RLInsert`, fixes `CONTROL_MODE` to `script`, and disables both learned
insertion actor paths. Its real contribution is a geometry-aware, fail-closed
perception stack coupled to bounded Cartesian impedance control, force feedback,
and event-confirmed seating.

Recommended thesis:

> Our solution converts a randomized cable-insertion task into a sequence of
> observable, mechanically meaningful subproblems: find a safe survey view,
> estimate both the target opening and the plug actually held by the gripper,
> align in the port frame, and seat under bounded force. This hybrid design
> combines learned keypoint perception where object appearance is difficult
> with explicit geometry and compliant feedback where safety, smoothness, and
> failure diagnosis matter most.

## Proposed structure and word budget

### 1. System overview — 80–100 words

- State the task: randomized SFP and SC cable-end insertion with a UR5e,
  Robotiq Hand-E, three wrist cameras, and a wrist force/torque sensor.
- Present the pipeline in one sentence:
  board survey -> motion-planned handoff -> multi-view plug/port pose ->
  plug-relative alignment -> force-regulated seating -> insertion-event check.
- Emphasize that perception and insertion are closed around the current scene
  and current grasp rather than fixed board coordinates or a nominal grasp.

### 2. Perception — 220–260 words

- **Board-level perception:** `check_board_visibility_skill_v4` uses only
  participant-accessible images, camera intrinsics, timestamped transforms, and
  joint state. It localizes the purple board insignia, reconstructs board pose,
  and searches CAD-derived, target-specific survey viewpoints for staged SFP
  modules, NIC destinations, and SC ports.
- **Viewpoint quality and safety:** candidate views are checked for target
  coverage, camera clearance, obliquity, UR5e reachability, wrist-camera
  keep-out, and arm visibility before a pose is returned. The custom skill
  returns a target; the standard motion planner executes the trajectory.
- **Insertion-level learned perception:** separate YOLO pose models represent
  the SFP port, physical SC mouth, SFP plug, and SC plug. Camera detections are
  transformed using timestamp-matched calibration and fused across views.
- **Geometry fusion:** SFP/SC plug keypoints are confidence-filtered,
  triangulated, and rigidly fit to known plug geometry. Port candidates are
  selected relative to the measured held plug. Multiple temporal estimates
  must agree before control commits.
- **Fail-closed behavior:** stale frames, fewer than two usable cameras,
  inconsistent rigid geometry, high reprojection error, or unavailable weights
  abort insertion instead of falling back to a fixed grasp or hand-tuned pose.
- Evidence worth quoting: the SC plug model used 4,050 simulated images and a
  YOLO11s-pose model trained at 960 px. With native-resolution crop refinement,
  held-out validation reached 0.273 mm median 3-D position error, 0.329 mm p95
  lateral error, 1.88 degrees p95 axis error, and zero group misses. The physical
  SC-mouth model achieved 1.06 px median / 2.56 px p95 held-out centre error;
  do not quote its weak single-view PnP metric because deployment uses
  multi-camera centre triangulation plus corner-based orientation.

### 3. Motion intent and fluidity — 150–180 words

- The insertion controller operates in the port coordinate frame and sends
  absolute Cartesian impedance targets at 20 Hz.
- Alignment steps are bounded in translation and rotation. SFP is limited to
  1.5 mm and 1.5 degrees per alignment update; SC uses a tighter 0.30 mm final
  lateral tolerance because of its smaller mechanical clearance.
- Axial speed decreases near contact. SFP uses 15 mm/s in free space and
  6 mm/s in contact; SC uses 8 mm/s and 4 mm/s, then scales further near the
  mouth.
- Wrench feedback is a low-pass, slew-limited correction rather than an
  accumulating offset, preventing abrupt lateral jumps. Nominal seating forces
  are capped below hard abort thresholds: 10/12/18 N for SFP and 5/7/12 N for
  SC (target/cap/abort).
- Stalls trigger bounded micro-recovery, unloading, re-perception, or a fresh
  retry. Hard force violations hold/abort. Success requires a fresh physical
  insertion event, not simply reaching a commanded depth.
- Connect these details directly to visible behavior in the submitted video:
  deliberate approach, pause at contact, compliant seating, and bounded retry.

### 4. Innovation and industrial viability — 160–190 words

- Innovation is the **plug-relative hybrid architecture**: it measures the
  transform of the plug produced by each new grasp rather than assuming a
  constant TCP-to-tip transform, then combines learned keypoints, rigid-body
  geometry, temporal consensus, force/moment feedback, and explicit recovery.
- The physical SC-mouth model replaced a visually misleading virtual label
  rectangle with a five-keypoint representation of the actual 22.407 x 8.10 mm
  opening.
- Native-resolution crop refinement was selected from a measured bias analysis,
  reducing SC plug median error from 0.456 mm to 0.263 mm on the held-out test
  split without increasing group misses.
- Industrial viability comes from modular ROS 2 interfaces, standard camera
  calibration/TF, target-specific mechanical parameters, hard safety envelopes,
  bounded deadlines, interpretable diagnostics, and failure rather than blind
  motion when observations are unreliable.
- Do not imply direct sim-to-real deployment. Say the design is *transferable*
  because it depends on calibrated images, geometry, force, and insertion
  events—not simulator ground truth—and identify lens calibration, appearance
  randomization, and hardware force tuning as the remaining commissioning work.

## Evidence map

| Claim | Primary repository evidence |
| --- | --- |
| Actual deployed policy and RL disabled | `docker/aic_model/Dockerfile`, `docker/aic_model/v50_overlay/aic_model/RLInsert.py` |
| Plug-relative SFP state machine | `docker/aic_model/v50_overlay/aic_model/v50_controller.py` |
| SC-specific force/control limits and retries | `docker/aic_model/v50_overlay/aic_model/sc_controller.py` |
| Multi-view plug estimation and rejection gates | `docker/aic_model/v50_overlay/aic_model/sfp_plug_pose.py`, `sc_plug_pose.py` |
| Learned SFP/SC port perception | `docker/aic_model/v50_overlay/aic_example_policies/ros/perception_core.py` |
| Board pose and survey-view search | `flowstate/aic_perception/check_board_visibility_skill.py`, `aic_perception/board_stage2.py` |
| SC plug held-out metrics | `docs/SC_PLUG_POSE_RESULTS.md`, `docs/reports/sc_plug_pose_*` |
| Physical SC-mouth training results | `.artifacts/HANDOFF_sc_mouth_pose_tacc_20260727.md` |
| Board-view sweep evidence | `docs/CHECK_BOARD_VISIBILITY_V4_HANDOFF.md`, `docs/BOARD_SEARCH_HANDOFF.md` |
| Packaged-image verification | `.artifacts/HANDOFF_FLOWSTATE_WSETO_FRESH_20260727.md` |
| RL/student-teacher work, not active deployment | `RL/student_teacher/README.md`, `docs/FLOWSTATE_MUJOCO_PARITY_20260711.md` |

## Claims that need final-run evidence

Before drafting the final version, fill in:

1. The exact submitted Docker image/bundle and Flowstate skill versions.
2. The final AIC Engine score and per-tier breakdown.
3. Success count across the official randomized runs.
4. Peak force, duration, path-efficiency, and jerk values from the best run.
5. Which recovery behaviors are visible in the submitted video.
6. Whether the uncommitted board-survey relaxation/coverage changes were
   packaged. The current working-tree handoff explicitly says they were not yet
   deployed, so their 144/144 sweep must not be attributed to the submitted
   solution without confirmation.

## Claims to avoid

- Do not call the deployed controller an RL policy or claim the TorchScript
  actor controls insertion. The container disables actor loading and inference.
- Do not claim the planned 300/300 frozen SFP gate passed. The repository
  contains the gate definition, but no committed passing result.
- Do not use MuJoCo actor success rates as final-system reliability; the
  student-teacher documents explicitly record deployment/parity gaps.
- Do not say the system uses simulator ground-truth object poses. The intended
  perception path uses images, intrinsics, timestamped TF, joint state, and
  wrench data.
- Do not present uncommitted, explicitly “not deployed” board-search results as
  submission behavior.
- Do not claim real-factory validation. Frame the architecture as
  factory-oriented and state the remaining real-hardware calibration work.

## Final drafting workflow

1. Freeze the identity of the submitted model image, custom perception skill,
   and Flowstate process.
2. Extract the best official run's score, force, duration, efficiency, jerk,
   and success evidence.
3. Choose two or three video timestamps demonstrating smooth alignment,
   compliant contact, and recovery.
4. Draft to approximately 800 words using the structure above.
5. Perform a claim audit: every number must resolve to a committed report or
   official final-run log; every active-behavior statement must resolve to the
   packaged overlay, not only the development source.
6. Tighten the prose around the three panel headings, then fit the required
   one-page layout.
