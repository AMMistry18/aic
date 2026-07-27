# Check Board Visibility v4 — complete next-session handoff

Updated: 2026-07-27

> **Read this first.** Stage 2 (the geometric survey) is in good shape: the SFP
> and SC view policies were reworked on 2026-07-26/27 and both are confirmed
> working on hardware. **Stage 1 (insignia acquisition) is inherently broken and
> is to be scrapped and rebuilt** — see §19, which is the next session's whole
> job. Sections 5 and 20 describe what is being replaced and why.

This is the consolidated handoff for
`ai.tar2.check_board_visibility_skill_v4`. It describes the current working
tree, including changes that are tested locally but not committed or fully
validated on hardware. It covers the shared pipeline, the separate SFP, NIC,
and SC survey policies, the Flowstate motion boundary, downstream IVM/filter
behavior, known failures, tests, and deployment.

Use this document as the entry point next session. The deeper reasoning trail
remains in:

- `docs/BOARD_SEARCH_HANDOFF.md`, especially **SC destination ports — full
  reference**;
- `docs/SURVEY_IK_SESSION_HANDOFF.md`, especially section 9. Section 10 is
  superseded history;
- `flowstate/README.md`, for the concise runtime contract.

## 1. Repository and current source state

The real repository is:

```text
C:\Users\anshu\College\aic\aic
```

The outer `C:\Users\anshu\College\aic\.git` is empty. Run all `git`, tests, and
searches from the inner `aic\aic` directory.

At the time of this handoff:

```text
branch: main
HEAD:   467636d  (working tree clean)
```

Unlike the previous revision of this handoff, the Stage-2 policy work is
**committed**. The SFP coverage fix and the SC depth fix described in §8 and
§10 are both in `HEAD`, and both are confirmed working on hardware.

The one relevant branch that is *not* merged is `origin/navigate-to-purple`
(tip `4a20097`), which carries an independent Stage-1 experiment. §20 covers it.

The v4 asset identity is defined in
`flowstate/aic_perception/check_board_visibility_skill.manifest.textproto`:

```text
ai.tar2.check_board_visibility_skill_v4
```

The main implementation files are:

| File | Responsibility |
| --- | --- |
| `flowstate/aic_perception/check_board_visibility_skill.py` | Skill orchestration, Stage 1, Stage 2 integration, target policy, result semantics |
| `flowstate/aic_perception/aic_perception/board_visibility.py` | Board/insignia image processing and Stage-1 mask reports |
| `flowstate/aic_perception/aic_perception/board_stage2.py` | CAD geometry, PnP, camera projection, bore gates, survey-pose search |
| `flowstate/aic_perception/aic_perception/arm_ik.py` | UR5e FK, closed-form IK, live calibration, self-clearance |
| `flowstate/aic_perception/aic_perception/camera_rig.py` | Fresh images, CameraInfo, force, frame validation, timestamp handling |
| `flowstate/aic_perception/aic_perception/gripper_masks.py` | Per-camera image-space gripper exclusions |
| `flowstate/aic_perception/aic_perception/robot_motion.py` | Bounded force-guarded Stage-1 motion and controller handoff |
| `flowstate/aic_perception/aic_perception/viewpoint_search.py` | Stage-1 phase machine — **broken, to be scrapped, see §19** |
| `flowstate/aic_perception/test/sc_sweep_runner.py` | 144-case SC pre-hardware sweep |
| `flowstate/aic_perception/test/sfp_sweep_runner.py` | 144-case staged-SFP sweep with an independent seat audit |
| `flowstate/aic_perception/check_board_visibility_skill.proto` | Flowstate parameter and result schema |
| `docs/reference/filter_estimates_sc_node.py` | Paste-ready downstream SC target selector; not part of the skill bundle |

## 2. What v4 does — and does not do

The skill has two conceptual stages:

```text
three wrist cameras + CameraInfo + timestamped robot TF + measured joints
                                |
                                v
Stage 1: expose a complete purple insignia if it is not already visible
                                |
                                v
Stage 2: insignia PnP -> full board pose -> target-specific camera-pose search
                                |
                                v
result.target.{x,y,z,qx,qy,qz,qw} and result.survey_pose
                                |
                                v
downstream Move Robot -> cloud IVM -> target-specific filter
```

The skill may move during Stage 1 using the AIC controller. It never executes
the final survey move. Stage 2 is perception and geometry only. It publishes a
Cartesian TCP pose in `base_link`; the downstream Move Robot node owns the
actual trajectory, collision planner, and final joint branch.

The code currently routes all deployed target values through the geometric
Stage 2:

| `survey_target` | Value | Sector |
| --- | ---: | --- |
| `UNSPECIFIED` | 0 | Historical SFP default |
| `STAGED_SFP_MODULE` | 1 | Loose SFP modules on the +Y staging rail |
| `NIC_SFP_DESTINATION` | 2 | Ten NIC-card SFP-port bores |
| `SC_DESTINATION_PORT` | 3 | Five SC adapter mouths |

Some comments in the proto and manifest still say only SFP uses geometric
Stage 2 and NIC/SC retain legacy behavior. Those comments are stale. The
runtime authority is `_uses_geometric_survey()`, which returns true for
0, 1, 2, and 3.

The skill never reads a task-board ground-truth transform, component pose,
simulation object pose, or scoring state. Its input allowlist is closed:

- images: `/left_camera/image`, `/center_camera/image`,
  `/right_camera/image`;
- intrinsics: the matching `/camera_info` topics;
- optical TF: `left_camera/optical`, `center_camera/optical`,
  `right_camera/optical`;
- TCP/base TF: `gripper/tcp` and `base_link`;
- arm state: `/joint_states` and `/aic_controller/controller_state`;
- force: `/fts_broadcaster/wrench`;
- Stage-1 commands: `/aic_controller/pose_commands`,
  `/aic_controller/joint_commands`, and
  `/aic_controller/change_target_mode`.

Images and CameraInfo may name either the camera's exact optical frame or its
matching `sensor_link`. Prefix matches and arbitrary frame overrides are
rejected.

## 3. Required Flowstate wiring

The safe serial sequence is:

```text
Move Robot to the pre-survey/start pose
-> Switch To AIC Controller
-> Check Board Visibility v4
-> Switch To Default Controller
-> require result.success && result.done && result.target_valid
-> pack result.target x/y/z/qx/qy/qz/qw as a base_link Cartesian TCP pose
-> Move Robot to the survey pose
-> cloud IVM
-> target-specific filter
-> create/use the requested belief
```

The controller switch back must run before the result gate. The skill returns
expected sensor/search failures normally so that Flowstate can always release
the AIC controller session. Throwing a skill error before cleanup previously
left `arm` in use and broke the next Move Robot node.

The three-field gate is mandatory. Stage 2 intentionally returns
`success=true, done=false, target_valid=false` when no safe pose exists. A
default protobuf `Pose` has quaternion `(0,0,0,0)`. Sending it to Move Robot
produces:

```text
Failed to create Pose from proto which contains a non-unit quaternion with
norm(quat) == 0.000000
```

That error is a missing behavior-tree gate, not an IK failure in Move Robot.
It has already occurred on SC.

The deployed process currently packs the seven scalar fields from
`result.target`. Keep that interface. `result.survey_pose` contains the same
pose, but changing the Flowstate wiring is not required.

## 4. Flowstate parameter and result interface

### 4.1 Parameters

Proto scalar zero means "use the code default" for the active fields:

| Field | Default / current behavior |
| --- | --- |
| `min_contrast` | 30 grey levels |
| `margin_px` | 15 px usable-image edge margin in Stage 1 |
| `ignore_bottom_frac` | 0; optional exclusion in addition to the calibrated mask |
| `step_m` | 0.04 m base translation scale |
| `backoff_step_m` | `step_m` |
| `timeout_seconds` | 10 s for fresh camera/TF/force acquisition |
| `min_area_frac` | 0.005 Stage-1 board evidence |
| `max_force_n` | 18 N absolute wrist-force guard |
| `max_moves` | Deprecated and ignored |
| `max_speed_mps` | 0.05 m/s Cartesian setpoint speed |
| `publish_hz` | 20 Hz minimum-jerk command publication |
| `settle_tolerance_m` | 0.008 m |
| `move_timeout_seconds` | 6 s per internal move |
| `max_travel_m` | Deprecated compatibility input; ignored as a policy limit |
| `force_delta_n` | 5 N change from initial force baseline |
| `search_timeout_seconds` | 60 s default, validated/logged but not used as the current aggregate deadline |
| `max_displacement_m` | Deprecated compatibility input; ignored as a policy limit |
| `angular_step_rad` | 0.10 rad Stage-1 angular scale |
| `max_angular_displacement_rad` | Deprecated compatibility input; ignored as a policy limit |
| `max_angular_travel_rad` | Deprecated compatibility input; ignored as a policy limit |
| `context_margin_frac` | 0.05 Stage-1 context pad |
| `min_detail_area_frac` | 0.06 requested, while the active planner floors its goal area at 0.26 |
| `min_rectangularity` | 0.50 |
| `stable_frames` | Validated in 1..5 and logged; active planner confirmation counts are fixed internally |
| `max_angular_speed_rps` | 0.30 rad/s; direct joints are capped at 0.20 rad/s |
| `settle_orientation_tolerance_rad` | 0.05 rad |
| `target_center_tilt_deg` | Retained compatibility field; not read by the current geometric path |
| `center_tilt_tolerance_deg` | Retained compatibility field; not read by the current geometric path |
| `ivm_min_center_board_area_frac` | Retained compatibility field; not read by the current geometric path |
| `ivm_max_center_board_area_frac` | Retained compatibility field; not read by the current geometric path |
| `survey_target` | Target enum 0..3 described above |

The deprecated motion-envelope inputs are parsed and validated where required
for descriptor compatibility, then replaced with infinity. The URDF/controller,
fresh measurement, force, cancellation, and per-move timeout remain the Stage-1
motion authority.

### 4.2 Results

The fields that control the process are:

| Result | Meaning |
| --- | --- |
| `success=true, done=true, target_valid=true` | Stage 2 published a valid survey pose |
| `success=true, done=false, target_valid=false` | Expected geometric/perception rejection; cleanup and retry or branch |
| `success=false` | Sensor, Stage-1 motion, force, controller, or runtime failure |
| `force_abort=true` | The absolute or baseline-delta wrist-force guard stopped motion |
| `component_coverage_ready=true` | Same terminal state as a valid geometric survey pose |
| `target` | Seven scalar Cartesian fields used by the existing Flowstate packer |
| `survey_pose` | Equivalent `intrinsic_proto.Pose` |
| `last_action`, `dx/dy/dz`, travel fields | Diagnostics only |

The complete result diagnostics are:

| Fields | Purpose |
| --- | --- |
| `message` | Human-readable terminal/rejection reason |
| `steer_camera`, `edges`, `area_frac`, `rectangularity`, `view_quality` | Latest useful image report or Stage-2 source camera |
| `num_cameras`, `seen` | Fresh-frame/evidence summary |
| `dx`, `dy`, `dz`, `backoff` | Last internal Cartesian diagnostic |
| `force_n`, `force_abort` | Force guard state |
| `moves_executed`, `travel_m`, `angular_travel_rad`, `moved`, `rollback_count` | Stage-1 motion accounting |
| `target_frame` | `base_link` when the target is valid |
| `elapsed_seconds`, `last_action` | Timing and state diagnostics |

`_stage2_not_done()` deliberately does not throw. It returns `success=true` and
`done=false` so the process can release the controller and decide whether to
retry.

The current Stage-2 diagnostics are still named `"SFP Stage 2..."` and
`"sfp_survey_pose_published"` even for NIC and SC. That is a naming bug, not
evidence that the wrong sector ran. The target-specific `survey_target`, sector
geometry, obliquity, cross tilt, and bore log lines identify the real policy.

The `search_timeout_seconds` proto comment is also stale. The value is
validated in `[10,60]` and logged, but current Stage 1 passes
`deadline_reached=False` to the planner. Stage 1 has no aggregate wall-clock
deadline; it is bounded by planner stall rules, per-move timeouts, cancellation,
force, and controller checks. Stage 2 likewise has no aggregate compute
deadline.

## 5. Stage 1 — expose the insignia (BROKEN — being replaced, see §19/§20)

> This section describes the Stage 1 that **is deployed today**. It is retained
> because you have to understand it to replace it safely, not because it is
> correct. It fails structurally, not by mistuning; §19 has the hardware
> evidence and §20 the replacement plan.


Every invocation is serialized with an execution lock. A second simultaneous
call returns normally with `"another board-visibility invocation is still
running"`.

Each Stage-1 iteration:

1. grabs fresh frames from the three approved cameras;
2. applies each camera's calibrated gripper mask;
3. computes a `MaskReport` with board evidence, clipping edges, area,
   rectangularity, centre error, long-axis direction, purple insignia, context,
   and gripper clearance;
4. checks any available force sample;
5. immediately hands the freshest triplet to Stage 2 if any calibrated camera
   contains a complete, unobstructed insignia;
6. otherwise asks `AdaptiveViewpointPlanner` for one bounded measured action.

The fallback planner is a strict phase machine:

```text
ACQUIRE (J1 sweep)
-> CENTER (J1 proportional centring)
-> ALIGN (J6 board-long-axis alignment)
-> LEVEL (Cartesian correction, primarily J2-J4)
-> ASCEND (clearance/scale and synchronized three-camera confirmation)
-> DONE
```

J1 and J6 corrections are small, measured transactions. Cartesian corrections
use minimum-jerk profiles and remeasure after every move. The wrapper learns
image-motion polarity from the next fresh frame, can reverse a wrong-way
vertical correction, and can recenter/relevel after Cartesian IK changes J1 or
J6.

For all current survey targets, this full sequence is only a fallback. In the
normal case the purple insignia is already complete at iteration 0 and Stage 1
commands no motion.

Perception-only evaluation may continue without a simultaneous wrench sample.
Any real Stage-1 motion requires a fresh force sample no older than 0.5 s.
Defaults are:

- absolute force limit: 18 N;
- force change from initial baseline: 5 N;
- Cartesian speed: 0.05 m/s;
- direct-joint angular speed: at most 0.20 rad/s;
- general angular setpoint speed: 0.30 rad/s;
- one-move timeout: 6 s;
- measured settle tolerances: 8 mm and 0.05 rad.

Deprecated start-relative travel/displacement proto fields are accepted for
binary compatibility but are not policy termination conditions. The motion
remains incremental, measured, cancellable, force-guarded, and controller
limited.

`execute()` always calls `RobotMotion.prepare_controller_handoff()` in a
`finally` block. This publishes a final measured-state hold target. It does not
release the controller bridge; the following Flowstate **Switch To Default
Controller** node does that.

## 6. Shared Stage 2 — insignia to survey pose

Stage 2 commands no motion and does not require force feedback.

### 6.1 Input validation

Stage 2 requires all three fresh images and all three CameraInfo messages. It
rejects:

- a missing image or calibration;
- an image/CameraInfo frame outside the exact camera allowlist;
- CameraInfo dimensions different from the image;
- malformed focal lengths or distortion;
- unavailable timestamp-bound TCP/camera TF.

Every transform is resolved at that camera image's timestamp. The code never
uses latest TF for PnP or candidate projection.

### 6.2 Board pose from the purple insignia

`detect_insignia_polygon()` finds the asymmetric magenta bracket. The Stage-2
landmark is its complete bounding rectangle plus material-mask centroid.

`estimate_board_pose_from_insignia()`:

1. orders the four image corners;
2. PnPs them against CAD `INSIGNIA_RECT_CORNERS`;
3. tries both windings and all four cyclic correspondences;
4. rejects a backside solution by requiring the camera to lie above the CAD
   board face;
5. uses the asymmetric material centroid to select the correct near-square
   rotation/mirror hypothesis;
6. applies the default gates:
   - reprojection RMS at most 8 px;
   - ambiguity ratio at least 1.2;
   - centroid/disambiguation error at most 40 px.

The result is the full six-degree-of-freedom `base_T_board`. All sector offsets
and tilt directions are expressed in the board frame and therefore rotate and
translate with the board. No SC or NIC camera displacement is a fixed world X/Y
offset.

A complete insignia in a side camera is sufficient. Accepted estimates from
multiple cameras must form a cluster within 5 cm and 8 degrees. If multiple
accepted estimates have no two-member consistent cluster, Stage 2 refuses to
guess. Within the largest cluster it prefers the center-camera estimate,
otherwise the lowest reprojection/centroid error.

### 6.3 Camera rig and gripper exclusion

The skill derives `tcp_T_cam` independently for all three cameras from
timestamped TF. Candidate construction specifies a desired center-camera
optical pose, then exactly inverts that extrinsic:

```text
base_T_tcp = base_T_center_camera * inverse(tcp_T_center_camera)
```

The full rig is then projected from that TCP. A pose is not accepted merely
because the target fits the center image.

Each camera has a calibrated binary gripper silhouette. Stage 2 creates a
`GripperExclusion` with 32 px uncertainty around that silhouette. The sector's
filled projected convex hull must clear both image boundaries and the gripper
exclusion. All three target policies now add a further 25 px minimum clearance
(SFP moved 40 -> 25 on 2026-07-26; the 40 px default survives only for callers
that do not override it).

### 6.4 Candidate grid and deterministic selection

The general search can vary:

- reference-camera standoff:
  `0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.64,
  0.66, 0.68, 0.70, 0.73, 0.76, 0.80, 0.85, 0.90, 1.00, 1.15,
  1.25 m`;
- board-X and board-Y aim offsets, normally `-60, -30, 0, +30, +60 mm`;
- optical roll;
- either isotropic near-normal obliquity or a target-specific directional tilt.

SC overrides the standoff to a `0.55 / 0.58 / 0.60 / 0.62 m` ladder and takes
the closest feasible rung. NIC and SC use 24 optical-roll samples at 15-degree
intervals. SFP uses the default seven roll samples.

For every candidate the search checks, in order:

1. base height and broad reach pruning;
2. Cartesian orientation change from the current TCP;
3. target-specific image-formation quality, if any;
4. sector framing and gripper clearance in the center camera;
5. the same gates in both side cameras;
6. live-seeded IK, physical winding, self-clearance, arm-in-view, and relative
   joint motion.

The cheap SC bore gate runs before expensive full-resolution gripper-mask
clearance. Camera checks stop at the first failed required camera. This
short-circuit preserves the accepted set but reduces Stage-2 time.

Standoff is the primary visual objective: closest for SFP/SC and farthest for
NIC. Within a standoff, target-specific view quality, directional angle or
overhead alignment, clearance, and Cartesian motion build the geometric rank.
When live joint motion is available, IK-valid candidates in the same
perception-equivalent plateau are reranked by worst-joint and total travel.

### 6.5 Workspace and component clearance

Before publication the TCP must satisfy:

- base-link Z at least 0.02 m;
- base-origin norm at most 1.20 m;
- board-normal clearance at least 0.2393 m.

The last value is the tallest NIC port plane, 0.1793 m above the board, plus
0.060 m tool margin. It prevents publishing a low SC-sector TCP that the
downstream collision planner would have to refuse.

The source contains `sampled_cartesian_path_is_safe()`, but the current Stage-2
integration does not pass a path callback into `search_survey_pose`. Do not
claim that the skill proves the downstream Cartesian transit safe. It validates
the endpoint and a predicted IK branch; Move Robot remains responsible for the
real path.

## 7. Live-seeded UR5e IK and self-visibility

The endpoint reachability gate uses the UR5e chain from the production MuJoCo
model:

- classical UR5e DH parameters;
- exact analytic enumeration of shoulder × elbow × wrist branches;
- physical joint limits: elbow ±180 degrees, the other joints ±360 degrees;
- no custom task-specific absolute joint window.

`UR5eArm.autocalibrate()` compares one static pair of measured arm joints and
`base_T_tcp` against several base conventions. The measured workcell convention
has selected:

```text
base=Rz180 tool=197.1mm axial=1.00
```

The recovered flange-to-TCP transform is used to solve every candidate.
`solve_ranked(pose, seed=live_joints)` lifts co-terminal solutions to the
physical winding nearest the live six-joint seed.

Two separate robot-visibility checks run on every physical branch:

1. `UR5eArm.self_clearance()` treats the three camera locations as flange
   probes and requires 140 mm clearance from the arm links;
2. `_arm_clear_of_own_cameras()` samples the upper-arm and forearm capsules,
   projects them into every wrist camera, grows their image footprint by link
   radius at depth, and rejects a branch if the robot would appear in any image.

The second check addresses the actual SC yaw-70 failure where the upper arm
occupied the center camera despite a nominally good top-down view. It is
unit-tested but has not yet been validated on hardware. This is the highest
priority hardware check.

If live IK is unavailable:

- SFP and NIC fall back to the older 0.85 m base-origin reach sphere;
- SC returns `done=false`, because its relative-motion and arm-clear guarantees
  require a valid live-seeded IK model.

The endpoint relative-motion caps are:

- SC: worst joint delta at most 185 degrees;
- SFP/NIC: worst joint delta at most 225 degrees when live IK is active.

These are deltas from the current measured state, not absolute J1-J6 positions.
The previous absolute SC window was removed. It rejected a valid chained start
at J4 = -143.9 degrees before evaluating any pose and caused an empty result to
reach the ungated Move Robot node.

## 8. SFP survey policy

### 8.1 Physical target

`STAGED_SFP_MODULE` frames the five loose SFP modules staged on the board.
They occupy five of **six** legal seats at 50 mm pitch spread over **both**
mount rails — board Y `-0.15625`, `-0.10625`, `-0.05625`, `+0.05625`,
`+0.10625`, `+0.15625` (`sfp_module_detail_boxes`). Which seat is empty does
not matter: the outermost seats are occupied, and they are what bind.

The coverage target is `sfp_module_strip_corners()`:

```text
X: 0.030 .. 0.115 m
Y: -0.1125 .. +0.1125 m
Z: 0.010 .. 0.060 m
```

This is the staging/pick strip, not the SFP cages on the NIC cards.

### 8.2 The one-rail bug this replaced (2026-07-26)

The superseded `sfp_sector_corners()` covered `Y: 0.000 .. 0.225` — the **+Y
rail alone**. Its centroid is what the survey aims at, so the aim point sat
112.5 mm off the middle of the module strip and every bit of the search's
framing slack was banked on the +Y side. Hardware returned **4 of 5 modules**;
the missing one was the outer -Y seat. Decoded from that run: detections at
board Y `-0.1066, -0.0565, +0.1066, +0.1566` and board X `+0.0862`.

`sfp_sweep_runner.py` reproduces it. At identical search settings the old box
clips a module in **96 of its 96** found poses — 35 of them holding only *four*
of the six seats — with the worst seat 123.9 px outside the image.

**The fix is placement, not size.** The replacement box has the same 0.225 m
board-Y extent, straddling Y=0. Board X was widened from `0.020..0.090` to
`0.030..0.115` because the detected module bodies sit at board X 0.0862, only
3.8 mm inside the old edge — the transceiver protrudes from its mount origin.
Board-X width costs nothing: 85 mm, 65 mm and 50 mm boxes sweep identically.

Do **not** enlarge the box along Y to "guarantee" the outer seats. Measured:

```text
y_half   found/144   all-5 framed   selected standoff
0.1125      92           92          0.64-0.85 m
0.1450      58           58          0.80-0.85 m
0.1600      35           35          0.85-0.90 m
0.1783       0            -          infeasible
```

A wider box pushes the standoff out and shrinks every module in the image, and
full-strip containment is not reachable at all: clearing the tool silhouette
with a hull that wide needs >=0.85 m, and past ~0.85 m the arm's own links
enter a wrist camera at every roll. The box sets the aim and the standoff; the
sweep's **seat audit**, not the geometry, certifies coverage.

### 8.3 View policy

SFP uses the general near-overhead search:

- all three cameras must frame the complete strip;
- closest feasible standoff wins for maximum component pixels;
- no directional tilt band;
- total reference-camera obliquity at most the general 20-degree cap;
- additional all-camera clearance: **25 px** (was 40);
- Cartesian reorientation cap: **90 degrees** (was 45);
- default seven optical-roll samples;
- live IK ranking when available, sphere fallback otherwise.

The 25 px / 90 deg values are NIC's already-proven ones, adopted for
availability rather than framing: the centred box finds a pose in 58 of 144
swept cases at 40 px / 45 deg and in **92** at 25 px / 90 deg, and all 92 still
frame every module, so the extra 34 are genuine gains rather than weaker views.
25 px is measured against a silhouette `GripperExclusion` has already dilated
by 32 px.

SFP has no recessed-bore depth-quality callback. It relies on strip framing,
gripper clearance, near-overhead conditioning, and reachability.

### 8.4 Current status

**Confirmed working on hardware (2026-07-27).** Sweep result:

```text
92 / 144 poses found, 92 / 92 frame all six seats, 0 clipped
selected standoff:  0.64 .. 0.85 m
worst seat margin:  118.5 px
```

The 52 no-pose cases are IK / arm-in-view failures concentrated on the
`home + J6 +90 deg` Stage-1 exit, not framing failures. That is a Stage-1
problem, and §19 is where it belongs.

The downstream SFP/NIC-card filtering code is a Flowstate code-execution node,
not part of this repository's skill bundle. The previously working version
fits detections along an orientation-invariant 3-D line rather than fixed world
axes. The user reported that fix working; do not replace it casually. Its
latest exact paste-ready source is not checked into this repository, which is a
handoff risk.

## 9. NIC destination policy

### 9.1 Physical target

`NIC_SFP_DESTINATION` targets the ten recessed SFP cages on five NIC cards.
The board-frame sector is:

```text
X: -0.1124 .. -0.0652 m
Y: -0.1978 .. -0.0138 m
Z:  0.1250 ..  0.1850 m
```

The box is centered on the port entrances, not on the card bodies. The ten
mouths span about 160 mm. Each mouth is 16 × 12 mm at the top of a 45.8 mm
recess. Its axis is only about 0.7 degrees from the board normal.

The limiting half-aperture is 6 mm, so the approximate unobstructed cone is:

```text
atan(6 / 45.8) = 7.5 degrees
```

### 9.2 View policy

NIC must look nearly straight down the cage bores:

- no cross-rail or side tilt;
- reference-camera obliquity at most 2 degrees;
- all three cameras frame the sector;
- additional clearance: 25 px;
- farthest feasible standoff wins;
- 24 optical-roll samples at 15-degree intervals;
- Cartesian reorientation cap: 90 degrees.

The far-view preference is physical, not aesthetic. The outermost port is
about 81 mm from the aimed center. To keep it inside the 7.5-degree cone:

```text
standoff >= 0.081 / tan(7.5 degrees) ~= 0.62 m
```

The 25 px requirement is applied after the gripper silhouette has already been
dilated by 32 px. Offline it leaves substantially more than 25 true pixels
between the cards and actual gripper pixels.

### 9.3 Do not repeat the NIC tilt experiment

Tilting across the NIC rail was tried. It made the camera see the cage walls
instead of their black interiors and resolved 0 of 10 ports at reachable board
yaws. The bores open along the board normal. Keep NIC near-normal.

### 9.4 Current status

NIC has worked in the three tested board orientations on the preceding
hardware build. The newest working-tree IK/arm-in-view changes have not been
rerun on hardware. Its wider 90-degree Cartesian orientation search and 24
rolls exist because the wrist-camera cluster otherwise remains too close to
the forearm at some board yaws.

## 10. SC destination policy

### 10.1 Physical target

`SC_DESTINATION_PORT` targets five SC adapters:

- `sc_port_0/1/2`: row at board Y = +0.0295 m;
- `sc_port_3/4`: row at board Y = +0.0705 m;
- legal centers span board X = -0.135 .. -0.020 m;
- mouth plane: board Z = 0.0301 m.

The conservative framed sector is:

```text
X: -0.152 .. -0.003 m
Y:  0.005 ..  0.095 m
Z:  0.020 ..  0.035 m
```

Each mouth is approximately 7.6 × 22.4 mm and 15.64 mm deep. The long face runs
along board Y. The narrow 7.6 mm dimension runs along board X. The bore opens
along the board normal.

### 10.2 Why SC is deliberately oblique

Hardware showed that a nearly head-on image often detected only two or four of
five ports even though all five blue mouths were visible. The working visual
cue appears to be separation between neighboring blue adapters plus a visible
black interior/depth displacement.

The view must reveal depth from the adapter's **long face**, not look across its
short end. Since the long face runs along board Y, the camera displacement is
explicitly along board X, the in-plane normal to that face. This axis is
transformed through `base_T_board`, so it follows arbitrary board yaw and tilt.
It is never a fixed base X/Y direction.

Current SC view policy:

- center-camera cross tilt: **16 to 20 degrees** along ±board X;
- along-board-Y tilt: at most 2 degrees;
- both board-X sides are searched;
- standoff: **a 0.55 / 0.58 / 0.60 / 0.62 m ladder**, closest feasible wins;
- all three cameras frame the complete SC sector;
- additional clearance: 25 px;
- 24 optical-roll samples at 15-degree intervals;
- Cartesian orientation search envelope: 180 degrees;
- live IK required;
- worst relative joint delta: at most 185 degrees.

**The long-face (board-X) direction is correct and hardware-proven. Do not add
an along-board-Y tilt to buy depth.** It is geometrically tempting — the wide
bore axis has a 35.61 degree cone against board X's 13.66, and it measured
~2.8x the depth cue and is rail-invariant — but it is not the face the
detector was validated against, and it was tried and reverted on 2026-07-27.
Geometric headroom is not evidence about what the model keys on.

Relaxing all-camera framing to get closer was tested and reverted. At the close
candidate the side cameras had gripper clearance of -13 to -32 px while the
center camera reported +58 px. In-frame is not unoccluded, and user-confirmed
IVM needs the full sector in all three cameras.

### 10.3 SC bore/depth gate

Framing alone is insufficient. The three camera origins are separated around
the wrist, and roll changes which rectangular mouth dimension each side camera
looks across.

Stage 2 evaluates six conservative mouth samples:

```text
X = -0.135, -0.0775, -0.020 m
Y = +0.0295, +0.0705 m
Z = +0.0301 m
```

For every mouth, at least two of the three cameras must:

1. have a nonnegative rectangular line-of-sight margin through a
   **7.6 mm** × 11.2 mm aperture tolerance over 15.64 mm depth
   (`SC_BORE_X_TOLERANCE_M`);
2. retain at least 3.0 px of projected mouth-to-back-center displacement.

**The narrow-axis tolerance is the acceptance criterion, and it is what used to
cap the depth.** At the mouth *half* width (3.8 mm) the ray must still reach
the back-plane **centre**, which is a hard 13.66 degree ceiling on the
long-face angle. Since the cue is `f*depth*tan(theta)/dist`, that ceiling *was*
the flakiness the user reported as "works about 75% of the time, breaks when an
adapter slides along its rail": the cue never exceeded 3.3-4.5 px.

At the *full* mouth width the criterion becomes "a displaced dark strip is
still visible", which is what the estimator actually reads. The back plane is
partly occluded rather than centred: at 18 degrees the strip is
`7.6 - 15.64*tan(18) = 2.5 mm` wide by 22.4 mm long, and its displacement more
than doubles. Measured worst mouth over the whole legal rail, 24 placements:

```text
band     criterion            worst cue   dark strip
10-13    back centre           3.34 px      4.4 mm
14-18    displaced strip       7.13 px      3.1 mm
16-20    displaced strip       7.99 px      2.5 mm
```

This is not a relaxed *safety* gate. Framing, gripper clearance, live-seeded IK
and arm-in-view are all unchanged and still hard.

Why the cue collapses when an adapter slides: the search aims at the sector
centroid, so a mouth offset `delta` along the displacement axis is seen at
`atan(tan(theta) - delta/d)`, not at `theta`. Over the 115 mm of legal rail
travel at 0.62 m that swings the per-port angle about 10 degrees. Measured on
the superseded pose, the centre camera ran 7.9 degrees at one end of the rail
to 18.0 at the other — **past its own 13.66 degree cone over half the rail** —
while the side cameras dropped to 2.6 degrees and saw almost no depth.

Always score SC depth on the **worst mouth anywhere on the legal rail**, not on
`sc_bore_sample_points()`'s six positions. A tuning that looked like +39% on
the six samples was +10% on the real metric.

All three cameras still independently pass sector framing and gripper
clearance. Requiring all three cameras to pass the bore gate was too strong and
forced the selected view back toward a visibly head-on 7-8 degrees. Two
depth-capable cameras preserves fused IVM while allowing the intended 10-13
degree view.

At the fixed standoff the selector first maximizes the two-camera depth cue.
Candidates within 0.1 px of the best reachable cue form a perception-equivalent
plateau. It then minimizes relative joint motion.

### 10.4 J6 preference

SC prefers an exact legal ±180-degree J6 target relative to the live start.
This changes the camera-cluster/arm side and is intended to keep the arm and
tool out of the useful view. It is a preference, not an absolute joint target.

The J6 preference may select a candidate whose worst-joint motion is at most 30
degrees above the minimum-motion candidate in the same perception plateau. The
185-degree hard relative cap remains authoritative.

### 10.5 Offline SC sweep

Run:

```powershell
cd C:\Users\anshu\College\aic\aic\flowstate\aic_perception
python test/sc_sweep_runner.py --workers 4
```

The current production-geometry sweep contains 144 cases:

- board yaw: `0, 45, 70, 90, 140, 180, 250, 315` degrees;
- board tilt: `0, 10` degrees;
- board XY placement:
  `(0,0)`, `(+50,+30) mm`, `(-50,-40) mm`;
- three live seeds:
  home, home with J6 +90 degrees, and the exact chained failure
  `[-17.5,-95.8,-19.5,-143.9,82.9,26.8]` degrees.

Latest local result (current policy):

```text
144 / 144 pass
selected standoff: 0.55 .. 0.58 m
two-camera depth cue: 7.3609 .. 8.5487 px
bore margin: 0.03126 .. 0.33149
minimum image/gripper clearance: 25.0088 .. 53.4225 px
selected relative joint-motion range: 30.0818 .. 184.8610 degrees
```

Superseded policy, same harness, for comparison:

```text
144 / 144 pass
selected standoff: 0.62 m in every case
two-camera depth cue: 3.3426 .. 4.4513 px
bore margin: 0.01346 .. 0.26738
minimum image/gripper clearance: 37.8355 .. 73.9906 px
```

Reproduce the old policy with
`--tilt-min-deg 10 --tilt-max-deg 13 --min-standoff-m 0.62
--bore-x-tolerance-m 0.0038`.

The 185-degree cap is measured. Tightening it to 182 degrees loses cases.

**Two things now run with no slack** and want watching on hardware: minimum
gripper clearance is 25.0 px against a 25 px floor (the closer standoff costs
margin), and worst joint travel is 184.9 degrees against the 185 degree cap.

The sweep validates the endpoint branch predicted by this skill. It does not
prove that the downstream Move Robot node will choose that same branch.

### 10.6 Current SC hardware evidence

Progressively stronger views improved IVM from two strong detections, to four,
to a recent run with all five strong physical candidates:

```text
score 0.874 at [165.0, 29.5, 1156.4] mm
score 0.904 at [125.0, 29.6, 1156.2] mm
score 0.906 at [124.9, 70.4, 1156.3] mm
score 0.954 at [ 85.0, 29.5, 1156.6] mm
score 0.880 at [165.0, 70.4, 1156.4] mm
```

This proves that run's IVM found five physical ports. The following filter
created only four labels and ignored the fifth; that was a downstream labeling
failure, not an IVM miss.

**2026-07-27: the 16-20 degree / full-width-tolerance policy is confirmed
working on hardware.** SC and staged-SFP Stage 2 are both considered good; the
remaining SC failures in that session were all traced to Stage 1 (§19), not to
the survey view.

The current arm-in-view rejection has not been independently isolated on
hardware. At board yaw 70 degrees the previously deployed endpoint put the
upper arm in the center camera. Validate all three images, not only the final
TCP geometry.

## 11. Motion-path limitation

The skill predicts and logs a good IK branch, but publishes only a Cartesian
pose. A Cartesian pose does not encode shoulder, elbow, or wrist winding.
Move Robot can choose a different co-terminal/collision-free branch.

Observed evidence:

- an earlier NIC move made an approximately 360-degree J6 swing;
- an SC move traveled 0.374 m but generated 1193 trajectory points and took
  29.44 s, versus about 91-111 points and 2.75-3.16 s for ordinary moves.

Do not add absolute J1-J6 bounds based on previously observed endpoints. They
constrain objective position rather than motion from the live origin and have
already caused a valid chain to fail.

What the skill can do:

- reject Cartesian endpoints with no live-seeded physical IK;
- prefer endpoints whose predicted branch has smaller relative motion;
- prefer the SC half-turn wrist family only inside a bounded plateau;
- reject predicted branches with camera/forearm collision or arm-in-view;
- limit endpoint Cartesian orientation change.

What the skill cannot do through the current interface:

- send the selected six-joint branch to Move Robot;
- seed Move Robot's planner;
- constrain its exact joint-space path;
- guarantee Move Robot will not choose a co-terminal winding.

On the Flowstate side, remove the obsolete custom absolute position bounds,
retain conservative velocity and acceleration limits, inspect the planned path,
and use any Move Robot option that explicitly limits travel relative to the
current state if one is available. Do not represent measured start-relative
motion as fixed world joint limits.

## 12. Downstream IVM and filters

Cloud IVM is outside this skill and cannot be changed. It is nondeterministic,
so camera quality must be robust rather than merely passing once.

Keep three failure classes separate:

1. **No model result:** for example `AccountsTokensService` /
   `DEADLINE_EXCEEDED`. This says nothing about the view.
2. **Model result with fewer than five physical candidates:** this is an
   actual perception/view failure.
3. **Five physical candidates but filter failure/mislabel:** IVM worked; debug
   the code-execution node.

The previous minute-long delay was mostly Stage-2 exhaustive search, not cloud
IVM. One measured trace was:

```text
Stage 2: 63.83 s
IVM total: 6.15 s
IVM request/inference: 5.71 s
```

Search short-circuiting and the fixed 0.62 m SC standoff reduced a
representative local case from about 8.0 s to 3.6 s without changing the
selected pose. Hardware timing after the latest build is still unmeasured.

### 12.1 Current SC target filter

`docs/reference/filter_estimates_sc_node.py` is a Flowstate code-node body, not
Python imported by v4. Its current contract is:

- input: `params.pose_estimates` and `params.selected_module_name`;
- valid target names: `sc_port_0` through `sc_port_4`;
- minimum score: 0.4;
- deduplication radius: 10 mm;
- no spacing, alignment, coplanarity, ambiguity, or metric 3/2-layout
  rejection;
- output: exactly one repeated Pose in `output.sc_ports`.

Returning exactly one pose is required by the current Flowstate
`ReturnValue`. Returning all five made the downstream Create Object expression
expect/index five beliefs and created every detected object instead of the
requested target.

The current positional implementation averages detection orientations to
obtain signed board X/Y, splits detections at the largest projected board-Y
gap, sorts each row by board X, and assigns:

```text
lower board-Y row: sc_port_0, sc_port_1, sc_port_2
upper board-Y row: sc_port_3, sc_port_4
```

This is not solved on hardware. In the five-candidate run above,
`axes_from_orientation=True`, but the averaged IVM orientation axes were not
the physical board axes. The code made a 4/1 positional split, labeled four
ports, and ignored `[165.0,70.4,1156.4]`.

Next-session filter work should preserve the requested permissive behavior
(do not reject metric spacing/alignment) while making axis/row assignment
robust to unreliable IVM orientations. At minimum, print every raw pose,
quaternion, score, transformed coordinates, chosen axes, split, labels, and
ignored candidates before changing it. The physical five-point layout in the
latest log is an exact 3/2 grid and is the regression case to add.

The debug-instrumented IVM/filter attachment from this session is outside the
repository:

```text
C:\Users\anshu\.codex\attachments\
c8b4859f-c438-40d6-8ec0-64e5da2c55ec\pasted-text.txt
```

It prints raw IVM contents but is based on the older strict init-only 3/2 fit.
Use it for diagnostics, not as the authoritative target selector.

### 12.2 SC filter harness

Run:

```powershell
cd C:\Users\anshu\College\aic\aic
python docs/reference/filter_estimates_sc_node_test.py
```

The harness is useful but must include the latest five-candidate orientation
regression before it can be treated as hardware-representative.

## 13. Common failures and their meaning

| Symptom | Meaning / first check |
| --- | --- |
| `norm(quat)==0` in Move Robot | Missing `success && done && target_valid` gate |
| `done=false` with `success=true` | Stage 2 deliberately found no safe publishable pose; inspect reason |
| `arm already in use` | Default-controller cleanup did not run after the skill |
| `IK not computable` in Move Robot | Downstream planner rejected/failed the Cartesian endpoint; compare live IK log and actual planner branch |
| Violent or nearly 360-degree wrist transit | Move Robot selected a different co-terminal joint path |
| `Only 2/4 distinct SC-port candidates` | IVM actually returned fewer than five physical ports |
| Five strong SC candidates but only four labels | Current downstream positional-axis bug |
| `No valid 3-port/2-port layout` | Old strict SC filter, often including a low-score false positive or fixed-axis error |
| `AttributeError: pose_estimates` on output | Wrong Flowstate return field; current SC output is repeated `sc_ports` |
| Create Object index error `index=1 size=1` | Target filter correctly returned one pose but downstream still indexed it as a five-pose list |
| Skill info service / registry unavailable | Skill pod/service startup or registry problem, before skill execution; inspect installed asset and pod logs |
| IVM token `DEADLINE_EXCEEDED` | Cloud/auth service failure; no perception conclusion is possible |
| **All three sectors return ~1 of 5 at once** | Not a per-sector bore/coverage bug — the survey pose was never correctly positioned. Go to §19 |
| `wrist force guard triggered while settling` repeating | Stage 1 is wedged, likely in contact. Identical camera readings across attempts confirm the arm is not moving (§19.1) |
| `logo=False` in all three cameras | Stage 1 has no gradient to follow and cannot recover on its own (§19.2) |
| `long_ratio=1.00` with `long_axis_error=+0.0deg` | Degenerate frame-aligned `minAreaRect` from a clipped mask; the orientation is not a measurement |
| `total` joint travel huge while `max` passes | Only the worst joint is capped; total is ungated (§18 item 2) |

## 14. Tests and validation

Local Windows Python has `numpy`, `cv2`, and `pytest`; there is no local pixi
environment.

Run the full package suite from the real repository:

```powershell
cd C:\Users\anshu\College\aic\aic\flowstate\aic_perception
python -m pytest test/ -q
```

Current result:

```text
286 passed in 65.11s
```

Run the two production sweeps separately. Both take minutes and are
pre-hardware validation tools, not part of the unit suite:

```powershell
python test/sc_sweep_runner.py --workers 8
python test/sfp_sweep_runner.py --workers 8
```

Run the code-node harness from the repository root:

```powershell
cd C:\Users\anshu\College\aic\aic
python docs/reference/filter_estimates_sc_node_test.py
```

Current results:

```text
unit suite:  286 passed
SC sweep:    144 / 144 found, depth cue 7.36 .. 8.55 px
SFP sweep:   92 / 144 found, 92 / 92 frame all six seats, 0 clipped
filter_estimates_sc reference harness: PASS
```

`sfp_sweep_runner.py` carries an independent **seat audit** that is the point
of the harness: `search_survey_pose` only guarantees the coverage target it was
handed is framed, so the audit separately projects all six legal seats into all
three cameras and reports how many survive. Reproduce the one-rail regression
with `--coverage sector`.

Before believing a shared-search change:

1. run the full unit suite;
2. run all 144 SC scenarios and all 144 SFP scenarios;
3. retain all three SFP/NIC orientations already known to work;
4. test the exact chained live joint seed;
5. capture all three actual camera frames;
6. compare the published-pose and predicted-joint log lines;
7. inspect the downstream Move Robot trajectory rather than only its endpoint.

Score per-sector view quality on the **physical worst case over the component's
legal placement range**, not on whatever fixed samples the gate happens to use.
Both bugs fixed this cycle hid behind a metric that averaged over the wrong
thing.

## 15. Build and install

Build from WSL/Linux AMD64. The known workspace flow is:

```bash
cd /mnt/c/tmp/ws_aic_phase1
rsync -a --delete --exclude .git \
  /mnt/c/Users/anshu/College/aic/aic/ src/aic/
find src/aic/flowstate -type f \( -name '*.py' -o -name '*.sh' \) \
  -exec sed -i 's/\r$//' {} +
INBUILD_BIN=$PWD/inbuild \
  bash src/aic/flowstate/scripts/build_check_board_visibility_skill.sh
AIC_SOLUTION=162d7a70-b696-4260-974d-fdae049e6eaa_BRANCH \
  bash install_skill.sh
```

Always pass `AIC_SOLUTION` explicitly. It persists in the shell, and a second
solution ID has been used during this work. Confirm the active Flowstate
solution/cluster before installing; do not infer it from an older terminal.

The build script:

1. builds the Linux/AMD64 skill image;
2. verifies the image lifecycle labels are exactly
   `ai.intrinsic.asset-id=ai.tar2.check_board_visibility_skill_v4` and
   `ai.intrinsic.skill-image-name=check_board_visibility_skill_v4`;
3. imports OpenCV and the generated proto inside the image;
4. cold-starts the skill twice and requires the
   `"gRPC server listening"` log on both starts;
5. exports the OCI image and descriptor set;
6. bundles them with `inbuild`;
7. requires the bundle manifest to embed
   `check_board_visibility_skill_v4.tar`;
8. prints SHA-256 hashes and bundle contents.

The label checks were added after a repeatable lifecycle failure: a freshly
uploaded pod worked, but stopping and restarting the solution left the v4
service unavailable on port 8003 until the asset was deleted and reinstalled.
The leading repository-side cause was concrete. The old image advertised
`ai.intrinsic.asset-id=ai.tar2.check_board_visibility_skill` (missing `_v4`)
and omitted `ai.intrinsic.skill-image-name` entirely, while the installed
manifest and Kubernetes service use `check_board_visibility_skill_v4`.
Intrinsic's supported skill image template requires both labels. Keep the
manifest, asset-id label, and image-name label aligned.

The first label-only rebuild was still incomplete: the bundle embedded
`check_board_visibility_skill.tar` while the corrected image label said
`check_board_visibility_skill_v4`. `SkillManifest.assets.image_filename` is
the platform's image locator, so an installed asset could exist without a
reconciled skill workload. The build now gives the output directory, OCI tar,
descriptor, bundle, and image-name label the same v4 basename and rejects a
bundle that does not contain `check_board_visibility_skill_v4.tar`.

The subsequent `+dbec2569...` bundle proved that labels plus filename were
still not the whole identity contract: after restart the asset was installed,
but the cluster had no v4 skill log target/workload. Its OCI metadata still
advertised `RepoTags=["flowstate:check-board-visibility-v4"]` and runtime
`SKILL_NAME=check_board_visibility_skill`. Intrinsic's supported builder uses
the exact `<skill_package>:<skill_name>` tag and the manifest skill name.
The current build therefore uses
`aic_perception:check_board_visibility_skill_v4`, passes
`SKILL_NAME=check_board_visibility_skill_v4`, and keeps the historical config
basename separate as `SKILL_CONFIG_NAME=check_board_visibility_skill`.

The smoke test also owns SIGINT/SIGTERM and stops gRPC plus the ROS executor in
order. The prior `RCLError: context is not valid` appeared only after the
eight-second smoke timeout had already observed port 8003; it was a forced
shutdown race, not a startup failure.

A real solution
stop/start after installing the corrected bundle is still the authoritative
end-to-end validation.

The output bundle is:

```text
images/check_board_visibility_skill_v4/
check_board_visibility_skill_v4.bundle.tar
```

The user will build and push/install. Do not deploy automatically.

## 16. Logs to capture on every hardware run

The most useful skill lines are:

```text
active search parameters: ...
insignia exposed in a calibrated camera; handing off to geometric Stage 2
arm IK joint-motion gate active: base=Rz180 tool=197.1mm axial=1.00 ...
SC survey image geometry required_depth_cameras=2
  bore_margin_2cam=... depth_cue_2cam=...px
survey IK motion current_deg=[...] target_deg=[...] delta_deg=[...]
  max=... total=... preferred_j6_deg=... j6_error=...
SFP Stage 2 published survey pose source=...
  target=(...) standoff=... min_clearance=... view_quality=...
  obliquity=... cross_tilt=... along_tilt=...
  joint_max=... joint_total=...
```

`obliquity` is measured on the center camera optical axis, not TCP +Z. The
wrist cameras are pitched about 15 degrees from the tool axis, so measuring
TCP orientation gives the wrong view angle.

Also capture:

- the three camera images at the final survey endpoint;
- Move Robot trajectory point count and duration;
- IVM request start/end and every raw returned estimate;
- filter score threshold, deduplicated list, axes, row split, label map,
  ignored list, and selected output.

## 17. Do not repeat these experiments

1. Do not relax SC to center-camera-only framing for a closer view. It put the
   tool on top of the ports in both side cameras.
2. Do not add cross-rail tilt to NIC. Its cage bores open along the board
   normal.
3. Do not infer the SC directional axis from the longer 3/2 cluster extent.
   The required displacement is board X, normal to each adapter's board-Y long
   face.
4. Do not use fixed world X/Y for port or card labeling. The board moves,
   yaws, and tilts.
5. Do not add fixed absolute J1-J6 windows to prevent violent motion. Motion
   must be measured from the live state.
6. Do not conclude that a cloud token timeout is an IVM miss.
7. Do not infer occlusion safety from projected points merely being inside the
   image.
8. Do not send a Stage-2 rejection pose to Move Robot.
9. Do not add an along-board-Y tilt to the SC view to buy depth. The long-face
   board-X direction is hardware-proven; the wide axis measured better and was
   still rejected (§10.2).
10. Do not enlarge the staged-SFP coverage box along board Y to "guarantee" the
    outer seats. It pushes the standoff out, shrinks every module, and full
    containment is unreachable anyway (§8.2).
11. Do not trust a board `minAreaRect` orientation when the mask is clipped on
    two or more image edges — `long_ratio ~= 1.00` with
    `long_axis_error = +0.0deg` is the degenerate signature, not a measurement.

## 18. Highest-priority next-session work

**§19 — rebuilding Stage 1 — is the whole job.** Everything below is backlog
behind it, because none of it can be evaluated while Stage 1 cannot reliably
put the insignia in front of a camera.

1. Add the missing Flowstate
   `result.success && result.done && result.target_valid` gate after the
   default-controller switch.
2. Add a **total** joint-travel cap. The 185 degree limit is on the worst joint
   only; `total_motion` is computed for ranking and never gated. A hardware run
   on 2026-07-27 published a pose with
   `delta_deg=[138.0, 16.9, 94.5, 56.1, -176.1, 19.6] max=176.1 total=501.3`,
   which passed. Suggested cap 250-300 degrees, swept to confirm no coverage
   loss.
3. Add the latest five-candidate hardware regression to the SC filter harness,
   then fix row/axis recovery without restoring metric alignment or spacing
   rejection.
4. Investigate `UR5eArm.autocalibrate()` instability: the same session logged
   `tool=197.1mm` in four invocations and `tool=201.3mm` in a fifth. A 4.2 mm
   shift in the flange-to-TCP estimate moves every projected candidate.
5. Verify that the obsolete absolute J1-J6 position bounds are removed from
   the SC Move Robot segment, then inspect actual planned relative travel.
6. Measure Stage-2 wall time on the deployed optimized build.
7. Check the stale proto/manifest descriptions and target-neutralize the
   `"SFP Stage 2"` diagnostics after behavior is stable.

The core rule for future changes is: measure the workcell geometry offline,
sweep the full board/placement/live-start matrix, and then validate all three
real camera frames. A reachable Cartesian endpoint and an in-frame projection
are necessary, but neither proves that IVM has an unoccluded, well-conditioned
view or that Move Robot will take the predicted joint path.

## 19. Stage 1 is broken and is being scrapped

### 19.1 What the hardware run of 2026-07-27 showed

Stage 2 is fine. Stage 1 is not, and it fails in a way that makes every
downstream result meaningless. Timeline from one session (`01:50`-`02:10`):

```text
01:55:10  force=0.00N   done=True   SC pose published, insignia strong
                                    (logo_area 0.0097 .. 0.0135)
01:58:48  force=13.78N  done=False  169 poses framed, none IK-valid
01:59:05  force=13.68N  done=True   published
01:59:23  force=13.59N  done=True   published, but the pose needed
                                    max=176.1deg TOTAL=501.3deg of joint travel
02:01:12  force=17.79N  success=False  wrist force guard triggered
02:04:31  force=14.73N  success=False  ... while settling
02:07:58  force=17.06N  success=False  ... while settling
02:10:51  force=14.31N  success=False  ... while settling
```

After the 501-degree reconfiguration at `01:59:23` the arm never recovered.
From `02:01` onward **all three cameras report `logo=False`** — the insignia is
not visible anywhere — and the camera readings are byte-identical across the
four remaining attempts (`area=0.365`, `center=(0.474,-0.325)`, ...). The arm
is wedged. Stage 1 tries its one corrective (`backoff`, ~0.06 m up) and the
force guard reverses it every time.

The force guard is `abs(|F| - |baseline|) >= 5 N` or `|F| >= 18 N`. 17.79 N is
under the absolute limit, so the delta guard fired: the force climbed 5+ N
during a 0.06 m *upward* move. Gravity norm is rotation-invariant, so that is
not reorientation bias — it reads as genuine contact, consistent with the
captured images where the tool sits on the board.

Downstream, all three sectors collapsed at once: SFP returned 1 of 5, NIC 1 of
5, SC 2 candidates of which one was 260 mm off the board plane
(`z=1436.7 mm`, score 0.117). **When all three targets fail simultaneously the
cause is the view, not any per-sector bore geometry.**

### 19.2 Why the design is wrong, not just mistuned

`AdaptiveViewpointPlanner` is a strict phase machine —
`ACQUIRE (J1 sweep) -> CENTER (J1) -> ALIGN (J6) -> LEVEL (Cartesian) ->
ASCEND -> DONE` — 1614 lines in `viewpoint_search.py` plus the wrapper's
polarity learning, reversal and re-centre logic. Its problems are structural:

1. **It is open-loop about where the insignia actually is.** It never forms a
   hypothesis of the board pose; it nudges and re-measures, so it can only
   hill-climb. When the insignia is out of frame in all three cameras there is
   no gradient at all, which is exactly the wedged state above.
2. **It leans on a degenerate orientation cue.** A board mask clipped on two or
   more image edges yields a frame-aligned `minAreaRect`; the log shows
   `long_axis_error=+0.0deg long_ratio=1.00` on every camera in the stuck runs.
   That is the documented degenerate signature, and phases ALIGN/LEVEL consume
   it as if it were real.
3. **Its corrective moves are unbounded in consequence.** Each is a small
   Cartesian nudge chosen from one camera's error, with no check that the
   resulting configuration is still one from which recovery is possible.
4. **It has no notion of a safe home.** There is no "the search failed, return
   to a known-good observation pose" path, so a bad exit poisons every
   subsequent invocation in the process.
5. **It is the largest and least-tested part of the skill**, and the phase
   interactions (J1 vs J6 at the levelled pose, polarity reversal, rollback)
   have produced repeated field surprises.

Stage 2, by contrast, is deterministic, hypothesis-first, offline-sweepable,
and has been debugged to the point of being trustworthy. **The fix is to make
Stage 1 look like Stage 2.**

## 20. Plan: replace Stage 1 with a deterministic acquisition search

### 20.1 The prior art on `origin/navigate-to-purple`

Branch tip `4a20097`, skill `move_to_board_skill` (asset
`ai.tar2.move_to_board_skill`, v3). Files worth reading first:

- `flowstate/aic_perception/aic_perception/purple_insignia.py` (new, 185 lines)
- `flowstate/aic_perception/move_to_board_skill.py`
- `flowstate/aic_perception/test/test_move_to_board_loop.py` (448 lines)

What it does well, and should be kept:

- **A dedicated purple detector, ROS-free and unit-testable.**
  `analyze_purple()` thresholds HSV `[125,45,45]-[165,255,255]`, morphological
  open/close, largest contour over 100 px, and returns
  `seen / full / edges / area_frac / centroid / center_error`. The band is
  copied from the *proven* `PerceptionInsert._sc_purple_logo_centroid_px`, so
  it is not a new guess.
- **A flat greedy loop instead of a phase machine.** `_execute_inner` scans all
  three cameras, and if any sees purple it drives on purple, else on the board
  mask (`select_work_target`). No J1/J6 special-casing, no polarity learning,
  no rollback.
- **Pure image-plane translation with orientation held fixed.**
  `_center_in_image` turns clipped edges plus centre error into a unit
  (image-right, image-down) direction, maps it to base frame through the
  camera's *actual* TF axes, and issues one 0.03 m `move_smooth`. Constant
  orientation is what removes the J1/J6-at-the-levelled-pose coupling that bit
  v4.
- **An explicit, checkable terminal condition.** `purple_done` = all three
  cameras see the *unclipped* insignia AND the centre camera centroid is within
  10% of image centre. Budget `MAX_CENTER_MOVES = 14`.

What it does **not** solve, and why it is a starting point rather than the
answer: it is still a greedy servo. If no camera sees purple it falls back to
centring the *board*, which is the same gradient-free situation that wedged v4,
and it has no board-pose hypothesis and no reachability gate.

### 20.2 The target design

Make acquisition a **search over commanded poses**, not a sequence of nudges,
reusing the Stage-2 machinery that already works.

**Step 1 — hypothesise the board pose from what is visible.**
The board plate is large and reliably visible (every failing log line still has
`seen=True`, `area` 0.30-0.37); it is the *insignia* that is missing. Recover a
board-pose hypothesis from non-insignia evidence:

- `estimate_board_pose()` already exists in `board_stage2.py:839` and PnPs
  `BOARD_OUTLINE_CORNERS`. **It cannot be used naively** — the plate is clipped
  in exactly these situations, and a clipped mask gives the frame-aligned
  degenerate rectangle described above. Any use must reject `long_ratio ~= 1.00`
  and clipped-on-2+-edges masks outright.
- Better: use the coloured landmarks that stay in frame. The blue SC adapters
  and the green NIC cards are strong, well-separated, and at known board-frame
  positions (`sc_sector_corners`, `nic_sector_corners`). Two identified
  landmark clusters plus the board plane give enough for an in-plane pose.
- The hypothesis only has to be good enough to say **which direction the
  insignia is**, not to survey. Its uncertainty should be explicit and should
  shrink as evidence accumulates.

**Step 2 — solve for a pose that would expose the insignia.**
This is the "another inverse kinematics thing like the current 3 targets" the
task calls for. `INSIGNIA_RECT_CORNERS` is a known board-frame box, so
acquisition is the *same* problem Stage 2 already solves for SFP/NIC/SC:
generate candidate TCP poses that frame a board-frame target, then gate them
with the existing, trusted machinery — `UR5eArm.solve_ranked` with the live
joint seed, `self_clearance`, `_arm_clear_of_own_cameras`, workspace and
component-clearance guards, and relative joint-travel caps. Command the single
best candidate in **one** move rather than a chain of greedy nudges.

**Step 3 — a deterministic fallback sweep when there is no hypothesis.**
If nothing identifiable is visible, fall back to a fixed, precomputed ladder of
*joint* configurations that tile the plausible board region, visited in a fixed
order, each one IK-valid and arm-clear by construction. Deterministic, bounded,
no gradient required, and trivially unit-testable. This is what replaces
`ACQUIRE`.

**Step 4 — a safe home, always.**
On exhaustion, force abort, or any terminal failure, return to a known-good
observation pose before releasing the controller. The 2026-07-27 session shows
why: without it one bad exit poisons every later invocation.

### 20.3 Constraints the rebuild must respect

- Keep Stage 2 untouched. It is validated; §8 and §10 are hardware-confirmed.
- Keep the input allowlist closed (§2) — no ground-truth board transform, no
  component pose, no scoring state.
- Keep returning expected failures as `success=true, done=false` so Flowstate
  can always release the AIC controller (§3).
- Preserve the force guard semantics, but treat a force abort as a **state to
  recover from**, not merely a reason to return.
- Everything must be offline-sweepable the way `sc_sweep_runner.py` and
  `sfp_sweep_runner.py` are: board yaw x tilt x placement x live start, scored
  on whether the insignia ends up unclipped in a calibrated camera.

### 20.4 Suggested order of work

1. Build the offline acquisition sweep harness **first**, scoring the current
   Stage 1 so there is a baseline number to beat.
2. Port `purple_insignia.py` from `4a20097` and unit-test it against the
   captured hardware frames.
3. Implement the board-pose hypothesis with explicit degeneracy rejection.
4. Implement the pose search reusing the Stage-2 gates.
5. Implement the deterministic joint ladder fallback and the safe home.
6. Delete `viewpoint_search.py` and the wrapper's polarity/rollback logic only
   once the replacement beats the baseline on the sweep.
