# Check Board Visibility v4 — complete next-session handoff

Updated: 2026-07-26

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
HEAD:   d499d5f953ee00bcf14d777a1cfb3fcca9c8ecf7
```

The survey work is an uncommitted working-tree change on top of that commit.
Do not assume `HEAD` alone contains the behavior described here, and do not
discard the working tree.

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
| `flowstate/aic_perception/aic_perception/viewpoint_search.py` | Deterministic Stage-1 fallback phase machine |
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

## 5. Stage 1 — expose the insignia

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
exclusion. The target policies add a further 25 or 40 px minimum clearance.

### 6.4 Candidate grid and deterministic selection

The general search can vary:

- reference-camera standoff:
  `0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.64,
  0.66, 0.68, 0.70, 0.73, 0.76, 0.80, 0.85, 0.90, 1.00, 1.15,
  1.25 m`;
- board-X and board-Y aim offsets, normally `-60, -30, 0, +30, +60 mm`;
- optical roll;
- either isotropic near-normal obliquity or a target-specific directional tilt.

SC overrides the standoff to exactly 0.62 m. NIC and SC use 24 optical-roll
samples at 15-degree intervals. SFP uses the default seven roll samples.

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

`STAGED_SFP_MODULE` frames the loose SFP modules on the board's +Y staging
rail. The conservative board-frame sector is:

```text
X: 0.020 .. 0.090 m
Y: 0.000 .. 0.225 m
Z: 0.010 .. 0.060 m
```

This is the staging/pick sector, not the SFP cages on the NIC cards.

### 8.2 View policy

SFP uses the general near-overhead search:

- all three cameras must frame the complete sector;
- closest feasible standoff wins for maximum component pixels;
- no directional tilt band;
- total reference-camera obliquity at most the general 20-degree cap;
- additional all-camera clearance: 40 px;
- Cartesian reorientation cap: the general 45 degrees;
- default seven optical-roll samples;
- live IK ranking when available, sphere fallback otherwise.

SFP has no recessed-bore depth-quality callback. It relies on sector framing,
gripper clearance, near-overhead conditioning, and reachability.

### 8.3 Current status

SFP has worked in the three tested board orientations on the preceding
hardware build. Preserve this policy while changing NIC or SC. The newest
working-tree IK/arm-in-view changes are unit-tested but have not been rerun on
hardware, so a shared-search change must be tested against SFP, not only
against the target being tuned.

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

- center-camera cross tilt: 10 to 13 degrees along ±board X;
- along-board-Y tilt: at most 2 degrees;
- both board-X sides are searched;
- standoff: exactly 0.62 m;
- all three cameras frame the complete SC sector;
- additional clearance: 25 px;
- 24 optical-roll samples at 15-degree intervals;
- Cartesian orientation search envelope: 180 degrees;
- live IK required;
- worst relative joint delta: at most 185 degrees.

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
   3.8 mm × 11.2 mm half-aperture over 15.64 mm depth;
2. retain at least 3.0 px of projected mouth-to-back-center displacement.

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

Latest local result:

```text
144 / 144 pass
selected standoff: 0.62 m in every case
two-camera depth cue: 3.3426 .. 4.4513 px
bore margin: 0.01346 .. 0.26738
minimum image/gripper clearance: 37.8355 .. 73.9906 px
selected relative joint-motion range: 27.4051 .. 182.4348 degrees
```

The 185-degree cap is measured. Tightening it to 182 degrees loses cases.

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

The current arm-in-view rejection has not been run on hardware. At board yaw
70 degrees the previously deployed endpoint put the upper arm in the center
camera. Validate all three images, not only the final TCP geometry.

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
282 passed in 79.69s
```

Run the SC production sweep separately:

```powershell
python test/sc_sweep_runner.py --workers 4
```

Run the code-node harness from the repository root:

```powershell
cd C:\Users\anshu\College\aic\aic
python docs/reference/filter_estimates_sc_node_test.py
```

Current results:

```text
SC sweep: 144 / 144 pass
filter_estimates_sc reference harness: PASS
```

Before believing a shared-search change:

1. run the full unit suite;
2. run all 144 SC scenarios;
3. retain all three SFP/NIC orientations already known to work;
4. test the exact chained live joint seed;
5. capture all three actual camera frames;
6. compare the published-pose and predicted-joint log lines;
7. inspect the downstream Move Robot trajectory rather than only its endpoint.

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
AIC_SOLUTION=dc50ce22-2362-4345-85b3-89945912e761_BRANCH \
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
7. prints SHA-256 hashes and bundle contents.

The label checks were added after a repeatable lifecycle failure: a freshly
uploaded pod worked, but stopping and restarting the solution left the v4
service unavailable on port 8003 until the asset was deleted and reinstalled.
The leading repository-side cause was concrete. The old image advertised
`ai.intrinsic.asset-id=ai.tar2.check_board_visibility_skill` (missing `_v4`)
and omitted `ai.intrinsic.skill-image-name` entirely, while the installed
manifest and Kubernetes service use `check_board_visibility_skill_v4`.
Intrinsic's supported skill image template requires both labels. Keep the
manifest, asset-id label, and image-name label aligned. A real solution
stop/start after installing the corrected bundle is still the authoritative
end-to-end validation.

The output bundle is:

```text
images/check_board_visibility_skill/
check_board_visibility_skill.bundle.tar
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

## 18. Highest-priority next-session work

1. Deploy the current arm-in-view rejection and validate SC at board yaw
   70 degrees. Confirm the upper arm and forearm are absent from all three
   images.
2. Add the latest five-candidate hardware regression to the SC filter harness,
   then fix row/axis recovery without restoring metric alignment or spacing
   rejection.
3. Add the missing Flowstate
   `result.success && result.done && result.target_valid` gate after the
   default-controller switch.
4. Verify that the obsolete absolute J1-J6 position bounds are removed from
   the SC Move Robot segment, then inspect actual planned relative travel.
5. Measure Stage-2 wall time on the deployed optimized build.
6. Capture a known-working SFP Stage-2 published-pose line and compare its true
   standoff and optical obliquity with the SC policy instead of inferring model
   preferences from screenshots.
7. Check the stale proto/manifest descriptions and target-neutralize the
   `"SFP Stage 2"` diagnostics after behavior is stable.

The core rule for future changes is: measure the workcell geometry offline,
sweep the full board/placement/live-start matrix, and then validate all three
real camera frames. A reachable Cartesian endpoint and an in-frame projection
are necessary, but neither proves that IVM has an unoccluded, well-conditioned
view or that Move Robot will take the predicted joint path.
