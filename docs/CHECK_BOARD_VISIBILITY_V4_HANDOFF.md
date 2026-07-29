# Check Board Visibility v4 — complete next-session handoff

Updated: 2026-07-28 (evening)

> **Start at §24,** then §23, then §22. §24 is the open work: branch divergence is
> now measured on hardware and the 140 mm keep-out is why the arm-in-view gate
> misses it — §24.4 is the plan and §24.5 is what is still unexplained. §23 is the
> staged-SFP tool-occlusion fix and supersedes the coverage box in §8.2/§8.3 **and**
> §22.4. §22 describes the rest of
> the system and supersedes any earlier section it contradicts — the roll counts,
> the reorientation caps (§8.3/§9.2), and the "do not gate/do not widen" entries
> in §17 that were measured under the old 90° cap, 7-roll family and legacy
> harness board position. §21 records the debugging that got there. Everything in
> §22 and §23 is **uncommitted and not deployed.**

> **Read this first.** **Stage 1 no longer exists.** The skill is pure
> perception: it observes once and either runs Stage 2 or fails. Three
> successive acquisition designs failed on hardware and the search was deleted
> rather than tuned again -- §5 and §19 record what and why, so it is not
> reinvented. Stage 2 (SC + staged-SFP survey geometry) is validated and
> hardware-confirmed; §8 and §10 are the policy of record.

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
HEAD:   e929eca + uncommitted working tree
```

The SFP coverage fix and the SC depth fix (§8, §10) are committed and
hardware-confirmed. **Everything from the 2026-07-27 late session is
uncommitted working tree**: the Stage-1 deletion, the untared-force guard
(§19.3), the SC total joint-travel cap and the all-branch arm-in-view hard stop
(§7.1), and the IK rejection diagnostics. Do not discard the working tree.

`origin/navigate-to-purple` (tip `4a20097`) carries the Stage-1 image-plane
search experiment. It was ported, run on hardware, and removed -- see §19.

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

## 5. Stage 1 — removed

The skill commands **no motion at all**. `_execute_inner` grabs one fresh
triplet, and:

- if any calibrated camera holds a complete unobstructed insignia, it hands the
  triplet to Stage 2;
- otherwise it fails, naming which cameras saw a *partial* insignia so the
  operator can tell "nothing there" from "nearly framed".

**One camera is enough** (`REQUIRED_INSIGNIA_CAMERAS = 1`). Two was tried on
2026-07-28 and reverted the same day -- see §22.5.
One was enough until a single-view PnP of one small quad was shown to be too
weak a *range* measurement -- see §6.2 and §21. One `full=True` alongside two
`full=False` is now a refusal, and the message names how many complete views
were found so the operator can tell "one short" from "nothing there".

The failure **raises** `skill_interface.SkillError(9, ...)` after the controller
handoff is published. Two things to know about that:

- `SkillError` takes `(status_code, message)`. Passing one argument raises
  `TypeError` inside the skill service and the real diagnosis is lost. A test
  pins the two-argument call shape, because the intrinsic SDK is not importable
  locally and nothing else can catch it.
- **Raising still aborts the behaviour tree before `Switch To Default
  Controller`**, so the AIC bridge keeps its ICON session on `arm` and later
  Move Robot calls fail with `upstream connect error ... connection
  termination`. `prepare_controller_handoff()` publishes a measured-state hold;
  it does **not** release the bridge lease. If that wedging is unacceptable,
  either revert to `success=False` without raising, or restructure the BT so
  the controller switch is in an always-run branch.

Deleted along with the search: `viewpoint_search.py` (the
ACQUIRE/CENTER/ALIGN/LEVEL/ASCEND phase machine), `stage1_acquisition.py` (the
deterministic joint plan), `board_seek.py` (the ported image-plane servo) and
`_execute_inner_legacy`. A test asserts they are gone rather than dormant.

`purple_insignia.py` is kept: `analyze_purple` supplies the per-camera
`purple_seen/full/edges/area` diagnostics in the Stage-1 log line, which is
what makes a failure readable.

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

**One accepted estimate is enough; two or more must agree** (§22.5).
Estimates from multiple cameras must form a cluster within 5 cm and 8 degrees;
Stage 2 now refuses unless at least `REQUIRED_INSIGNIA_CAMERAS` of them land in
that cluster, rather than applying the check only when a second view happened to
exist. Within the cluster it prefers the center-camera estimate, otherwise the
lowest reprojection/centroid error.

The cluster's board **origin is then averaged** across the agreeing views, while
rotation stays with the preferred view. Range is the weak axis of a single-view
PnP, the cameras are ~115 mm apart so their range errors are largely
independent, and the survey search runs against a 25 px clearance floor that a
few millimetres of range error can cross. Rotation is deliberately *not*
averaged: an orientation mean over a near-square landmark can interpolate
between two different mirror hypotheses, and the 8 degree cluster test already
bounds the disagreement.

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

### 7.1 Joint-travel and arm-in-view gates (2026-07-27 late)

Two hardware failures produced two additional gates.

**Total joint travel, SC only.** The 185/225 deg cap is on the *worst single
joint*. Hardware published this and it passed:

```text
delta_deg=[-163.7, -16.0, 147.5, 33.7, 166.0, -34.9]
max=166.0deg  total=561.8deg
```

Three joints swinging 150-166 deg at once -- a whole-arm reconfiguration, not a
survey move, and what the field reported as "the arm contorts and does almost a
360". `total_motion` was already computed for ranking and simply never gated.
`max_total_joint_motion_rad` now caps it at **400 deg for SC**
(`TOTAL_JOINT_MOTION_LIMIT_RAD`); SFP and NIC stay `math.inf`.

The offline distribution is bimodal -- poses are either under ~300 deg or over
450 -- so 300/350/400/450 select the identical set; 400 sits in the gap with
margin over the 208 deg field maximum. Gating SFP as well costs 26 of 144
placements, and SFP has not reported the problem, hence SC-only. Note the
561.8 deg move above was logged on an *SFP* run (`view_quality=+inf`,
`cross_tilt=0.0`, 225 deg gate), so an SC-only cap would not have refused that
particular pose.

**Every IK branch must be arm-clear.** `select_clear_ik_solution` used to
accept a pose when *any* branch cleared the wrist cameras. But the skill
publishes only a Cartesian pose: Move Robot re-solves it and may take a
different co-terminal branch (§11), so validating one branch guaranteed
nothing. Measured exposure at three board yaws:

```text
yaw   probed  some-clear  ALL-clear  partial
  0     20        16          0        16
140     31        15          0        15
315     16        16          1        15
```

Essentially every accepted pose had a branch that puts the arm in frame. The
gate now requires all branches clear, so whichever one Move Robot picks is
safe. Cost: SC unchanged at 144/144, SFP 92 -> 91 of 144. Applied to all three
sectors.

This cannot cover a branch outside `solve_ranked`'s set. Closing that needs the
skill to publish joint targets rather than a Cartesian pose, which the current
interface does not allow.

**Diagnostics.** A Stage-2 refusal now logs a per-gate breakdown, an explicit
`BINDING GATE = ...` verdict (reachability / arm-in-view / travel cap), the
live seed, reach and TCP-height spans, and the eight nearest misses. The old
message conflated three failures needing opposite fixes.

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
- Cartesian reorientation cap: **180 degrees** (was 90, was 45 -- see §21);
- **12** optical-roll samples at 30-degree intervals (24 was tried and cost 160 s of search; see 22.6);
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

**Confirmed working on hardware (2026-07-27).** Sweep result, at the corrected
board distance and the 180 deg / 24-roll policy (2026-07-28):

```text
144 / 144 poses found, 144 / 144 frame all six seats, 0 clipped
selected standoff:  0.64 .. 0.76 m
clearance:          25.3 .. 62.8 px
worst seat margin:  103.1 px
worst joint travel: 178.8 deg (225 deg cap)
```

The previous row read `92 / 144 found, worst seat margin 118.5 px, standoff
0.64 .. 0.85 m`, and this handoff attributed the 52 no-pose cases to the
`home + J6 +90 deg` Stage-1 exit -- "a Stage-1 problem". **That was wrong.**
They were the 90 degree Cartesian reorientation cap silently choosing the
candidate set (§21). The 144/144 above is measured with the board 13 cm
*farther* from the base than the run that scored 92, i.e. on strictly harder
geometry.

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
- Cartesian reorientation cap: 180 degrees (was 90 -- see §21).

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
| `BINDING GATE = reachability` on a pose you know the arm can reach | Believe it only since 2026-07-28. Before that `solve_ranked` filtered the 140 mm wrist-camera keep-out *before* returning and the empty list was reported as "no analytic IK solution at all"; at the real board distance 231 of 926 such verdicts were keep-out rejections. The log now says `camera_keepout=` and can name `BINDING GATE = wrist-camera keep-out` |
| Two invocations at the *same* arm pose disagree | Single-view board range jitter crossing the 25 px clearance floor. Damped by averaging the origin whenever two or more views agree (§22.5); read `origin_spread` on the `board pose fused over N agreeing cameras` line. Requiring two views was tried and reverted |

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
unit suite:  263 passed
SC sweep:    144 / 144 found, depth cue 6.02 .. 9.97 px, bore margin >= 0.0188
SFP sweep:   144 / 144 found, 144 / 144 frame all six seats, 0 clipped
filter_estimates_sc reference harness: PASS
```

Both sweeps now place the board at its **measured hardware position**,
`(-0.5189, 0.2054)` in `base_link` -- 0.558 m horizontally from the base. They
had been pinned at `(-0.3445, 0.2602)`, 0.4317 m, which is 13 cm nearer than the
real cell. That matters more than it sounds: the survey viewpoint sits 0.64 m
*above* the board, so board distance is spent directly out of the arm's
envelope, and a start pose that swept clean could still fail in the field.
`--board-center-mm -344.5 260.2` reproduces the old pin for comparing against
any recorded number.

Watch the SC worst-case bore margin: the extra distance moved it from 0.0313 to
**0.0188**. Still positive, still 144/144, but it is the number the extra reach
is being paid for.

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

`rebuild_and_install_stage1_v4.sh` (in the WSL workspace, **not** version
controlled) pins `board_stage2.py` by hash so Stage 2 cannot drift by accident.
It is a pin, not a freeze: an intentional change is allowed once its sweeps have
been re-run.

```bash
# normal build+install
AIC_SOLUTION=<uuid>_BRANCH bash rebuild_and_install_stage1_v4.sh

# after an intentional Stage-2 change, with both sweeps re-run
ALLOW_STAGE2_CHANGE=1 AIC_SOLUTION=<uuid>_BRANCH   bash rebuild_and_install_stage1_v4.sh
# then paste the printed hash into EXPECTED_STAGE2_SHA
```

`AIC_SOLUTION` has no default -- it persists in the shell and more than one
solution has been in play, so a stale default silently installs to the wrong
cluster.

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
10. ~~Do not enlarge the staged-SFP coverage box along board Y.~~ **Superseded
    by §22.4.** Fixtures mount on any rail with +/-0.09425 m of travel, so
    the +/-0.1125 box covered half the legal range; coverage is now a
    widest-first ladder, and the old frontier table used the old policy.
11. Do not rebuild a Stage-1 acquisition search without reading §20. Three
    designs have failed on hardware for three different reasons.
12. ~~Do not gate summed joint travel on SFP/NIC.~~ **Re-open this.** The
    26-of-144 cost was measured under the old policy, and a 2026-07-28 field
    run published `total=616.5 deg` on SFP (§22.9 item 1).
13. Do not accept a survey pose because *one* IK branch is arm-clear. Move
    Robot picks the branch, not this skill (§7.1).
14. Do not trust a board `minAreaRect` orientation when the mask is clipped on
    two or more image edges — `long_ratio ~= 1.00` with
    `long_axis_error = +0.0deg` is the degenerate signature, not a measurement.
15. Do not reintroduce a Cartesian reorientation cap below 180 degrees to bound
    arm motion. It is measured from the live TCP, so it *selects the candidate
    set* before anything is scored; the live-seeded joint-travel gate is what
    bounds real motion (§21).
16. Do not size an offline sweep's board position by anything but the measured
    cell. A sweep 13 cm closer than hardware certified a policy that scored
    92/144 on easy geometry while failing in the field (§21).

## 18. Highest-priority next-session work

1. **Decide the raise-vs-return question** (§5). Raising on a missing insignia
   aborts the BT before `Switch To Default Controller` and wedges the AIC
   bridge, so later Move Robot calls fail with `upstream connect error`. Either
   revert to `success=False`, or put the controller switch in an always-run BT
   branch.
2. **Add the Flowstate `result.success && result.done && result.target_valid`
   gate** after the default-controller switch. Still missing, and it has now
   crashed Move Robot with `norm(quat)==0` on SFP as well as SC.
3. **SC J6 preference is not engaging.** **See §24.4 item 4 — measured at
   `j6_error=266.7deg` on 2026-07-28 20:14, the worst on record.** §10.4 prefers a +/-180 deg J6 flip
   specifically to keep the camera cluster off the occluding side, and field
   logs show `j6_error` of 129-245 deg -- it lands on the wrong side every
   time. It only wins inside a 30 deg travel plateau. This is the remaining
   lead for the arm occluding its own view, and it is unmeasured.
4. ~~**Start-pose sensitivity.**~~ **Done 2026-07-28 -- see §21.** This was the
   cause of the field "no IK pose from certain start poses". The cap is now
   180 deg in every sector and availability no longer depends on the start
   wrist roll.
5. **`UR5eArm.autocalibrate()` instability**: `tool=197.1mm` in most runs,
   `200.4mm` and `201.1mm` in others. A 4.2 mm shift moves every projected
   candidate.
6. Add the five-candidate hardware regression to the SC filter harness, then
   fix row/axis recovery without restoring metric spacing rejection.
7. Check the stale proto/manifest descriptions and target-neutralize the
   `"SFP Stage 2"` diagnostics.

The core rule for future changes is: measure the workcell geometry offline,
sweep the full board/placement/live-start matrix, and then validate all three
real camera frames. A reachable Cartesian endpoint and an in-frame projection
are necessary, but neither proves that IVM has an unoccluded, well-conditioned
view or that Move Robot will take the predicted joint path.

## 19. The Stage-1 hardware record

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

### 19.3 The wrist force guard was the shared root cause (fixed 2026-07-27)

The rebuilt deterministic Stage 1 failed on its very first hardware run without
commanding a single joint:

```text
16:06:47  board visibility: success=False done=False force=25.72N
          msg=wrist force guard active at 25.72N;
              deterministic acquisition refused
```

**The wrist FTS reading is raw and untared.** It carries a large constant
sensor bias plus the tool weight, and only the tool weight rotates in the
sensor frame. Writing `F = b + R*g`, the free-space magnitude `|F|` sweeps the
whole interval `[ | |b|-|g| |, |b|+|g| ]` as the wrist reorients — it is **not**
rotation-invariant, which is exactly what the previous "compare magnitudes
instead of vectors" guard assumed.

Fitting the two extremes actually observed in free space:

```text
min free-space reading   13.59 N
max free-space reading   25.72 N
  =>  |b| = 19.66 N  (constant bias)
      |g| =  6.07 N  (tool weight, ~0.62 kg)
      free-space magnitude swing = 12.13 N
```

Both shipped thresholds sat **inside** that swing, so both fired in free space:

- the **18 N absolute ceiling** lies inside `[13.6, 25.7]`. It refused the new
  Stage 1 outright at 25.72 N. It is also why the *old* Stage 1 aborted at
  17.79 N and 17.06 N;
- the **5 N magnitude-delta** is less than the 12.13 N swing, so any large
  reorientation tripped it. That is the `02:01`-`02:10` sequence where four
  consecutive invocations force-aborted while the arm was in free space.

This retires the "the tool is in contact with the board" reading of §19.1.
Those were free-space readings at different wrist orientations.

**The fix** (`robot_motion.force_guard_tripped` / `contact_force_n`, now the
single source of truth for both the skill and `RobotMotion`): contact is force
the static load **cannot explain**, i.e. magnitude above the measured envelope
`STATIC_FORCE_ENVELOPE_HI_N = 27 N`, plus a `FORCE_RUNAWAY_CEILING_N = 45 N`
backstop and a magnitude-change test that only trips beyond the entire
free-space swing. A caller-supplied ceiling below the envelope is deliberately
ignored — on an untared sensor it only reports which way the wrist points.

Only force *above* the envelope counts. A reading below it is left alone on
purpose: a lightly-loaded or properly-tared sensor sits there, and treating
that as contact fires the guard on a healthy zero reading.

Verified against every magnitude the hardware actually produced:

```text
raw 13.59 / 14.31 / 14.73 / 17.06 / 17.79 / 25.72 N  ->  unexplained 0.00 N, no trip
raw 35 N  ->  unexplained  8.00 N, trips
raw 50 N  ->  unexplained 23.00 N, trips
```

**Cost, stated plainly:** contact below ~5 N on top of the worst-case static
load is no longer detectable. That is the unavoidable price of an untared
sensor. Recovering it needs real gravity compensation — fit `b` and `g` from
orientation-tagged samples (the skill already resolves `base_T_tcp` at every
snapshot, so the data is there) and gate on the residual against the predicted
static wrench. That is the right next step if Stage 1 ever needs finer contact
sensing; it was out of scope for unblocking the run.

Always log the unexplained component next to the raw magnitude. A bare
"25.72 N" reads as alarming and is in fact free space.

## 20. What was tried for Stage 1, and why it is gone

Do not rebuild an acquisition search without reading this. Three designs
reached hardware; all three failed for different reasons.

**1. Phase machine** (`viewpoint_search.py`, ACQUIRE -> CENTER -> ALIGN ->
LEVEL -> ASCEND, 1614 lines). Steered on the board mask's `minAreaRect`
orientation, which is degenerate exactly when needed: a mask clipped on two or
more image edges yields a frame-aligned rectangle, logged as
`long_ratio=1.00 long_axis_error=+0.0deg`. It also split J1/J6 authority across
phases, which couples badly at the levelled pose where base-Z and wrist_3 are
parallel. Wedged in the field with `logo=False` on all three cameras and
identical readings across four consecutive attempts.

**2. Deterministic joint plan** (`stage1_acquisition.py`). Precomputed a safe
path to a fixed observation pose. Never executed:

- the min-jerk profile outran its deadline (a 33 deg segment needs 5.4 s at the
  0.20 rad/s cap; the 6 s budget also had to absorb up to 2 s of target-mode
  retry). Sizing the deadline from the segment fixed that;
- then the deployed controller dropped joint target mode 0.43 s in --
  `controller left joint target mode; joint target reversed`. The controller
  rejects `/aic_controller/change_target_mode` around in-flight executions, so
  the joint-target path is not dependable.

**3. Image-plane servo** (`board_seek.py`, ported from
`origin/navigate-to-purple` tip `4a20097`). Small Cartesian translations at
fixed orientation, steering on clipped edges plus centre error, purple taking
over from the board mask once visible. Avoids both problems above -- no joint
mode, no orientation cue. Its limit is structural: when the board overflows the
frame on opposite edges there is no gradient. `image_plane_direction` falls
back to the centre error, which on the real stuck frame
(`edges=left,right,top`, centre `(0.034, -0.596)`) was under the trigger on X,
so the whole signal was a single vertical nudge. It also never backs off, and
backing off is what a too-close view actually needs.

**Resolution.** Hardcoded Flowstate Move Robot poses that see the whole board.
That works, and it removed the problem rather than solving it. The skill now
requires the caller to supply such a pose.

If acquisition is ever rebuilt, the pieces worth reusing are
`purple_insignia.analyze_purple` (HSV band copied from the proven
`PerceptionInsert._sc_purple_logo_centroid_px`) and the flat no-phase loop
shape. The missing piece in every attempt was a **board-pose hypothesis**: none
of them ever formed one, so none could aim. `estimate_board_pose()`
(`board_stage2.py`) PnPs the plate outline but must not be trusted on a clipped
mask; the coloured landmarks that stay in frame -- blue SC adapters, green NIC
cards, at known board-frame positions -- are the untried route.

Whatever is built, score it offline the way `sc_sweep_runner.py` and
`sfp_sweep_runner.py` do, and beat the current behaviour on that sweep before
deleting anything.

## 21. The reorientation cap, and why "definitively reachable" poses failed

Field report (2026-07-28): *"for certain starting poses the perception just
doesn't output an IK pose, and they are definitively reachable."* Correct on
both counts. Three separate faults stacked up.

### 21.1 Reconstructing the run offline

The logs carry enough to rebuild the cell exactly. FK on the logged
`target_deg` under the logged calibration (`base=Rz180 tool=197.1mm`)
reproduces the published `target=(-0.5114, 0.1864, 0.4506)` to **0.4 mm** and
the logged `move=0.280m` to 0.1 mm. Do this first on any future report -- it
converts a log into a measurable scene.

That reconstruction put the board's aim point at `(-0.5189, 0.2054, 0.0355)`,
**0.558 m** horizontally from `base_link`, against the **0.4317 m** both sweep
harnesses were pinned at.

### 21.2 The cap was choosing the candidate set

`max_angular_motion_rad` is measured against the *current* TCP. It reads like a
motion bound; it is actually a filter applied before any candidate is scored.
Measured at the real board distance over 8 board yaws:

```text
live start pose      cap=45   cap=90   cap=180   cap=180 + 24 rolls
field 01:29            1/8      5/8       7/8           8/8
sweep home             3/8      6/8       7/8           8/8
home + J6 +90 deg      0/8      0/8       7/8           8/8
chained start          5/8      7/8       7/8           8/8
```

From the rolled-wrist start the 90 degree cap admitted **1036 framed candidates
of which zero had any IK solution** -- the exact `BINDING GATE = reachability`
refusal seen in the field. The poses the arm *could* reach were never in scope.

SC already ran at `math.pi` for this reason. SFP and NIC now do too, and SFP
moves 7 -> 24 rolls (the last case at each start pose needs a camera-cluster
orientation the coarse family skips). Cost: ~3.7x search time, ~2.1 s -> ~8 s
per case offline. Hardware Stage-2 timing after this change is **unmeasured**.

### 21.3 "Unreachable" was often not unreachable

`UR5eArm.solve_ranked` filters the 140 mm wrist-camera/forearm keep-out before
returning, so an empty list meant either "outside the workspace" or "reachable,
but every branch folds a camera into the forearm" -- and the skill logged both
as *"no analytic IK solution at all"*. At the real board distance **231 of 926**
such verdicts were keep-out rejections. The true kinematic wall for a 0.64 m
standoff is ~0.74 m of board distance; the keep-out wall is ~0.62 m.

The failure path now re-solves without the keep-out and reports
`camera_keepout=` plus a `BINDING GATE = wrist-camera keep-out` verdict. The hot
path is unchanged.

**Still open:** 140 mm was calibrated from a single planner rejection whose best
branch sat at 111 mm, and it is measured to the forearm *centreline* of a 50 mm
capsule -- roughly 90 mm of real surface clearance. Offline, dropping it to
110 mm takes three of the four start poses from 7/8 to 8/8. Nobody has checked
it against the workcell planner's actual mesh test. That is the next cheap win.

### 21.4 The board pose was a single weak measurement

Two invocations 7 s apart, at the same arm pose, on visually identical frames,
disagreed: one published at 0.640 m standoff, the next framed a single candidate
at 0.837 m reach and refused. The insignia held 0.45% of the centre image, only
the centre camera had it complete, and selected poses sit at 25.3-26.7 px
against a 25 px clearance floor -- so millimetre-level range error flips the
whole near-standoff family across that floor. Hence `REQUIRED_INSIGNIA_CAMERAS`
and the averaged cluster origin (§6.2).

**The cost is a stricter start pose**, and it is not hypothetical: the 01:29 run
that *did* publish had only one complete view and would now be refused at the
gate. If that proves too tight in the cell, lower the one constant rather than
spreading the policy back across three call sites.

### 21.5 Result

```text
SFP sweep:  92 / 144  ->  144 / 144   (and at a board 13 cm farther out)
SC sweep:  144 / 144  ->  144 / 144
unit suite:      259  ->  263
```

The SFP jump is the headline, but note what it means about the *previous*
number: 92/144 was measured on geometry the robot never sees, and this handoff
then blamed the 52 misses on Stage 1. A sweep that does not sit where the
hardware sits will keep producing confident, wrong conclusions.

## 22. Current system (2026-07-28, end of session)

Everything below is **uncommitted working tree** and **not deployed**. The last
build hardware ran predates the coverage ladder, the roll reduction and the
insignia-tier fixes. `board_stage2.py` is unchanged since the `reach_cap` fix,
so `EXPECTED_STAGE2_SHA` is
`765038043ab15e01c7c2eecd9e75995dc2b558d3c96e1e0937e62555ea8c604e`.

### 22.1 The two invariants

Stage 2 is organised around two things no fallback may touch:

1. **The view.** Per sector: SFP near-overhead strip view, NIC <=2 deg
   straight-down look into the cages, SC 16-20 deg long-face band and
   two-camera bore gate. Coverage boxes, all-three-camera framing, the
   obliquity/tilt bands and `min_view_quality` are fixed.
2. **Collisions.** The `UR5eArm.self_clearance` 140 mm wrist-camera keep-out,
   applied inside `solve_ranked`. **Never relaxed by anything.** The field
   instruction was explicit: collisions are a hard no.

The gripper is *allowed* in frame -- that is what the calibrated silhouette is
for. What must stay clear of the sector is the gripper keep-out and the arm
limbs.

### 22.2 The relaxation ladder

`_run_sfp_geometric_stage2` runs `search_survey_pose` once per tier and returns
on the first that yields a pose. Everything varied is comfort, not correctness.

```text
0  strict + insignia kept in view   sector caps + insignia readable in >=1 camera
1  strict                           exactly the previously deployed behaviour
2  joint-travel caps lifted         225/185 -> 360 deg, total cap off
3  any arm-clear IK branch          instead of every branch clear
4  reduced clearance margin         25 px -> 12 px
5  angled view (8 deg off normal)   isotropic-obliquity sectors only, NOT SC
6  angled view (15 deg off normal)  as above
```

Tier 0 sits **in front of** the old strict tier rather than modifying it, so a
board where the insignia cannot be kept in view falls through to exactly what
shipped. A test pins the four original tier definitions verbatim plus the
break-on-first-success.

Only tiers 5-6 degrade the picture; they log a distinct warning. **Tier 0
success is not a relaxation** -- an earlier build warned on every first-tier
success because the tier had been renamed, and that noise made healthy runs look
like fallbacks.

### 22.3 Per-sector settings

| | SFP (0/1) | NIC (2) | SC (3) |
| --- | --- | --- | --- |
| reorientation cap | `math.pi` | `math.pi` | `math.pi` |
| optical rolls | **12** (30 deg) | 24 (15 deg) | 24 (15 deg) |
| standoffs | default 0.30-1.25 | **floored at 0.66** | 0.55-0.62 |
| clearance floor | 25 px | 25 px | 25 px |
| coverage | **3-rung ladder** (22.4) | `nic_sector_corners` | `sc_sector_corners` |

The 90 deg reorientation cap is gone everywhere (see 21.2). The NIC 0.66 m floor
exists because below it the outer ports leave the 7.46 deg bore cone while still
framing -- 21 of 126 poses published a ~6-of-10 view before it.

### 22.4 SFP coverage is a ladder

Zones 3/4 are a **high-mix supply area**: LC/SC/SFP fixtures mount on *any* rail
in *any* order, translation limits +/-0.09425 m, orientation +/-60 deg. There is
no seat list to aim at. Legal fixture origins span board Y +/-0.2005, and
`sfp_module_strip_corners` (+/-0.1125) covers about half of it -- a fixture
parked at the end of its rail falls outside the only box the search checked.

`_coverage_targets_for_target` now returns widest-first:

```text
module_coverage_corners   X -0.0275..0.1535   Y +/-0.2575   all rail families
sfp_envelope_corners      X  0.0000..0.1100   Y +/-0.2575   SFP rail X
sfp_module_strip_corners  X  0.0300..0.1150   Y +/-0.1125   what ships today
```

A wider box cannot be framed close, so it forces a farther standoff -- that *is*
"move the arm up" expressed as geometry. Measured at the real board distance:
8/8 board yaws at +/-0.1125 and 0.64-0.70 m, 7/8 at +/-0.145 and 0.80-0.85 m,
3/8 at +/-0.160, 0/8 at +/-0.178. The last rung is the deployed box, so an
unreachable placement keeps current behaviour.

**The 8.2 frontier table is obsolete** -- measured at cap 90 deg, 7 rolls and the
legacy board position, and it declared `y_half 0.1783` infeasible on evidence
that no longer holds.

### 22.5 Insignia contract

`REQUIRED_INSIGNIA_CAMERAS = 1`. Requiring two complete views was tried on
hardware on 2026-07-28 and **reverted the same day**: it refused five
consecutive invocations with "0 have one" at poses where the board was plainly
in view. With Stage 1 gone, each of those is a dead stop.

Kept is the free half: when two or more cameras accept an estimate they must
agree within 5 cm / 8 deg, and their board **origins are averaged** (rotation
stays with the preferred view -- an orientation mean over a near-square landmark
can interpolate between mirror hypotheses). Field evidence:
`fused over 3 agreeing cameras origin_spread=0.0059m shift_from_source=0.0017m`.

`SLIVER_EDGE_MARGIN_PX = 12.0`: when *no* camera holds a fully-framed insignia,
one clipped by up to 12 px is accepted, with a warning. The landmark is the
bracket bounding rectangle, so a clipped extreme biases the recovered range; the
agreement check still applies. The Stage 1 gate and Stage 2 share this two-pass
rule (`_cameras_with_usable_landmark`) -- previously they could disagree and the
gate admitted a triplet Stage 2 then rejected.

### 22.6 Performance: where the time goes

A field SFP tier measured **160.31 s**. Causes, in order:

1. `view_quality` is evaluated on **every** candidate surviving the cheap prunes
   (~10k), while only the framed handful reaches the IK gate (`probed=68` in the
   same trace). An insignia check placed there cost ~100 s. It now runs inside
   the IK gate.
2. The grid is `21 standoffs x 25 offsets x rolls`, so each roll is ~500
   candidates of full-resolution gripper-mask work. 24 -> 12 rolls halves it.

The coverage ladder is **not** a cost: `_best_for_target` is lazy and returns on
the first target that yields a pose.

**Rule: never put per-candidate work in `view_quality` unless it is cheap. Put
it in the `joint_motion` gate, which sees ~100x fewer poses.**

### 22.7 Diagnostics

```text
survey policy target=N sector_boxes=N coverage_x=[..] coverage_y=[..] ...
survey inputs board_origin=(..) board_normal=(..) source=.. reprojection=..
  tcp=(..) seed_deg=[..] live_ik=..
survey search tier=... joint_cap=.. total_cap=.. any_branch=.. clearance=..
  -> FOUND/none | probed=.. unreachable=.. camera_keepout=.. insignia_lost=..
  arm_in_view=.. arm_clear=.. best_worst_joint=.. took=..s
```

On total failure every tier line repeats at ERROR. Reading guide: `probed=0`
means nothing reached IK (framing/clearance/obliquity); `unreachable` dominant
means the board is too far for that sector standoff; `camera_keepout` or
`arm_in_view` dominant means the pose was reachable and something else refused.

`unreachable` no longer masks the binding gate in the verdict -- NIC probes
hundreds of unreachable far poses by design, and that used to report
`BINDING GATE = reachability` while every near-miss was `arm_in_view`.

### 22.8 Test and sweep status

```text
unit suite:  266 passed
SFP sweep:   144 / 144        <- measured BEFORE the coverage ladder
SC sweep:    144 / 144        <- measured BEFORE the tiers
NIC sweep:   105 found / 105 passed / 39 honest refusals (new harness)
```

`test/nic_sweep_runner.py` is new: NIC had shipped on three hardware
orientations and no offline matrix while SC and SFP each had 144 cases. Its port
audit checks all ten mouths framed in all three cameras **and** each inside the
7.46 deg cone -- framing is not sufficiency.

**The harnesses do not exercise the relaxation ladder or the coverage ladder.**
They call `search_survey_pose` directly with a single box and their own copy of
the arm-in-view rule, now stricter than the skill. Their numbers are a
conservative lower bound and do not validate tier behaviour. Closing that is the
highest-value harness work.

### 22.9 Open items

1. **SFP total joint travel is still ungated.** A field run published
   `joint_max=175.5 total=616.5 deg` -- three joints swinging ~175 deg at once,
   the exact contortion `TOTAL_JOINT_MOTION_LIMIT_RAD` exists to stop, but the
   cap is SC-only. Section 17 item 12 ("costs 26 of 144") was measured under the
   old policy and needs re-measuring.
2. **The exhaustion message counts all tiers** even for SC, which skips the two
   angled ones. Cosmetic, wrong in an SC log.
3. **Parallel invocation is impossible by construction**:
   `_execute_lock.acquire(blocking=False)` returns a failure immediately, so
   concurrent Flowstate nodes bail rather than queue. Sequential-immediate works
   and is what the field uses. Making it concurrent means scoping the lock to
   the snapshot grab and `prepare_controller_handoff()`, which publishes to a
   live actuator interface.
4. **Staged-object geometry is not grounded.** The task-board URDF models two
   SFP and two SC mount *fixtures* (`sfp_mount_rail_0/1` etc). The five pick
   objects in the field are `sfp_sc_cable` SDF models -- separate spawned
   entities with `lc_plug_link`/`sc_plug_link` ends, attached by `CablePlugin` --
   and their staged poses are **not** in the board URDF. Every coverage constant
   in `board_stage2.py` describes the rails, not the cables. Find where cables
   are spawned before trusting any coverage number.
5. The 140 mm keep-out is calibrated from a single planner rejection at 111 mm
   and measured to a capsule *centreline* (~90 mm real clearance). Relaxing to
   110 mm recovered cases offline. **Superseded by §24.3: the workcell planner has
   now been observed executing a 122 mm branch, so 140 mm is provably tighter than
   the real constraint, and being tight here makes the arm-in-view gate blind.**
   Still prefer checking against the workcell planner mesh test -- publishing a pose the planner refuses is a
   hard move failure, and the field instruction on collisions is absolute.

## 23. Staged SFP: the outer module was behind the tool (2026-07-28, late)

Field report: *"at the edges it doesn't even see the bottom sfp port"*, on a run
that looked healthy — `tier='strict + insignia kept in view' -> FOUND`,
`standoff=0.660m`, `min_clearance=27.0px`, `obliquity=5.8deg`, all six seats
comfortably framed. Uncommitted and **not deployed.**

### 23.1 It was tool occlusion, not clipping

FK on the logged `target_deg` under the logged calibration reproduces the
published `target=(-0.3940,0.3293,0.4518)` to **0.49 mm** and `move=0.266m` to
0.1 mm, which turns the log into a measurable scene (§21.1's method — do this
first, always). Projecting the six legal seats into all three cameras at that
pose:

```text
center_camera  seat +156.2  edge=+159.2px  gripper mask 4/8 corners
center_camera  seat +200.5  edge= +75.2px  gripper mask 8/8 corners
```

Every seat is 75–355 px *inside* every image. The +Y outer module is invisible
because it lands **behind the centre camera's gripper silhouette**. Edge margin
was never the binding constraint, so every number the policy was tuned against —
and the sweep's whole seat audit — was measuring the wrong obstacle.

Same class of bug as the one-rail sector in §8.2, one level up: **the coverage box
is the only region anything checks**, so a seat outside it is on faith. There the
box's placement was wrong; here its size was.

### 23.2 The frontier, re-measured against the right obstacle

81 cases (9 board yaws × 3 placements × 3 live starts) at the hardware board
distance, auditing all six seats for image margin **and** gripper occlusion:

```text
y_half    found   all 6 framed   all 6 unoccluded   standoff
0.11250   81/81       81/81            0/81         0.64-0.66   <- shipped
0.14500   81/81       81/81           81/81         0.80
0.15625   81/81       81/81           81/81         0.85
0.16000   78/81       78/81           78/81         0.85-0.90
0.17000   45/81       45/81           45/81         0.90
0.17825    0/81         -                -          infeasible
```

`SFP_COVERAGE_HALF_Y` is now **0.145** — the narrowest span that clears the tool
everywhere, and therefore the closest standoff that shows all six seats. Wider
only costs standoff and availability. `0.17825` (outer seat centre + body
half-extent, i.e. strict geometric containment) cannot be framed and
gripper-cleared from any reachable pose.

The cost is real and was accepted deliberately: **0.64 m → 0.80 m of standoff**,
every module ~20% smaller. The field instruction was that a higher view is fine
and probably preferred. A smaller module is a worse measurement; a hidden one is
no measurement.

### 23.3 Framing the box is necessary, not sufficient

`_staged_seats_are_visible` gates every candidate on the six seat bodies
themselves — inside the usable image and clear of the gripper mask, in all three
cameras — exactly as the SC bore gate does for its mouths. It runs in the **IK
gate** (~68 poses/search), never in `view_quality` (~10k); that rule is §22.6 and
it still holds.

`board_stage2.sfp_seat_bodies()` is now the single definition of a seat body
(mount origin through protruding tip). The skill and `sfp_sweep_runner.py` had
separate copies, which is how they came to disagree.

With the gate active the ±0.1125 fallback box can no longer publish blind: it
reaches 138/144 by stepping *back* until the seats are visible.

### 23.4 The arm-in-view rule is now absolute for SFP — for free

Field instruction: the arm getting between the cameras and the board **cannot**
happen (the SC yaw-70 failure, §10.6). Staged SFP therefore passes
`sector_regions=None` — no arm limb anywhere in any image, the strong rule §7.1
had relaxed away.

It costs nothing, which is the whole reason to take it. Over 144 cases, keep-out
= coverage box, = rail column, = whole board face, = nowhere-in-any-image all
select **138/144 with identical standoff and joint-travel ranges**. Moving the
view up to 0.80–0.85 m removed the arm-occlusion problem structurally instead of
trading anything for it. NIC and SC keep the region rule — they sit close to their
bore cones and cannot afford it (§7.1's 29-of-435 measurement).

### 23.5 Time: 154.86 s tier down to ~2.6 s of search work

Three separate causes, all of them waste:

1. **The coverage ladder ran a full grid per rung.** `_best_for_target` is lazy
   about *success*, not failure, so §22.4's two wide rungs (±0.2575) — measured
   here as infeasible in 81/81 — cost two complete standoff × offset × roll
   searches on every invocation. §22.6's "the ladder is not a cost" was wrong.
   Now two rungs, and the first one succeeds.
2. **Nine standoff rungs below the frontier.** A ±0.145 box cannot be framed
   nearer than ~0.75 m, and the default ladder opens at 0.30 m. Flooring SFP at
   0.70 m (as NIC is at 0.66) halves the search and selects the **bit-identical**
   TCP, standoff, clearance and roll at every yaw tested.
3. **The insignia tier** — see §23.6.

Measured on a reconstructed scene: 21.0 s → 2.6 s for the same search, and the new
pose has all six seats visible where the old one did not.

### 23.6 The Stage-2 insignia check is gone

Field instruction, and it was right. `_insignia_visible_from`, tier 0
(`"strict + insignia kept in view"`), `require_insignia` and the `insignia_lost`
counter are deleted. Stage 1 already refuses the triplet unless a calibrated
camera holds a complete insignia, so the board is localized before Stage 2 runs;
re-checking it on the survey *endpoint* rejected **zero** candidates on the trace
it was added for (`probed=68 = 4 unreachable + 50 keepout + 14 clear`) while
costing a whole extra grid search — and ~90 s more in the deployed build, where it
still sat in `view_quality`.

If a chained NIC/SC call cannot find the insignia from where the previous survey
left the arm, that belongs in how the process sequences its Move Robot poses, not
in narrowing every survey view to protect a measurement already taken.

### 23.7 Summed joint travel is now gated on SFP too

§17 item 12 said not to; §22.9 item 1 said re-open it. Re-opened.
`TOTAL_JOINT_MOTION_LIMIT_RAD` (400°) now applies to targets 0/1/3, not SC only.
The old justification failed on both halves: a field SFP run published
`total=616.5deg`, and the "costs 26 of 144" was measured under the 90° cap and
one-rail coverage. Re-measured: 123/144 instead of 138 at the strict tier, worst
summed travel 640° → 342°. **The 15 are not lost** — the next tier lifts the cap,
so a placement with no civilised pose still gets the contorted one, logged as a
relaxation instead of published as if it were normal.

**This does not fully solve the contortion.** The 18:07 field run published
`joint_max=108.4 total=363.7deg` and looked contorted in the viewport; 363.7° is
*under* the 400° cap, so this gate would not have refused it. See §23.9 item 1.

### 23.7b Letting joint travel outrank standoff — tried, measured, reverted

The obvious next lever, and it does not work. Standoff dominates the objective
lexicographically, so at the closest feasible rung *any* IK-valid candidate wins
however far the arm folds — a cheaper pose one rung out is in a different plateau
group and never compared. Collapsing the whole band into one plateau so travel
decides is a four-line change (`standoff_ranks_first`).

Measured over 144 cases:

```text
                        jmax med   jmax p90   jmax max   >150deg   standoff med
standoff dominant          69.4      141.9      167.9       8          0.80
joint travel ranks         66.7      136.7      167.0       8          0.85
```

Three degrees of median travel, and every module gets smaller to pay for it. Once
the coverage box reaches the outer seats the feasible band is only 0.76–0.90 m
wide, so there is no travel left to win by crossing it, and the eight >150° cases
need a large base rotation at that placement regardless of rung. Standoff stays
dominant; the parameter was removed rather than left as a dead knob.

The lesson is §23.9 item 1: **endpoint ranking is the wrong lever for
contortion.** The skill ranks where the arm ends up, and the complaint is about
how it gets there.

### 23.8 Current numbers

```text
unit suite:  269 passed
SFP sweep (strict tier, 144 cases, per-seat gate active):
    coverage   found     passed    standoff     joint max     joint total
    sector      22/144    22/144   0.73-0.85     18-151deg      43-374deg
    narrow     138/144   138/144   0.76-0.90      9-174deg      36-347deg
    shipped    138/144   138/144   0.76-0.90      9-168deg      33-347deg
```

Strict-tier numbers: a "no pose" here is recovered by the relaxation ladder, so
138/144 is a floor. **SC and NIC sweeps were not re-run** — their policy is
untouched, but that is an assumption, not a measurement.

### 23.8b An unusable insignia was reported as success (fixed)

Field run 2026-07-28 19:28. The mask gate admitted a triplet, Stage 2 then refused
it, and the skill returned **`success=True done=False`**:

```text
W  no camera holds a fully-framed insignia; accepting one clipped by up to 12 px
I  board visibility: success=True seen=True done=False target=False
   msg=... needs 1 accepted insignia pose estimates and has 0;
       rejected: center_camera=board reprojection error 18.91px exceeds 8.00px
```

`purple_area` was 0.0003/0.0019/0.0000 — the bracket held 0.19% of the centre
image, against the 0.45% that §21.4 already called too weak a range measurement.

**The bug was classification, not detection.** Two Stage-2 failures need opposite
responses from the caller and both went through `_stage2_not_done()`:

| Failure | Means | Caller must |
| --- | --- | --- |
| geometric refusal | board located, no safe view exists | retry or branch |
| localization failure | no board pose at all | **move the arm** |

Because the second returned `success=True`, a process branching on `success` saw a
healthy skill, never ran its reposition-and-retry fallback, and re-invoked from the
same unusable pose indefinitely. The 18:07 log shows the fallback working correctly
when the *mask* gate raises — two raises, then a successful third invocation — so
the mechanism was already there and this path simply was not wired into it.

Both localization failures now raise `InsigniaNotExposedError`, the same signal the
mask gate uses, with the remedy in the message:

- no accepted insignia pose estimate (PnP reprojection / ambiguity / centroid);
- accepted estimates that do not agree within 5 cm / 8°.

Geometric refusals are untouched and still return `success=True, done=False`.

Note this is the limit of what §22.5's shared `_cameras_with_usable_landmark` can
do. The gate and Stage 2 agree about *edge clipping* because they share that rule,
but the gate cannot know a PnP quality it has not run — reprojection, ambiguity
ratio and centroid error are only knowable after the solve. The two can therefore
always disagree about pose quality; what matters is that the disagreement is
classified by its remedy, which is what this change does.

Two tests pin it: the raise sites and their remedy text, and that nothing between
the raise and `execute` swallows it (`execute` orders
`except InsigniaNotExposedError` ahead of `except Exception`, holds it until the
controller handoff is published, then re-raises as `SkillError(9, ...)`).

**§18 item 1 is now more urgent, not less.** Raising still aborts the BT before
`Switch To Default Controller`, and this change makes the raise fire more often.

### 23.9 What is still open

1. **Contortion is not fixed by any cap, and cannot be.** The skill publishes a
   *Cartesian* pose; Move Robot re-solves it and picks its own branch (§11). Every
   gate here constrains the endpoint the skill predicts, not the path taken. The
   field's own next step — driving the move natively from the selected six-joint
   branch instead of handing a Cartesian TCP to Move Robot — is the actual fix, and
   it also retires §7.1's all-branches-clear compromise, the "cannot cover a branch
   outside `solve_ranked`'s set" limitation, and the 1193-point/29 s transits in
   §11.
2. **The 140 mm wrist-camera keep-out is now the dominant gate.** The 18:07 trace
   reads `probed=68 unreachable=0 camera_keepout=61 arm_in_view=2 arm_clear=5` — it
   is refusing 90% of reachable framed poses, so it is what limits how good a
   branch the search can pick. §21.3 and §22.9 item 5 still apply: it was
   calibrated from one planner rejection whose best branch sat at 111 mm, measured
   to a capsule *centreline* (~90 mm real surface clearance), and 110 mm recovered
   cases offline. Checking it against the workcell planner's mesh test is the
   cheapest remaining win.
3. **`sfp_sweep_runner.py` can no longer reproduce either regression**, because the
   seat gate fixes them from inside. Add a `--no-seat-gate` switch before the next
   policy change, or the harness stops being able to fail.
4. The insignia removal means a survey pose may crop the insignia. Nothing observed
   needs it, but a chained SFP → NIC → SC sequence has not been re-run on hardware
   since.

## 24. Branch divergence is measured, and the keep-out is why the gate misses it

Field session 2026-07-28 20:12-20:14, three chained invocations (SFP → NIC → SC)
on the §23 build. **Nothing in this section is implemented.** It is the next
session's work, and §24.4 is the plan.

### 24.1 How to read the executed branch out of any log

There is no extra instrumentation needed, and this should be the first move on any
future report. Each invocation logs `seed_deg=` — the joints it started from — so:

> **Invocation N+1's `seed_deg` is where invocation N's move actually ended.**

For the last invocation in a chain there is no follow-up, so run the skill once
more after the move and read its `seed_deg` (or `ros2 topic echo /joint_states
--once`).

### 24.2 The measurement

SFP, 20:12. `target_deg` is what the skill predicted; the NIC invocation's
`seed_deg` 71 s later is what the arm actually did:

```text
skill predicted   [-212.5, -101.2,  56.4,  -60.6, 266.3, 27.8]  TCP (-0.3738, 0.1028, 0.5968)
Move Robot ran    [  16.5,  -80.1, -52.6, -129.9,  76.1, 76.4]  TCP (-0.3734, 0.1027, 0.5966)
```

Same TCP to **0.5 mm**, and **229° apart on J1** — a different arm configuration,
not a variant. §11 confirmed rather than suspected.

Then the part that matters:

```text
solve_ranked returned 1 branch    (140 mm keep-out applied)
solve_all    returned 4 branches
executed branch present in solve_ranked?   NO
executed branch self_clearance = 122 mm    (keep-out demands >= 140)
```

**The 140 mm keep-out filtered out the branch Move Robot chose, so the arm-in-view
gate never evaluated it.** §7.1's all-branches-clear rule validated 1 of 4 branches
while the planner selected from all 4. Its own caveat — "cannot cover a branch
outside `solve_ranked`'s set" — is exactly what fired.

Divergence is **intermittent**, which is worth knowing before chasing it: the NIC
invocation's predicted `target_deg=[-20.0,-112.3,-10.6,-138.5,84.2,40.3]` equals
the SC invocation's `seed_deg` exactly, so that move executed as predicted. SFP
diverged, NIC did not.

### 24.3 Two conclusions, one of them the fix

**The keep-out is a pre-filter, and that is the bug.** Two independent questions
are being conflated:

| Question | Right instrument | Should match |
| --- | --- | --- |
| will the planner accept this configuration? | `self_clearance` keep-out | the workcell planner's mesh test |
| is the view usable on whatever branch runs? | arm-in-view | every branch the planner could pick |

`solve_ranked` applies the first *before returning*, so the second only ever sees
the survivors. Over-conservatism in the keep-out therefore does not merely cost
availability — **it makes the gate blind to precisely the branches most likely to
be chosen**, because a low-clearance branch is both the one our model rejects and
the one that puts a camera near the forearm.

**140 mm is provably tighter than the real constraint.** The workcell planner
executed a 122 mm branch without complaint. This settles the §22.9 item 5 /
§21.3 question with hardware evidence instead of a single inferred rejection: 140 mm
was calibrated from one planner refusal whose best branch sat at 111 mm, measured
to a capsule *centreline* (~90 mm of real surface clearance), and the cell has now
demonstrably run 122 mm.

### 24.4 The plan — four changes, measured together

The field instruction is explicit: **do these, hold native motion.** They interact,
so sweep them as a set, not one at a time. The unifying idea, in the field's own
words: *lower the keep-out, then reject any pose that sees the arm on any joint
branch — the filtering gets more accurate, not looser.* Lowering the keep-out
**widens** the branch set, and requiring all of it arm-clear is then a **stronger**
gate than today's, because it finally covers the branches the planner can pick.

1. **Lower `min_self_clearance_m` from 140 mm.** Candidates: 120 mm (just under the
   observed 122), 110 mm (§21.3 measured it recovering three of four start poses
   from 7/8 to 8/8). Ideally validate against the workcell planner's mesh test
   rather than picking a number — that is the §3.3 agreement test from
   `NATIVE_SURVEY_MOTION_PLAN.md`, and it is worth doing even though native motion
   is deferred.
2. **Evaluate arm-in-view over `solve_all`, not `solve_ranked`.** Decouple the two
   questions above: keep the keep-out as the *reachability* gate, but run the
   arm-clear test over every kinematically valid branch. Expect this to cost
   availability — it is strictly stronger — and expect (1) to pay some of it back.
3. **Fix the near-plane hole.** `_arm_clear_of_own_cameras` does
   `if not ahead: continue`, so a capsule whose centreline is behind the camera is
   skipped even though its radius may reach in front. Clip segments at the near
   plane, and reject outright when a camera origin falls inside a link capsule.
   Latent rather than the current failure, but real and cheap.
4. **Fix the J6 preference (§18 item 3).** SC's `j6_error` was **266.7°** this run
   (`preferred_j6_deg=220.3`, achieved -46.4) — the worst on record, against 129-245°
   previously. The ±180° flip exists specifically to keep the camera cluster off the
   occluding side and has never once engaged, because it only wins among candidates
   within `joint_preference_motion_tolerance_rad` = 30° of minimum travel and the
   flipped-wrist family costs far more than that. Either widen the plateau or make
   the flip a hard constraint for SC.

Also worth considering while in here: extend `link_segments` beyond its two
capsules (upper arm, forearm) to the wrist links and the tool. The wrist housings
are fat blobs sitting right behind the camera mount, and a thin centreline capsule
is a poor model of them — see §24.5.

### 24.5 What is NOT yet explained — do not close this early

**The arm-in-view model does not account for the reported occlusion.** At all three
published poses, and on the diverged executed branch as well, every checked link
comes back 0% in frame in all three cameras:

```text
                        self_clear   forearm surface -> camera origin (mm)
SFP predicted              181 mm     left +177  center +179  right +131
SFP executed (diverged)    122 mm     0% in frame in all three cameras
NIC predicted              166 mm     left +181  center +172  right +116
SC  predicted              159 mm     left +109  center +168  right +183
```

Every link centreline sits 70-114° off the optical axis against a **31.9° diagonal
half-FOV** (25.0° horizontal, 22.5° vertical). The wrist links sit at a constant
71/103 mm from the cameras at *every* pose, including the two that looked fine, so
nothing there distinguishes the bad SC view.

So one of these is true and it must be settled before trusting any fix:

- the capsule model under-represents the hardware (wrist housings, camera mount,
  the neighbouring cameras themselves — none of which are in `link_segments`);
- the screenshots were not taken at the survey pose. The joint readout supplied
  alongside them was `[-9.15,-77.58,-95.38,-97.01,90.01,80.84]` — exactly `HOME`,
  not any published target — so this is a live possibility.

**Get `seed_deg` from one extra invocation after the SC move first.** If the SC
branch also diverged, this is one story; if it executed as predicted and the arm is
still in frame, the model is wrong and (1)-(4) will not fix it.

### 24.6 Also from this run: the SFP changes behaved as designed

```text
tier='strict' joint_cap=225 total_cap=400 -> none | probed=1376 unreachable=697
    camera_keepout=10 seat_hidden=326 arm_in_view=315 arm_clear=28 took=42.25s
tier='joint-travel caps lifted' joint_cap=360 total_cap=none -> FOUND | probed=76
    ... arm_clear=1 best_worst_joint=176.3deg took=12.17s
published: standoff=0.800 joint_max=176.3 joint_total=586.1
```

Read honestly:

- **`seat_hidden=326`** — the §23.3 seat gate is doing real work on hardware, rejecting
  326 candidates whose coverage box framed fine while a module hid behind the tool.
- **standoff 0.800 m** — the §23.2 coverage widening selected exactly what the sweep
  predicted.
- **The 400° total cap did not prevent contortion, it only added a tier.** Strict had
  28 arm-clear candidates and `best_worst_joint=161.6°` (inside the 225° cap), so the
  total cap was the binder; the next tier lifted it and published **586.1°**. At this
  board placement no pose under 400° exists, so the cap converts a contorted pose
  into a *warned* contorted pose. That is better than silence and is not a fix.
  Reordering the ladder so the total cap is relaxed last is worth measuring.
- **Time 54.4 s** (42.25 + 12.17), down from 154.86 s but not the ~3 s the offline
  sweep suggests. The cost is now the *failing* tier: `probed` goes 68 → 1376 because
  a tier that finds nothing must probe every framed candidate in every standoff
  group instead of stopping at the first success. Cheap win available: order the
  cheap seat gate before the IK solve inside `select_clear_ik_solution` (it already
  is), and consider bailing a standoff group early once its candidates are exhausted.
