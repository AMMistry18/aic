# AIC Flowstate skills

## Check Board Visibility

`ai.tar2.check_board_visibility_skill_v4` uses only documented participant data:
three wrist-camera image topics, measured `/joint_states`, controller state,
wrist wrench, and
robot-mounted TF. Its TF access is code-restricted to these pairs:

```text
base_link <- left_camera/optical
base_link <- center_camera/optical
base_link <- right_camera/optical
base_link <- gripper/tcp
```

It does not request task-board, port, module, cable, Gazebo, entity-state, or
scoring transforms. One invocation performs the camera/motion loop itself. If
no camera sees the board, it sends a bounded joint-1 horizontal acquisition
sweep as a small base-Z Cartesian TCP arc through
`/aic_controller/pose_commands`. This target rotates TCP position and
orientation together and is the local forward-kinematics motion of shoulder
pan, without requiring the live controller to accept a joint-target-mode
switch. Detection does not end horizontal alignment: a visible left/right edge,
missing camera, or low board-area fraction seeds a small yaw probe.
The result is scored across all three cameras, prioritizing camera count and
then the worst camera's visible board area. The search continues toward an
improving yaw and tests the opposite side when a probe makes coverage worse.
Only after all three cameras meet the horizontal coverage target does it send
base-frame `+Z` Cartesian clearance steps through
`/aic_controller/pose_commands`, passing the measured TCP orientation through
unchanged.

The search is feedback-driven, not random and not a fixed sequence. If neither
yaw direction improves three-camera coverage, it restores the best measured
yaw, takes one upward escape step, and immediately repeats horizontal
optimization at the new height. It does not stack upward moves before that yaw
retry. The visible-board policy never pitches the camera, combines rotation
with translation, or approaches an uncertain board. There is no fixed
move-count terminal.

`done` first requires every configured wrist camera to see at least 12% board
pixels by default. After that horizontal gate latches, the board silhouette plus
a dynamic 5%-of-projected-board context envelope must remain inside one usable
camera frame for two fresh snapshots. It also requires enough board pixels for
downstream NIC/SC detail, a sufficiently clean silhouette, and no broad
board-body contact with the gripper-exclusion boundary. A narrow arm/finger
bridge into that band is removed before this contact test. This is a geometric
coverage guarantee for the plate and protruding component zones; IVM downstream
remains responsible for semantic NIC/SC pose detection.

The internal motion safeguards are:

- quintic minimum-jerk Cartesian setpoints at 20 Hz;
- at most 0.04 m/s and 0.20 rad/s, with coarse-to-fine 0.05-0.15 rad
  visible-board yaw actions;
- a 90 second overall deadline, 0.50 m start-relative TCP envelope, 0.80 m
  cumulative translation, 1.60 rad start-relative J1 yaw, and 2.20 rad
  cumulative angular travel by default;
- a finite four-leg joint-1 scan when no camera detects the board, plus
  image-feedback joint-1 centering when a partial board is already visible;
- no motion above 18 N absolute wrist force (2 N below the documented 20 N
  scoring threshold) or a 5 N change from the initial force baseline;
- immediate reversal to the beginning of the current step on force or
  cancellation;
- measured-TCP pose settling and controller-subscriber checks after every move;
- confirmation of a strictly newer joint-mode controller sample before any
  shoulder-pan command; and
- bounded time, force, displacement, and cumulative-travel termination rather
  than a trial-step limit.

`aic_controller` provides command clamping, smoothing, impedance control, and a
tracking-error safety reset. It is not the Flowstate world-model motion planner,
so the survey pose and the full configured search envelope must be free space. The
skill cannot guarantee collision-free motion from arbitrary starting poses.

Inputs are the skill parameters in `CheckBoardVisibilitySkillParams`; no world
objects or task poses are passed in. The ROS bridge must expose the three image
topics, `/fts_broadcaster/wrench`, `/aic_controller/controller_state`, the
measured `/joint_states` topic, Cartesian command and target-mode interfaces,
and the four allowlisted
robot-mounted TF pairs above.

Build from a Linux/AMD64 workspace laid out as:

```text
ws_aic_phase1/
  src/aic/
  src/sdk-ros/
```

```bash
cd ~/ws_aic_phase1
bash src/aic/flowstate/scripts/build_check_board_visibility_skill.sh
```

Install against the current cluster, which must be re-read after a simulator
restart:

```bash
inctl asset install \
  --org tar-2@xfa-prod-aic-us \
  --cluster "$CLUSTER" \
  images/check_board_visibility_skill/check_board_visibility_skill.bundle.tar
```

Recommended serial process wiring:

```text
Move Robot (fixed survey pose)
Switch To AIC Controller
Check Board Visibility (the loop and motion happen inside this skill)
Switch To Default Controller
Require result.success == true && result.done == true
IVM NIC estimate -> filter estimates -> remaining process
```

Do not run Move Robot, another Insert Cable policy, or any other motion session
in parallel with this skill. Always switch back to the default controller before
using Flowstate Move Robot again. Route both the successful and unsuccessful
board-search result through `Switch To Default Controller`; validate the returned
`success` and `done` fields only after that cleanup node. Expected sensor/search
failures are returned as `success=false` so the serial cleanup step still runs.
Cancellation still aborts execution and requires the process's cancellation
cleanup path to switch back to the default controller. `target`, `dx/dy/dz`, and
`target_valid` remain
as diagnostics for the last attempted internal move; they are no longer wired
to another skill. Use `moves_executed`, `travel_m`, `force_abort`, `seen`, and
`done` for process diagnostics.
