# AIC Flowstate skills

## Check Board Visibility

`ai.tar2.check_board_visibility_skill` uses only documented participant data:
three wrist-camera image topics, controller state, wrist wrench, and
robot-mounted TF. Its TF access is code-restricted to these pairs:

```text
base_link <- left_camera/optical
base_link <- center_camera/optical
base_link <- right_camera/optical
base_link <- gripper/tcp
```

It does not request task-board, port, module, cable, Gazebo, entity-state, or
scoring transforms. One invocation performs the camera/motion loop itself. It
sends bounded full-pose Cartesian setpoints to `/aic_controller/pose_commands`,
waits for measured TCP position and orientation settling, then scores a fresh
three-camera snapshot before choosing another move. Camera pitch/yaw changes
are Cartesian orientation targets, so the controller can coordinate all six
joints without blind joint-space sweeps.

The search is feedback-driven, not random and not a fixed sequence. It selects
among camera-plane translation, optical-axis backoff/approach, camera aiming,
and combined corrections. An action that makes the image worse is rolled back
and blacklisted for that visual state; persistent edge evidence rotates through
other safe action types and cameras. There is no fixed move-count terminal.

`done` requires the board silhouette plus a dynamic 20%-of-projected-board
context envelope to remain inside one usable camera frame for two fresh
snapshots. It also requires enough board pixels for downstream NIC/SC detail,
a sufficiently clean silhouette, and no contact with the gripper-exclusion
boundary. This is a geometric coverage guarantee for the plate and protruding
component zones; IVM downstream remains responsible for semantic NIC/SC pose
detection.

The internal motion safeguards are:

- quintic minimum-jerk translation plus shortest-path quaternion SLERP at 20 Hz;
- at most 0.025 m/s and 0.12 rad/s, with nominal 0.02 m and 0.07 rad actions;
- a 90 second overall deadline, 0.25 m start-relative TCP envelope, 0.50 m
  cumulative translation, 0.35 rad start-relative orientation, and 1.2 rad
  cumulative angular travel by default;
- no blind move when the board is not detected;
- no motion above 18 N absolute wrist force (2 N below the documented 20 N
  scoring threshold) or a 5 N change from the initial force baseline;
- immediate reversal to the beginning of the current step on force or
  cancellation;
- measured-TCP pose settling and controller-subscriber checks after every move;
- full-pose rollback after view regression or guarded motion failure; and
- explicit stagnation failure only after all safe action types/cameras for the
  current visual states have been exhausted.

`aic_controller` provides command clamping, smoothing, impedance control, and a
tracking-error safety reset. It is not the Flowstate world-model motion planner,
so the survey pose and the full configured search envelope must be free space. The
skill cannot guarantee collision-free motion from arbitrary starting poses.

Inputs are the skill parameters in `CheckBoardVisibilitySkillParams`; no world
objects or task poses are passed in. The ROS bridge must expose the three image
topics, `/fts_broadcaster/wrench`, `/aic_controller/controller_state`, the
Cartesian command and target-mode interfaces, and the four allowlisted
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
using Flowstate Move Robot again. `target`, `dx/dy/dz`, and `target_valid` remain
as diagnostics for the last attempted internal move; they are no longer wired
to another skill. Use `moves_executed`, `travel_m`, `force_abort`, `seen`, and
`done` for process diagnostics.
