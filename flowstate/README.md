# AIC Flowstate skills

## Check Board Visibility

`ai.tar2.check_board_visibility_skill_v4` uses only documented participant data:
three wrist-camera image and CameraInfo topics, measured `/joint_states`,
controller state, wrist wrench, and
robot-mounted TF. Its TF access is code-restricted to these pairs:

```text
base_link <- left_camera/optical
base_link <- center_camera/optical
base_link <- right_camera/optical
base_link <- gripper/tcp
```

It does not request task-board, port, module, cable, Gazebo, entity-state, or
scoring transforms. TF is resolved at each image timestamp. One invocation
performs the camera/motion loop itself. If
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

Stage 1 is feedback-driven, not random and not a fixed sequence. If neither
yaw direction improves three-camera coverage, it restores the best measured
yaw, takes one upward escape step, and immediately repeats horizontal
optimization at the new height. It does not stack upward moves before that yaw
retry. Stage 1 never pitches the camera or approaches an uncertain board.
There is no fixed move-count terminal.

For NIC/SC routes, legacy `done` first requires every configured wrist camera
to see at least 12% board
pixels by default. After that horizontal gate latches, the board silhouette plus
a dynamic 5%-of-projected-board context envelope must remain inside one usable
camera frame for two fresh snapshots. It also requires enough board pixels for
downstream NIC/SC detail, a sufficiently clean silhouette, and no broad
board-body contact with the gripper-exclusion boundary. A narrow arm/finger
bridge into that band is removed before this contact test. This is a geometric
coverage guarantee for the plate and protruding component zones; IVM downstream
remains responsible for semantic NIC/SC pose detection.

For `STAGED_SFP_MODULE`, Stage 1 hands off only with a complete unobstructed
purple landmark. A calibrated CAD/PnP Stage 2 then searches one deterministic
board-relative survey pose. The complete conservative staged-SFP envelope and
individual legal-seat detail probes must be fully inside all three images,
clear of every conservative gripper mask by at least 32 pixels, in two fresh
triplets with at most 50 ms skew. Every camera must produce a board pose that
agrees with the plan and the other cameras before `done=true`.

The internal motion safeguards are:

- quintic minimum-jerk Cartesian setpoints at 20 Hz;
- at most 0.05 m/s and 0.30 rad/s, with direct-joint moves capped at 0.20 rad/s
  and coarse-to-fine 0.05-0.15 rad
  visible-board yaw actions;
- a 60 second overall deadline, 0.50 m start-relative TCP envelope, 0.80 m
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
tracking-error safety reset. It is not the Flowstate world-model motion planner
and this repository exposes no supported IK/collision-query service. Stage 2
therefore limits orientation change to 45 degrees, retreats with orientation
held, rotates only beyond a conservative 0.40 m camera-rig sweep radius, then
translates with final orientation fixed. It still fails closed outside its
guarded reachable range and cannot guarantee arbitrary-start collision-free
motion.

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

Recommended serial process wiring.

For geometric survey targets (perception-only; Move Robot executes the move):

```text
Move Robot (fixed pre-position that frames the board)
Switch To AIC Controller        (only if Stage 1 must move to expose the insignia)
Check Board Visibility          -> result.target.{x,y,z,qx,qy,qz,qw}
Switch To Default Controller
Require result.success && result.done && result.target_valid
Python code node: pack the seven scalar fields into the existing TCP Cartesian pose
Move Robot: Cartesian target
    moving frame  = gripper TCP
    target frame  = base_link (root)
    target frame offset = Python code-node output
IVM NIC estimate -> filter estimates -> remaining process
```

For `SC_DESTINATION_PORT`, configure the Move Robot segment's absolute J1..J6
position limits as constants; they are not skill outputs:

```text
min deg = [-53.6, -187.0, -122.4, -127.7, -116.1, -71.5]
max deg = [170.1,  -28.9,   94.1,   43.8,  114.8, 180.8]

min rad = [-0.9355, -3.2638, -2.1363, -2.2288, -2.0263, -1.2479]
max rad = [ 2.9688, -0.5044,  1.6424,  0.7645,  2.0036,  3.1556]
```

The skill mirrors those limits internally when selecting a Cartesian target,
but preserves the deployed seven-scalar output interface.

SC survey poses use a mandatory 10-13 degree board-X displacement normal to the
adapters' board-Y long face (at most 2 degrees along that face/port rows), at a
0.62 m standoff. The axis is explicit rather than inferred from the cluster box
and rotates into the base frame with the estimated board orientation. All three
cameras must fully frame the sector and remain gripper-clear; for every mouth,
at least two cameras must also retain a positive rectangular-bore margin and at
least 3.0 px of projected mouth-to-back-centre depth cue. SC also prefers the
best legal J6 half-turn from the live start, but only inside a 30-degree
worst-motion plateau. Arm-in-view, fixed joint-window, and 220-degree SC motion
gates remain authoritative.

Do not run Move Robot, another Insert Cable policy, or any other motion session
in parallel with this skill. Always switch back to the default controller before
using Flowstate Move Robot again. Route both the successful and unsuccessful
board-search result through `Switch To Default Controller`; validate the returned
`success` and `done` fields only after that cleanup node. Expected sensor/search
failures are returned without a valid/done target so the serial cleanup step
still runs.
Cancellation still aborts execution and requires the process's cancellation
cleanup path to switch back to the default controller. `target`, `dx/dy/dz`, and
`target_valid` remain
as the deployed Cartesian handoff and diagnostics. Use `moves_executed`,
`travel_m`, `force_abort`, `seen`, and
`done` for process diagnostics.
