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
scoring transforms. TF is resolved at each image timestamp.

Stage 1 is a deterministic acquisition policy:

1. capture one fresh calibrated three-camera triplet;
2. hand it directly to Stage 2 if the existing Stage-2 detector finds a
   complete unobstructed purple insignia in any camera;
3. otherwise validate and execute one offline-swept observation joint path;
4. capture one new triplet and either hand it to Stage 2 or return normally
   with `success=true, done=false, target_valid=false`.

The observation posture was swept over 8 board yaws, 2 tilts, 3 placements,
and 3 historical live starts. All 144 cases fully frame the insignia in at
least one calibrated camera. The policy has no image-gradient servo, J1/J6
phase machine, polarity learning, or open-ended correction loop. A wider
hardware-proven purple HSV detector is logged as an acquisition cue only; the
unchanged Stage-2 polygon detector remains the completion authority.

Before motion, the exact six-joint interpolation is checked against physical
joint limits, live-calibrated UR5e FK, wrist/forearm self-clearance, endpoint
arm-in-camera exclusion, TCP height/reach, a 185-degree worst-joint cap, and a
250-degree total-joint cap. The known 501.3-degree failure is therefore
rejected. The path is split into bounded minimum-jerk transactions, and every
transaction reverses to its measured start on force, cancellation, stale
feedback, mode change, or settling timeout. Exhaustion leaves the arm at the
validated observation posture rather than at an arbitrary failed search exit.

For every deployed `survey_target`, Stage 2 is unchanged: it PnPs the complete
purple landmark, searches the target-specific board-relative survey pose, and
publishes that pose for downstream Move Robot. Stage 2 commands no motion.

The internal motion safeguards are:

- quintic minimum-jerk six-joint setpoints at 20 Hz;
- direct-joint speed capped at 0.20 rad/s;
- deterministic segmentation sized from the configured per-move timeout;
- 185-degree worst-joint and 250-degree total Stage-1 travel caps;
- live-calibrated FK, joint-limit, self-clearance, arm-in-camera, height, and
  reach validation before the first command;
- no motion above 18 N absolute wrist force (2 N below the documented 20 N
  scoring threshold) or a 5 N change from the initial force baseline;
- immediate reversal to the beginning of the current step on force or
  cancellation;
- measured full-joint settling and controller-subscriber checks after every
  segment;
- confirmation of a strictly newer joint-mode controller sample before every
  segment; and
- exactly two observations and one bounded path rather than an open search.

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

The build must report both cold-start smoke tests reaching
`gRPC server listening`, and it fails if the image labels do not match the
manifest:

```text
ai.intrinsic.asset-id=ai.tar2.check_board_visibility_skill_v4
ai.intrinsic.skill-image-name=check_board_visibility_skill_v4
```

Those labels are part of Flowstate's skill-image lifecycle contract. The
previous non-v4/missing metadata is the leading repository-side cause of the
fresh-install-only behavior: it can allow the first pod to start while leaving
the skill unavailable after a solution stop/start reconciliation. Confirm the
fix with one real stop/start cycle after installing the rebuilt bundle.

The OCI filename is the third identity-bearing value. The generated bundle
must list:

```text
check_board_visibility_skill_v4.tar
```

Do not install a bundle that still contains
`check_board_visibility_skill.tar`; that disagrees with the v4
`skill-image-name` and can leave the asset installed without a runnable skill
workload.

The saved image must also carry the SDK-standard repository tag and logical
name:

```text
RepoTags:  aic_perception:check_board_visibility_skill_v4
SKILL_NAME=check_board_visibility_skill_v4
```

`SKILL_CONFIG_NAME=check_board_visibility_skill` intentionally remains the
source/config basename.

Install against the current cluster, which must be re-read after a simulator
restart:

```bash
inctl asset install \
  --org tar-2@xfa-prod-aic-us \
  --cluster "$CLUSTER" \
  --policy update_compatible \
  images/check_board_visibility_skill_v4/check_board_visibility_skill_v4.bundle.tar
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

Do not add task-specific absolute J1..J6 position bounds to the Move Robot
segment. The survey skill treats the measured live joint vector as the motion
origin, expresses each analytic IK branch at the physically equivalent winding
nearest that origin, and rejects SC candidates whose worst relative joint
travel exceeds 185 degrees. The deployed seven-scalar Cartesian output
interface is unchanged.

SC survey poses use a mandatory 10-13 degree board-X displacement normal to the
adapters' board-Y long face (at most 2 degrees along that face/port rows), at a
0.62 m standoff. The axis is explicit rather than inferred from the cluster box
and rotates into the base frame with the estimated board orientation. All three
cameras must fully frame the sector and remain gripper-clear; for every mouth,
at least two cameras must also retain a positive rectangular-bore margin and at
least 3.0 px of projected mouth-to-back-centre depth cue. SC also prefers the
best physical J6 half-turn from the live start, but only inside a 30-degree
worst-motion plateau. Arm-in-view and the 185-degree live-relative SC motion
gate remain authoritative.

The skill can rank and reject Cartesian targets using live-relative IK, but the
seven-scalar pose does not encode the selected joint branch. Move Robot remains
authoritative for the executed trajectory. Keep conservative velocity and
acceleration limits, and validate the generated path before hardware execution.

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

## Pose KV Store

`ai.tar2.pose_kv_store_skill_v1` carries the five labelled module poses of each
type from the one-time survey in Participant Initialize into every later trial,
through the Flowstate key-value store. It is one skill with two modes over one
key scheme:

```text
aic/phase1/sfp/0 .. /4
aic/phase1/nic/0 .. /4
aic/phase1/sc/0  .. /4
aic/phase1/home
```

The last segment is the module's label, so the key index *is* the order and no
separate order blob is stored. `poses[i]` is written to `.../{type}/{i}`, and a
read resolves one index and returns that one `intrinsic_proto.Pose`. `home` is
the one type that holds a single pose, so its key carries no index.

This skill is C++ rather than Python like the two above. The key-value store is
reached through `intrinsic::PubSub().KeyValueStore()`, and the ROS CMake SDK
this repository builds against excludes `intrinsic/platform/pubsub/python/` from
its build, so the Python `KeyValueStore` bindings do not exist inside a skill
image. The zenoh session is created on the first invocation and reused.

### Write mode

```text
mode         = MODE_WRITE
object_type  = OBJECT_TYPE_SFP | OBJECT_TYPE_NIC | OBJECT_TYPE_SC
poses        = exactly 5 poses, poses[i] is the module labelled i

mode         = MODE_WRITE
object_type  = OBJECT_TYPE_HOME
pose         = the one home pose
```

For the three indexed types, anything other than five poses is rejected instead
of written, because a partially filled type makes every later read by index
meaningless; `home` takes exactly one pose instead. Each pose
must also carry a set, finite, non-degenerate quaternion: an all-zero
orientation reaching Move Robot is a crash rather than a bad move, and it is
cheaper to reject it here than to debug it three skills later. Each key is set
with `high_consistency`, and the result lists `keys_written`.

Wire it once, in Participant Initialize, after the three IVM surveys:

```text
IVM x 3
store_sfp -> kv write (object_type=SFP,  poses=sfp_modules)
store_nic -> kv write (object_type=NIC,  poses=nic_cards)
store_sc  -> kv write (object_type=SC,   poses=sc_ports)
             kv write (object_type=HOME, pose=home_pose)
```

### Read mode

```text
mode         = MODE_READ
target_name  = a module name from this trial's process inputs, or home
object_type  = unset, or the type to read at that name's index
```

`target_name` supplies both the index and, by default, the type:
`sfp_mount_2`, `nic_card_mount_3` and `sc_port_1` resolve to `sfp/2`, `nic/3`
and `sc/1`. Matching tolerates case, hyphens, a leading path and a `rail` infix,
but the trailing number and the type word are required, and the index must be
0--4. `index` plus an explicit `object_type` is accepted when there is no name
to parse.

Setting `object_type = OBJECT_TYPE_SFP` together with a NIC `target_name` reads
`aic/phase1/sfp/<nic index>`. That is how a trial picks its cable: the task
needs an unused SFP module, not one specific module, so the NIC card number of
the trial is used as the SFP index. This needs no counter, no initialisation, no
increment-on-success, and no state that can desync when a trial fails halfway.
The one thing it does assume is that a run does not repeat a NIC card number; if
it does, that trial will be pointed at an SFP module the earlier trial already
took. Storing a used-mask in the same key space is the fallback if that ever
shows up, and it is deliberately not implemented yet.

Per trial:

```text
process inputs (this trial's cable and ports)
  -> kv read (target_name=sc_port_*)                        -> sc pose
  -> kv read (target_name=nic_card_mount_*)                 -> nic pose
  -> kv read (target_name=nic_card_mount_*, object_type=SFP) -> sfp pose
  -> kv read (target_name=home)                             -> home pose
  -> load_* offset scripts -> Move Robot
```

A read that finds no key fails the skill rather than returning an empty pose, so
a missing Participant Initialize write stops the trial at the read instead of
sending a zero pose to Move Robot. The message names the key and the write that
is missing. `resolved_type`, `resolved_index` and `key` echo what was resolved,
and the message records where the index came from, which is what makes a
NIC-numbered SFP read readable in the logs.

### Home pose

`OBJECT_TYPE_HOME` stores one pose at `aic/phase1/home` instead of five. Write
it by setting `object_type = OBJECT_TYPE_HOME` and filling the singular `pose`
field --- a one-element `poses` list is accepted too, but setting both is
rejected, as is the `pose` field on an indexed type, because either one means
the parameters were wired to the wrong mode. Read it back with
`target_name = home` or with `object_type = OBJECT_TYPE_HOME` and no name; the
`index` parameter is ignored either way, and the result reports
`resolved_type = "home"` with `resolved_index = -1` to say the key has no index.
Everything else is shared with the indexed types: the same prefix, the same pose
validation, the same "fail the read rather than return an empty pose" behaviour.

Because SFP modules are selected by NIC number, the SFP write does not have to
be rail-labelled the way NIC and SC already are: the five entries only have to
be five distinct, real, unused modules, stable for the run. If `store_sfp`
remains "top 5 by score", `aic/phase1/sfp/2` is simply "the third module that
survey found", which is enough for cable selection but not enough to answer
"give me `sfp_mount_2`". Label `store_sfp` like `store_nic` before relying on
SFP reads by official name.

### Build and install

```bash
cd ~/ws_aic_phase1
bash src/aic/flowstate/scripts/build_pose_kv_store_skill.sh
```

Unlike the Python skills, a C++ skill cannot reach "listening" offline: the
generated entrypoint blocks on the in-cluster world, geometry, motion-planner
and skill-registry addresses before it serves. The cold-start check in the build
script therefore proves the packaging instead --- every shared library resolves,
and the binary starts and loads its own `pose_kv_store_skill_v1` config before
it fails on those addresses.

```bash
inctl asset install \
  --org tar-2@xfa-prod-aic-us \
  --cluster "$CLUSTER" \
  images/pose_kv_store_skill/pose_kv_store_skill.bundle.tar
```

Values live in the workcell's key-value store, so they outlive a process run but
not a solution that is torn down: after a restart, run Participant Initialize
again before any trial reads. `key_prefix` defaults to `aic/phase1` and can be
pointed somewhere else to rehearse a write without touching the live keys.
