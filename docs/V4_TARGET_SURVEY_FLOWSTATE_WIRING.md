# Flowstate v4 target-survey wiring

Use one `check_board_visibility_skill_v4` asset in three explicit target modes.
The SC-plug estimate reuses the saved SFP-source camera pose, because removing
the SFP cable exposes that plug at the same useful viewpoint.

## Select the target mode

Select a v4 node, then open **Properties -> Inputs -> Skill parameters ->
survey_target** and choose:

| `survey_target` | View optimized for | Run before |
| --- | --- | --- |
| `STAGED_SFP_MODULE` | The staged SFP cable/module source region | SFP-module IVM |
| `NIC_SFP_DESTINATION` | All five NIC cards and their SFP destination ports | NIC-card/SFP-port IVM |
| `SC_DESTINATION_PORT` | The five blue SC destination ports | SC-port IVM |
| `UNSPECIFIED` | The existing whole-survey behavior | Only unmigrated trees |

`UNSPECIFIED` remains value zero, so an existing saved v4 node keeps its legacy
behavior. New nodes should select an explicit target.

## Add the SFP survey-pose output node

Immediately after the `STAGED_SFP_MODULE` v4 node, add a **Run Python Script**
node named `save SFP survey pose`. It has no inputs. Create exactly one output:

| Property | Value |
| --- | --- |
| Name | `saved_sfp_survey_pose` |
| Display name | `saved_sfp_survey_pose` |
| Parameter type | `Pose` (`intrinsic_proto.Pose`) |
| Set as list | Off |
| Set as optional | Off |

Do not use `PoseEstimateInRobot`: Move Robot's Cartesian target offset consumes
an `intrinsic_proto.Pose` directly.

Flowstate generates the imports, the `gen.<node-specific-id>` import, the
`compute(...)` declaration, and the output comment. Keep that generated header
and replace the complete body after the `compute` declaration with this code:

```python
  # Capture root_t_tool0 after the SFP-source survey has finished. The robot's
  # available TCP frame is tool0; gripper/tcp is not present in this workcell.
  from intrinsic.math.python import proto_conversion

  world = context.object_world
  robot = world.get_kinematic_object("robot")
  tool0 = robot.get_frame("tool0")

  root_t_tool0 = world.get_transform(
      node_a=world.root,
      node_b=tool0,
  )

  output = code_execution_pb2.ReturnValue()
  output.saved_sfp_survey_pose.CopyFrom(
      proto_conversion.pose_to_proto(root_t_tool0)
  )

  print(
      "Saved SFP survey pose:",
      output.saved_sfp_survey_pose,
  )
  return output
```

This reads only Object World state; it does not acquire or move the arm.

## Return to the saved pose

After the entire SFP insertion process, add a **Move Robot** node before the
SC-plug IVM call. Configure its Cartesian motion target as follows:

| Move Robot field | Value |
| --- | --- |
| Robot | `robot_controller` |
| arm_part | `robot` |
| motion_target | `cartesian_pose` |
| moving_frame | `robot/tool0` |
| target_frame | `root` |
| target_frame_offset | Connect `save SFP survey pose.saved_sfp_survey_pose` |
| motion_type | `ANY` |

`moving_frame` is the frame being moved and must be `robot/tool0`; it is not
`root`. The saved value is `root_t_tool0`, so `root` is the target/reference
frame and the Pose output is its offset.

## Recommended process-tree sequence

Use ordinary **Sequence** control flow:

```text
SFP operation
  check_board_visibility_skill_v4
    survey_target = STAGED_SFP_MODULE
  Switch To Default Controller
  save SFP survey pose
    output = saved_sfp_survey_pose
  estimate_pose_ivm_cloud                 # SFP-module model
  filter SFP-module estimates             # module paired to chosen NIC
  create object / store SFP-module belief

  check_board_visibility_skill_v4
    survey_target = NIC_SFP_DESTINATION
  Switch To Default Controller
  estimate_pose_ivm_cloud                 # NIC card / SFP destination model
  filter NIC-card or SFP-port estimates
  create object / store destination belief

  complete the entire SFP grasp, extraction, and insertion process

SC operation
  Move Robot to saved_sfp_survey_pose
  estimate_pose_ivm_cloud                 # SC-plug model; no new v4 search
  filter SC-plug estimates
  create object / store SC-plug belief

  check_board_visibility_skill_v4
    survey_target = SC_DESTINATION_PORT
  Switch To Default Controller
  estimate_pose_ivm_cloud                 # SC-port model
  filter SC-port estimates
  create object / store destination belief

  complete the SC grasp and insertion process
```

Do not run v4 and Move Robot in parallel: both use the arm. Keep the explicit
`Switch To Default Controller` node after every v4 call. The skill stops its
command stream and releases its invocation lock before returning, and the
switch hands control back to the controller expected by subsequent arm skills.
No sleep node is required.

## Perception dataflow

`survey_target` configures camera positioning only; it is not passed to IVM.
Connect each IVM `pose_estimates` output to its matching filter, then connect
the filter's selected `pose_estimates` / `root_ts_target` output to Create
Object, as in the existing SFP flow.

Keep `selected_module_name` connected to filters that preserve absolute task
pairing. SFP-module and SC-module selection must derive their index from the
chosen `nic_card_mount_0` through `nic_card_mount_4`; do not reinterpret that
input as an `sc_port_*` name.
