import math

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .aic_task_env_cfg import AICTaskEnvCfg
from .mdp.events import randomize_board_and_parts, randomize_dome_light


_PLUG_TIP_IN_WRIST = {
    "sc": ((0.04026, 0.00907, 0.14939), (0.85472, 0.01261, -0.51889, -0.00920)),
    "sc2": ((0.04026, 0.00907, 0.14939), (0.85472, 0.01261, -0.51889, -0.00920)),
    "sfp": ((0.05631, 0.00137, 0.14857), (0.86526, 0.01717, -0.50059, -0.02774)),
}


@configclass
class HybridInsertObservationsCfg:
    """Sensor-rich state observations for rotation-and-seat policy stage."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.arm_joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.arm_joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        eef_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        tcp_vel_port = ObsTerm(
            func=mdp.tcp_velocity_port,
            params={"target_key": "${TARGET_KEY}"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        port_pose_perceived = ObsTerm(
            func=mdp.port_pose_obs_perceived,
            params={"target_key": "${TARGET_KEY}"},
            noise=Unoise(n_min=-0.003, n_max=0.003),
        )
        tip_to_port_delta = ObsTerm(
            func=mdp.tip_to_port_delta_perceived,
            params={"target_key": "${TARGET_KEY}"},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        perception_action_hint = ObsTerm(
            func=mdp.perception_action_hint,
            params={"target_key": "${TARGET_KEY}"},
        )
        scripted_action_hint = ObsTerm(
            func=mdp.scripted_insert_action_hint,
            params={"target_key": "${TARGET_KEY}"},
        )
        perception_valid = ObsTerm(
            func=mdp.perception_valid_flag,
            params={"target_key": "${TARGET_KEY}"},
        )
        wrench = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
        )
        last_action = ObsTerm(func=mdp.last_action)
        tip_axes_port = ObsTerm(
            func=mdp.tip_axes_port,
            params={"target_key": "${TARGET_KEY}"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HybridInsertRotObservationsCfg(HybridInsertObservationsCfg):
    """Hybrid observations plus explicit tip axes for rotation-stage learning."""

    @configclass
    class PolicyCfg(HybridInsertObservationsCfg.PolicyCfg):
        tip_axes_port = ObsTerm(
            func=mdp.tip_axes_port,
            params={"target_key": "${TARGET_KEY}"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class HybridInsertRewardsCfg:
    # Keep policy tightly centered at port while it learns twist + seating.
    approach_xy_coarse = RewTerm(
        func=mdp.tip_to_port_xy_l2, weight=-8.0, params={"target_key": "${TARGET_KEY}"}
    )
    approach_xy_progress = RewTerm(
        func=mdp.tip_to_port_xy_progress,
        weight=20.0,
        params={"target_key": "${TARGET_KEY}", "clip": 0.05},
    )
    approach_xy_fine = RewTerm(
        func=mdp.tip_to_port_xy_exp,
        weight=30.0,
        params={"target_key": "${TARGET_KEY}", "sigma": 0.01},
    )
    approach_z_fine = RewTerm(
        func=mdp.tip_to_port_z_exp,
        weight=24.0,
        params={"target_key": "${TARGET_KEY}", "sigma_z": 0.05, "xy_gate": 0.12},
    )
    approach_z_dense = RewTerm(
        func=mdp.tip_to_port_z_inv,
        weight=14.0,
        params={"target_key": "${TARGET_KEY}", "xy_gate": 0.20},
    )
    approach_descend = RewTerm(
        func=mdp.tip_to_port_descend_reward,
        weight=22.0,
        params={"target_key": "${TARGET_KEY}", "sigma_z": 0.06, "xy_gate": 0.20},
    )
    approach_z_progress = RewTerm(
        func=mdp.tip_to_port_z_progress,
        weight=45.0,
        params={"target_key": "${TARGET_KEY}", "xy_gate": 0.15, "clip": 0.03},
    )
    axis_alignment = RewTerm(
        func=mdp.plug_port_axis_alignment, weight=6.0, params={"target_key": "${TARGET_KEY}"}
    )
    twist_alignment = RewTerm(
        func=mdp.plug_port_twist_alignment, weight=16.0, params={"target_key": "${TARGET_KEY}"}
    )
    twist_progress = RewTerm(
        func=mdp.plug_port_twist_progress,
        weight=50.0,
        params={"target_key": "${TARGET_KEY}", "clip": 0.08},
    )
    insertion_depth = RewTerm(
        func=mdp.insertion_depth_reward,
        weight=420.0,
        params={
            "target_key": "${TARGET_KEY}",
            "xy_gate": 0.012,
            "axis_gate": 0.55,
            "twist_gate": 0.55,
        },
    )
    success_bonus = RewTerm(
        func=mdp.insertion_success_bonus,
        weight=180.0,
        params={
            "target_key": "${TARGET_KEY}",
            "xy_thresh": 0.005,
            "axis_thresh": 0.75,
            "twist_thresh": 0.70,
            "depth_fraction": 0.75,
        },
    )
    alive = RewTerm(func=mdp.alive_bonus, weight=0.05)
    force_penalty = RewTerm(
        func=mdp.wrist_force_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0008)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)


@configclass
class HybridInsertTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insertion_success = DoneTerm(
        func=mdp.insertion_success, params={"target_key": "${TARGET_KEY}"}
    )
    drift_failure = DoneTerm(
        func=mdp.drift_failure, params={"target_key": "${TARGET_KEY}", "xy_thresh": 1.20}
    )
    force_failure = DoneTerm(
        func=mdp.force_failure,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]),
            "force_thresh": 45.0,
            "duration_steps": 60,
        },
    )


@configclass
class HybridInsertEventCfg:
    randomize_light = EventTerm(
        func=randomize_dome_light,
        mode="reset",
        params={
            "intensity_range": (1500.0, 3500.0),
            "color_range": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        },
    )
    randomize_board_and_parts = EventTerm(
        func=randomize_board_and_parts,
        mode="reset",
        params={
            "board_scene_name": "task_board",
            "board_default_pos": (0.2837, 0.229, 0.0),
            "board_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005)},
            "parts": [
                {"scene_name": "sc_port", "offset": (0.0067, -0.0362, 0.005), "pose_range": {"x": (-0.005, 0.02)}},
                {"scene_name": "sc_port_2", "offset": (0.0076, -0.0783, 0.005), "pose_range": {"x": (-0.005, 0.02)}},
                {"scene_name": "nic_card", "offset": (-0.03235, 0.02329, 0.0743), "pose_range": {"y": (0.0, 0.12)}, "snap_step": {"y": 0.04}},
            ],
            "sync_usd_xforms": False,
        },
    )
    reset_to_preinsertion = EventTerm(
        func=mdp.reset_to_preinsertion_pose,
        mode="reset",
        params={
            "target_key": "${TARGET_KEY}",
            "curriculum_steps": 250000,
            "joint_noise_start": 0.004,
            "joint_noise_end": 0.001,
            "xy_error_start": 0.004,
            "xy_error_end": 0.001,
            "z_above_start": 0.010,
            "z_above_end": 0.002,
            "roll_pitch_start": 0.08,
            "roll_pitch_end": 0.025,
            "yaw_start": 0.14,
            "yaw_end": 0.04,
        },
    )


@configclass
class AICInsertHybridEnvCfg(AICTaskEnvCfg):
    observations: HybridInsertObservationsCfg = HybridInsertObservationsCfg()
    rewards: HybridInsertRewardsCfg = HybridInsertRewardsCfg()
    terminations: HybridInsertTerminationsCfg = HybridInsertTerminationsCfg()
    events: HybridInsertEventCfg = HybridInsertEventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 120.0
        self.scene.center_camera = None
        self.scene.left_camera = None
        self.scene.right_camera = None
        # Zoomed-out enough to keep full arm and board visible.
        self.viewer.eye = (1.15, -0.15, 0.70)
        self.viewer.lookat = (0.28, 0.18, 0.08)
        target_key = getattr(self, "_target_key", "sc")

        # Keep translational action small and rotational action dominant.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=_PLUG_TIP_IN_WRIST[target_key][0],
                rot=_PLUG_TIP_IN_WRIST[target_key][1],
            ),
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="svd",
                ik_params={"k_val": 1.0, "min_singular_value": 1e-5},
            ),
            # Conservative contact-scale motions for terminal seating.
            scale=(0.008, 0.008, 0.018, 0.10, 0.10, 0.18),
        )

        self.commands.ee_pose.body_name = "wrist_3_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        self._resolve_target_key(target_key)

    def _resolve_target_key(self, target_key: str):
        for group in [
            self.observations.policy,
            self.rewards,
            self.terminations,
            self.events,
        ]:
            for name in dir(group):
                term = getattr(group, name, None)
                params = getattr(term, "params", None)
                if isinstance(params, dict) and params.get("target_key") == "${TARGET_KEY}":
                    params["target_key"] = target_key


@configclass
class AICInsertHybridSCEnvCfg(AICInsertHybridEnvCfg):
    def __post_init__(self):
        self._target_key = "sc"
        super().__post_init__()


@configclass
class AICInsertHybridSC2EnvCfg(AICInsertHybridEnvCfg):
    def __post_init__(self):
        self._target_key = "sc2"
        super().__post_init__()


@configclass
class AICInsertHybridSFPEnvCfg(AICInsertHybridEnvCfg):
    def __post_init__(self):
        self._target_key = "sfp"
        super().__post_init__()


@configclass
class AICInsertHybridSCDebugEnvCfg(AICInsertHybridSCEnvCfg):
    """Easy SC insertion curriculum for verifying depth/crossing behavior."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 8.0

        # Stop over-rewarding hover at the entrance plane.
        self.rewards.approach_z_fine.weight = 4.0
        self.rewards.approach_z_dense.weight = 2.0
        self.rewards.approach_descend.weight = 2.0
        self.rewards.approach_z_progress.weight = 15.0

        # Make below-plane motion visible immediately, before perfect seating.
        self.rewards.crossed_plane = RewTerm(
            func=mdp.crossed_port_plane,
            weight=80.0,
            params={"target_key": "sc"},
        )
        self.rewards.depth_raw = RewTerm(
            func=mdp.insertion_depth_raw,
            weight=650.0,
            params={"target_key": "sc"},
        )
        self.rewards.depth_progress = RewTerm(
            func=mdp.insertion_depth_progress,
            weight=1200.0,
            params={"target_key": "sc", "clip": 0.08},
        )
        self.rewards.diag_depth_gate = RewTerm(
            func=mdp.insertion_depth_gate,
            weight=1.0,
            params={"target_key": "sc", "depth_fraction": 0.35},
        )
        self.rewards.diag_xy_gate = RewTerm(
            func=mdp.insertion_xy_gate,
            weight=1.0,
            params={"target_key": "sc", "xy_thresh": 0.015},
        )
        self.rewards.diag_axis_gate = RewTerm(
            func=mdp.insertion_axis_gate,
            weight=1.0,
            params={"target_key": "sc", "axis_thresh": 0.45},
        )
        self.rewards.diag_twist_gate = RewTerm(
            func=mdp.insertion_twist_gate,
            weight=1.0,
            params={"target_key": "sc", "twist_thresh": 0.45},
        )
        self.rewards.diag_depth_xy_gate = RewTerm(
            func=mdp.insertion_depth_xy_gate,
            weight=1.0,
            params={"target_key": "sc", "depth_fraction": 0.35, "xy_thresh": 0.015},
        )

        self.rewards.insertion_depth.weight = 240.0
        self.rewards.insertion_depth.params.update(
            {"xy_gate": 0.020, "axis_gate": 0.20, "twist_gate": 0.20}
        )
        self.rewards.success_bonus.weight = 120.0
        self.rewards.success_bonus.params.update(
            {"xy_thresh": 0.010, "axis_thresh": 0.45, "twist_thresh": 0.45, "depth_fraction": 0.50}
        )
        self.terminations.insertion_success.params.update(
            {"xy_thresh": 0.010, "axis_thresh": 0.45, "twist_thresh": 0.45, "depth_fraction": 0.50}
        )

        # Start within a few millimeters of the entrance, with some episodes
        # already slightly below the plane. This verifies insertion geometry
        # before we restore a harder handoff distribution.
        self.events.reset_to_preinsertion.params.update(
            {
                "curriculum_steps": 120000,
                "joint_noise_start": 0.002,
                "joint_noise_end": 0.0005,
                "xy_error_start": 0.002,
                "xy_error_end": 0.0005,
                "z_above_start": 0.004,
                "z_above_end": 0.001,
                "z_below_start": 0.004,
                "z_below_end": 0.010,
                "roll_pitch_start": 0.035,
                "roll_pitch_end": 0.012,
                "yaw_start": 0.05,
                "yaw_end": 0.015,
            }
        )

        self.actions.arm_action.scale = (0.006, 0.006, 0.012, 0.08, 0.08, 0.12)


@configclass
class AICInsertHybridSCDepthDebugEnvCfg(AICInsertHybridSCDebugEnvCfg):
    """Bootstrap task: learn to stay inserted before enforcing twist seating."""

    def __post_init__(self):
        super().__post_init__()

        self.events.randomize_board_and_parts.params.update(
            {
                "board_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                "parts": [
                    {"scene_name": "sc_port", "offset": (0.0067, -0.0362, 0.005), "pose_range": {"x": (0.0, 0.0)}},
                    {"scene_name": "sc_port_2", "offset": (0.0076, -0.0783, 0.005), "pose_range": {"x": (0.0, 0.0)}},
                    {"scene_name": "nic_card", "offset": (-0.03235, 0.02329, 0.0743), "pose_range": {"y": (0.0, 0.0)}, "snap_step": {"y": 0.04}},
                ],
            }
        )
        self.rewards.crossed_plane.weight = 10.0
        self.rewards.depth_raw.weight = 20.0
        self.rewards.depth_progress.weight = 60.0
        self.rewards.scripted_action_imitation = RewTerm(
            func=mdp.scripted_action_imitation_reward,
            weight=0.0,
            params={"target_key": "sc", "sigma": 0.45},
        )
        self.rewards.scripted_action_mse_penalty = RewTerm(
            func=mdp.scripted_action_error,
            weight=0.0,
            params={"target_key": "sc"},
        )
        self.rewards.scripted_action_error = RewTerm(
            func=mdp.scripted_action_error,
            weight=1.0,
            params={"target_key": "sc"},
        )
        self.rewards.diag_xy_l2 = RewTerm(
            func=mdp.tip_to_port_xy_l2,
            weight=1.0,
            params={"target_key": "sc"},
        )
        self.rewards.diag_z = RewTerm(
            func=mdp.tip_to_port_z,
            weight=1.0,
            params={"target_key": "sc"},
        )
        self.rewards.approach_xy_coarse.weight = -800.0
        self.rewards.approach_xy_progress.weight = 650.0
        self.rewards.approach_xy_fine.weight = 450.0
        self.rewards.approach_xy_fine.params.update({"sigma": 0.020})
        self.rewards.adaptive_xy_depth = RewTerm(
            func=mdp.adaptive_centered_depth_curriculum,
            weight=2600.0,
            params={
                "target_key": "sc",
                "sigma_xy": 0.015,
                "xy_thresh": 0.015,
                "xy_ready": 0.35,
                "ema_alpha": 0.02,
            },
        )
        self.rewards.adaptive_xy_gate_ema = RewTerm(
            func=mdp.adaptive_xy_gate_ema,
            weight=1.0,
        )
        self.rewards.adaptive_depth_phase = RewTerm(
            func=mdp.adaptive_depth_phase,
            weight=1.0,
        )
        self.rewards.centered_depth = RewTerm(
            func=mdp.centered_insertion_depth_reward,
            weight=900.0,
            params={"target_key": "sc", "sigma_xy": 0.015},
        )
        self.rewards.inserted_xy_alignment = RewTerm(
            func=mdp.inserted_xy_alignment,
            weight=1200.0,
            params={"target_key": "sc", "sigma_xy": 0.015, "min_depth_fraction": 0.05},
        )
        self.rewards.offcenter_depth = RewTerm(
            func=mdp.offcenter_depth_penalty,
            weight=-6000.0,
            params={"target_key": "sc", "free_xy": 0.004},
        )
        self.rewards.insertion_depth.weight = 120.0
        self.rewards.insertion_depth.params.update(
            {"xy_gate": 0.030, "axis_gate": 0.0, "twist_gate": 0.0}
        )
        self.rewards.success_bonus.weight = 300.0
        self.rewards.success_bonus.params.update(
            {"xy_thresh": 0.015, "axis_thresh": 0.0, "twist_thresh": 0.0, "depth_fraction": 0.35}
        )
        self.terminations.insertion_success.params.update(
            {"xy_thresh": 0.015, "axis_thresh": 0.0, "twist_thresh": 0.0, "depth_fraction": 0.35}
        )
        self.terminations.drift_failure.params.update({"xy_thresh": 1.20})
        self.events.reset_to_preinsertion.params.update(
            {
                "xy_error_start": 0.001,
                "xy_error_end": 0.00025,
                "ik_gain": 0.90,
                "ik_delta_limit": 0.10,
                "ik_iters": 48,
            }
        )
        self.actions.arm_action.scale = (0.0015, 0.0015, 0.014, 0.05, 0.05, 0.08)


@configclass
class AICInsertHybridSCStrictDebugEnvCfg(AICInsertHybridSCDepthDebugEnvCfg):
    """Intermediate task: keep DepthDebug reset/action scale but require rotation-correct seating."""

    def __post_init__(self):
        super().__post_init__()
        self.use_kinematic_tip = True
        self.rewards.force_penalty.weight = 0.0
        self.terminations.force_failure.params.update({"force_thresh": 1.0e9, "duration_steps": 1000000})

        self.rewards.insertion_depth.weight = 180.0
        self.rewards.insertion_depth.params.update(
            {"xy_gate": 0.020, "axis_gate": 0.20, "twist_gate": 0.20}
        )
        self.rewards.success_bonus.weight = 300.0
        self.rewards.success_bonus.params.update(
            {"xy_thresh": 0.010, "axis_thresh": 0.45, "twist_thresh": 0.45, "depth_fraction": 0.50}
        )
        self.terminations.insertion_success.params.update(
            {"xy_thresh": 0.010, "axis_thresh": 0.45, "twist_thresh": 0.45, "depth_fraction": 0.50}
        )
        self.events.reset_to_preinsertion.params.update(
            {
                "xy_error_start": 0.002,
                "xy_error_end": 0.0005,
                "z_above_start": 0.006,
                "z_above_end": 0.002,
                "z_below_start": 0.0,
                "z_below_end": 0.0,
                "roll_pitch_start": 0.08,
                "roll_pitch_end": 0.02,
                "yaw_start": 0.14,
                "yaw_end": 0.03,
            }
        )
        self.actions.arm_action.scale = (0.0015, 0.0015, 0.010, 0.08, 0.08, 0.12)


@configclass
class AICInsertHybridSCRotDebugEnvCfg(AICInsertHybridSCDepthDebugEnvCfg):
    """Rotation bootstrap task between relaxed depth insertion and strict seating."""

    observations: HybridInsertRotObservationsCfg = HybridInsertRotObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.use_kinematic_tip = True
        self.rewards.force_penalty.weight = 0.0
        self.terminations.force_failure.params.update({"force_thresh": 1.0e9, "duration_steps": 1000000})

        # Stop paying heavily for ungated shove-depth while the policy is still
        # discovering the seated orientation.
        self.rewards.depth_raw.weight = 5.0
        self.rewards.depth_progress.weight = 15.0
        self.rewards.crossed_plane.weight = 2.0

        self.rewards.axis_alignment.weight = 220.0
        self.rewards.twist_alignment.weight = 360.0
        self.rewards.twist_progress.weight = 500.0
        self.rewards.rotation_alignment = RewTerm(
            func=mdp.tip_rotation_exp,
            weight=700.0,
            params={"target_key": "sc", "sigma": 0.55},
        )
        self.rewards.rotation_progress = RewTerm(
            func=mdp.tip_rotation_progress,
            weight=300.0,
            params={"target_key": "sc", "clip": 0.10},
        )

        self.rewards.scripted_action_imitation.weight = 250.0
        self.rewards.scripted_action_imitation.params.update({"sigma": 0.35})
        self.rewards.scripted_action_mse_penalty.weight = -20.0

        # Start from the intended last-mile handoff: within a couple of
        # millimeters XY and just above the entrance. The kinematic TCP state
        # makes this stable without depending on whole cable settling.
        self.events.reset_to_preinsertion.params.update(
            {
                "xy_error_start": 0.002,
                "xy_error_end": 0.0005,
                "z_above_start": 0.006,
                "z_above_end": 0.002,
                "z_below_start": 0.0,
                "z_below_end": 0.0,
                "roll_pitch_start": 0.08,
                "roll_pitch_end": 0.02,
                "yaw_start": 0.14,
                "yaw_end": 0.03,
            }
        )

        self.rewards.insertion_depth.weight = 100.0
        self.rewards.insertion_depth.params.update(
            {"xy_gate": 0.020, "axis_gate": 0.05, "twist_gate": 0.05}
        )
        self.rewards.success_bonus.weight = 240.0
        self.rewards.success_bonus.params.update(
            {"xy_thresh": 0.012, "axis_thresh": 0.30, "twist_thresh": 0.30, "depth_fraction": 0.40}
        )
        self.terminations.insertion_success.params.update(
            {"xy_thresh": 0.012, "axis_thresh": 0.30, "twist_thresh": 0.30, "depth_fraction": 0.40}
        )

        self.actions.arm_action.scale = (0.0015, 0.0015, 0.010, 0.08, 0.08, 0.12)


@configclass
class AICInsertHybridSCStrictRotDebugEnvCfg(AICInsertHybridSCRotDebugEnvCfg):
    """Strict rotation-observation follow-up with final nonzero-success gates."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.scripted_action_imitation.weight = 0.0
        self.rewards.scripted_action_mse_penalty.weight = 0.0
        self.rewards.insertion_depth.weight = 180.0
        self.rewards.insertion_depth.params.update(
            {"xy_gate": 0.020, "axis_gate": 0.20, "twist_gate": 0.20}
        )
        self.rewards.success_bonus.weight = 300.0
        self.rewards.success_bonus.params.update(
            {"xy_thresh": 0.010, "axis_thresh": 0.45, "twist_thresh": 0.45, "depth_fraction": 0.50}
        )
        self.terminations.insertion_success.params.update(
            {"xy_thresh": 0.010, "axis_thresh": 0.45, "twist_thresh": 0.45, "depth_fraction": 0.50}
        )
