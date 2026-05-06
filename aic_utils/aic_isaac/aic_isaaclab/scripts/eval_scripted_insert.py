"""Evaluate the scripted final-insertion controller without PPO.

This is a geometry/sign sanity check for the hybrid insertion task. If this
controller cannot keep XY centered and insert, reward tuning will not make PPO
solve the same behavior quickly.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate scripted AIC insertion controller.")
parser.add_argument("--task", type=str, default="AIC-Insert-Hybrid-SC-DepthDebug-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=240)
parser.add_argument("--zero_action", action="store_true", help="Apply zero actions after reset for stability diagnostics.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.tasks.manager_based.aic_task.kinematic_last_mile_env import KinematicLastMileRLEnv
from aic_task.tasks.manager_based.aic_task import mdp
from aic_task.tasks.manager_based.aic_task.mdp.mdp_insert import (
    _TARGETS,
    _tip_in_port_frame,
    plug_port_axis_alignment,
    plug_port_twist_alignment,
)

STRICT_XY_THRESH = 0.010
STRICT_AXIS_THRESH = 0.45
STRICT_TWIST_THRESH = 0.45
STRICT_DEPTH_FRACTION = 0.50
PHASE_LABELS = ("align", "center", "descend", "settle")
FAILURE_LABELS = ("none", "lost_xy", "lost_rotation", "overshoot")


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    use_kinematic_tip = getattr(env_cfg, "use_kinematic_tip", False)
    env_cls = KinematicLastMileRLEnv if use_kinematic_tip else ManagerBasedRLEnv
    env: ManagerBasedRLEnv = env_cls(cfg=env_cfg)
    target_key = getattr(env_cfg, "_target_key", "sc")
    full_depth = _TARGETS[target_key]["depth"]

    env.reset()
    ever_success = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    first_xy = None
    first_z = None
    min_xy = torch.full((args.num_envs,), 1.0e9, device=env.device)
    max_depth = torch.zeros(args.num_envs, device=env.device)
    max_axis = torch.zeros(args.num_envs, device=env.device)
    max_twist = torch.zeros(args.num_envs, device=env.device)
    phase_counts = torch.zeros(len(PHASE_LABELS), dtype=torch.long, device=env.device)
    first_failure = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)

    print(f"[INFO] Evaluating scripted controller: task={args.task} envs={args.num_envs} steps={args.max_steps}")

    for step in range(args.max_steps):
        with torch.no_grad():
            delta_before, *_ = _tip_in_port_frame(env, target_key)
            _, _, _, tip_pos_before, _, port_pos_before, _ = _tip_in_port_frame(env, target_key)
            joint_pos_before = env.scene["robot"].data.joint_pos.clone()
            xy_before = torch.norm(delta_before[:, :2], dim=-1)
            z_before = delta_before[:, 2]
            depth_before = torch.clamp(-z_before, min=0.0)
            axis_before = plug_port_axis_alignment(env, target_key)
            twist_before = plug_port_twist_alignment(env, target_key)
            success_before = (
                (depth_before > STRICT_DEPTH_FRACTION * full_depth)
                & (xy_before < STRICT_XY_THRESH)
                & (axis_before > STRICT_AXIS_THRESH)
                & (twist_before > STRICT_TWIST_THRESH)
            )
            if args.zero_action:
                action = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)
                phase_cmd = torch.zeros(args.num_envs, device=env.device, dtype=torch.long)
            else:
                action = mdp.scripted_insert_action_hint(env, target_key)
                phase_cmd = getattr(
                    env,
                    "_scripted_insert_phase",
                    torch.zeros(args.num_envs, device=env.device, dtype=torch.long),
                )
            phase_counts += torch.bincount(
                torch.clamp(phase_cmd, min=0, max=len(PHASE_LABELS) - 1),
                minlength=len(PHASE_LABELS),
            )[: len(PHASE_LABELS)]
            _, _, terminated, timed_out, _ = env.step(action)
            delta_port, _, _, tip_pos_after, _, port_pos_after, _ = _tip_in_port_frame(env, target_key)
            joint_pos_after = env.scene["robot"].data.joint_pos
            tip_move = torch.norm(tip_pos_after - tip_pos_before, dim=-1)
            port_move = torch.norm(port_pos_after - port_pos_before, dim=-1)
            joint_move = torch.norm(joint_pos_after - joint_pos_before, dim=-1)
            xy = torch.norm(delta_port[:, :2], dim=-1)
            z = delta_port[:, 2]
            depth = torch.clamp(-z, min=0.0)
            axis = plug_port_axis_alignment(env, target_key)
            twist = plug_port_twist_alignment(env, target_key)
            strict_gate = (
                (depth > STRICT_DEPTH_FRACTION * full_depth)
                & (xy < STRICT_XY_THRESH)
                & (axis > STRICT_AXIS_THRESH)
                & (twist > STRICT_TWIST_THRESH)
            )
            # Kinematic last-mile envs reset successful episodes inside step,
            # so post-step poses can be fresh reset poses rather than terminal
            # poses. Force/drift terminations are disabled there, making
            # terminated equivalent to insertion success for this diagnostic.
            success = success_before | strict_gate | (terminated if use_kinematic_tip else torch.zeros_like(terminated))
            failure_open = first_failure == 0
            active_insert = phase_cmd >= 2
            first_failure[:] = torch.where(
                failure_open & (depth > 0.030),
                torch.full_like(first_failure, 3),
                first_failure,
            )
            first_failure[:] = torch.where(
                (first_failure == 0) & active_insert & (xy > 1.2 * STRICT_XY_THRESH),
                torch.full_like(first_failure, 1),
                first_failure,
            )
            first_failure[:] = torch.where(
                (first_failure == 0)
                & active_insert
                & ((axis < STRICT_AXIS_THRESH) | (twist < STRICT_TWIST_THRESH)),
                torch.full_like(first_failure, 2),
                first_failure,
            )

        if first_xy is None:
            first_xy = xy.detach().clone()
            first_z = z.detach().clone()
        min_xy = torch.minimum(min_xy, xy)
        max_depth = torch.maximum(max_depth, depth)
        max_axis = torch.maximum(max_axis, axis)
        max_twist = torch.maximum(max_twist, twist)
        ever_success |= success | strict_gate

        if step % 20 == 0 or step == args.max_steps - 1:
            print(
                "[STEP {step:03d}] "
                "success={success_rate:.3f} term={term_count} timeout={timeout_count} "
                "pre_xy_mm={pre_xy:.2f} pre_z_mm={pre_z:.2f} pre_depth_mm={pre_depth:.2f} "
                "xy_mean_mm={xy_mean:.2f} xy_min_mean_mm={xy_min:.2f} "
                "z_mean_mm={z_mean:.2f} depth_mean_mm={depth_mean:.2f} max_depth_mean_mm={max_depth_mean:.2f} "
                "axis_mean={axis_mean:.3f} twist_mean={twist_mean:.3f} "
                "max_axis_mean={max_axis_mean:.3f} max_twist_mean={max_twist_mean:.3f} "
                "act_xyz_mean={act_xyz} act_rot_mean={act_rot} "
                "tip_move_mm={tip_move:.2f} port_move_mm={port_move:.2f} joint_move={joint_move:.4f} "
                "phase={phase_summary}"
                .format(
                    step=step,
                    success_rate=ever_success.float().mean().item(),
                    term_count=int(terminated.sum().item()),
                    timeout_count=int(timed_out.sum().item()),
                    pre_xy=xy_before.mean().item() * 1000.0,
                    pre_z=z_before.mean().item() * 1000.0,
                    pre_depth=depth_before.mean().item() * 1000.0,
                    xy_mean=xy.mean().item() * 1000.0,
                    xy_min=min_xy.mean().item() * 1000.0,
                    z_mean=z.mean().item() * 1000.0,
                    depth_mean=depth.mean().item() * 1000.0,
                    max_depth_mean=max_depth.mean().item() * 1000.0,
                    axis_mean=axis.mean().item(),
                    twist_mean=twist.mean().item(),
                    max_axis_mean=max_axis.mean().item(),
                    max_twist_mean=max_twist.mean().item(),
                    act_xyz="/".join(f"{v:.4f}" for v in action[:, :3].mean(dim=0).tolist()),
                    act_rot="/".join(f"{v:.4f}" for v in action[:, 3:].mean(dim=0).tolist()),
                    tip_move=tip_move.mean().item() * 1000.0,
                    port_move=port_move.mean().item() * 1000.0,
                    joint_move=joint_move.mean().item(),
                    phase_summary="/".join(
                        f"{name}:{int((phase_cmd == idx).sum().item())}"
                        for idx, name in enumerate(PHASE_LABELS)
                    ),
                )
            )

    final_delta, *_ = _tip_in_port_frame(env, target_key)
    final_xy = torch.norm(final_delta[:, :2], dim=-1)
    final_depth = torch.clamp(-final_delta[:, 2], min=0.0)
    final_axis = plug_port_axis_alignment(env, target_key)
    final_twist = plug_port_twist_alignment(env, target_key)
    print("[SUMMARY]")
    print(f"initial_xy_mean_mm={first_xy.mean().item() * 1000.0:.3f}")
    print(f"initial_z_mean_mm={first_z.mean().item() * 1000.0:.3f}")
    print(f"final_xy_mean_mm={final_xy.mean().item() * 1000.0:.3f}")
    print(f"final_depth_mean_mm={final_depth.mean().item() * 1000.0:.3f}")
    print(f"final_axis_mean={final_axis.mean().item():.3f}")
    print(f"final_twist_mean={final_twist.mean().item():.3f}")
    print(f"min_xy_mean_mm={min_xy.mean().item() * 1000.0:.3f}")
    print(f"max_depth_mean_mm={max_depth.mean().item() * 1000.0:.3f}")
    print(f"max_axis_mean={max_axis.mean().item():.3f}")
    print(f"max_twist_mean={max_twist.mean().item():.3f}")
    print(f"ever_success_rate={ever_success.float().mean().item():.3f}")
    total_phase = torch.clamp(phase_counts.sum(), min=1)
    phase_summary = ", ".join(
        f"{name}={phase_counts[idx].item() / total_phase.item():.3f}"
        for idx, name in enumerate(PHASE_LABELS)
    )
    print(f"phase_fraction={phase_summary}")
    failure_summary = ", ".join(
        f"{name}={(first_failure == idx).float().mean().item():.3f}"
        for idx, name in enumerate(FAILURE_LABELS)
    )
    print(f"first_failure_fraction={failure_summary}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
