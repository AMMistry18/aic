"""Print reset geometry for AIC hybrid insertion.

This checks whether the reset target, actual tip pose, and port entrance pose
agree immediately after reset. Use this before RL training when success is low.
"""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug AIC insertion reset geometry.")
parser.add_argument("--task", type=str, default="AIC-Insert-Hybrid-SC-DepthDebug-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.tasks.manager_based.aic_task.mdp.mdp_insert import (
    _TIP_IN_WRIST,
    _TARGETS,
    _plug_tip_pose_w,
    _port_pose_w,
    _tip_in_port_frame,
)
from isaaclab.utils.math import combine_frame_transforms


def _fmt_vec(name: str, value: torch.Tensor, scale: float = 1.0) -> str:
    data = (value.detach().cpu().numpy() * scale).tolist()
    return f"{name}: [{', '.join(f'{x:.6f}' for x in data)}]"


def main() -> None:
    env = None
    try:
        print("[DEBUG] parsing env cfg", flush=True)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        print("[DEBUG] creating env", flush=True)
        env = ManagerBasedRLEnv(cfg=env_cfg)
        target_key = getattr(env_cfg, "_target_key", "sc")

        print("[DEBUG] calling env.reset()", flush=True)
        env.reset()
        print("[DEBUG] env.reset() returned", flush=True)

        print("[DEBUG] updating scene after reset", flush=True)
        env.scene.update(env.physics_dt)
        print("[DEBUG] scene update returned", flush=True)

        for i in range(args.settle_steps):
            print(f"[DEBUG] settle step {i + 1}/{args.settle_steps}", flush=True)
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

        print("[DEBUG] reading plug tip pose", flush=True)
        tip_pos_w, tip_q_w = _plug_tip_pose_w(env, target_key)
        print("[DEBUG] plug tip pose returned", flush=True)

        robot = env.scene["robot"]
        wrist_idx = robot.data.body_names.index("wrist_3_link")
        wrist_pos_w = robot.data.body_pos_w[:, wrist_idx, :]
        wrist_q_w = robot.data.body_quat_w[:, wrist_idx, :]
        tip_cfg = _TIP_IN_WRIST[_TARGETS[target_key]["port_type"]]
        offset = torch.tensor(tip_cfg["pos"], device=env.device, dtype=wrist_pos_w.dtype).expand(env.num_envs, -1)
        offset_q = torch.tensor(tip_cfg["quat_wxyz"], device=env.device, dtype=wrist_pos_w.dtype).expand(env.num_envs, -1)
        controller_tip_pos_w, controller_tip_q_w = combine_frame_transforms(
            wrist_pos_w, wrist_q_w, offset, offset_q
        )

        print("[DEBUG] reading port pose", flush=True)
        port_pos_w, port_q_w = _port_pose_w(env, target_key)
        print("[DEBUG] port pose returned", flush=True)

        print("[DEBUG] computing tip in port frame", flush=True)
        delta_port, rel_q, rot_err_port, *_ = _tip_in_port_frame(env, target_key)
        print("[DEBUG] tip in port frame returned", flush=True)

        desired_tip_local = getattr(env, "_last_reset_desired_tip_local", None)
        desired_tip_pos_w = getattr(env, "_last_reset_desired_tip_pos_w", None)
        desired_wrist_pos_w = getattr(env, "_last_reset_desired_wrist_pos_w", None)
        ik_initial_pos_err = getattr(env, "_last_reset_ik_initial_pos_err", None)
        ik_final_pos_err = getattr(env, "_last_reset_ik_final_pos_err", None)
        ik_joint_ids = getattr(env, "_last_reset_ik_joint_ids", None)

        print(f"[INFO] task={args.task} target_key={target_key} num_envs={args.num_envs}", flush=True)
        print("[ENV 0]", flush=True)
        print(_fmt_vec("port_pos_w_m", port_pos_w[0]), flush=True)
        print(_fmt_vec("port_quat_wxyz", port_q_w[0]), flush=True)
        print(_fmt_vec("wrist_pos_w_m", wrist_pos_w[0]), flush=True)
        print(_fmt_vec("wrist_quat_wxyz", wrist_q_w[0]), flush=True)
        print(_fmt_vec("tip_pos_w_m", tip_pos_w[0]), flush=True)
        print(_fmt_vec("tip_quat_wxyz", tip_q_w[0]), flush=True)
        print(_fmt_vec("controller_tip_pos_w_m", controller_tip_pos_w[0]), flush=True)
        print(_fmt_vec("controller_tip_quat_wxyz", controller_tip_q_w[0]), flush=True)
        print(_fmt_vec("delta_port_m", delta_port[0]), flush=True)
        print(_fmt_vec("delta_port_mm", delta_port[0], scale=1000.0), flush=True)
        print(_fmt_vec("rot_err_port", rot_err_port[0]), flush=True)
        print(_fmt_vec("rel_quat_wxyz", rel_q[0]), flush=True)

        if desired_tip_local is not None:
            print(_fmt_vec("desired_tip_local_m", desired_tip_local[0]), flush=True)
            print(_fmt_vec("desired_tip_local_mm", desired_tip_local[0], scale=1000.0), flush=True)
        if desired_tip_pos_w is not None:
            print(_fmt_vec("desired_tip_pos_w_m", desired_tip_pos_w[0]), flush=True)
            print(_fmt_vec("tip_minus_desired_tip_mm", tip_pos_w[0] - desired_tip_pos_w[0], scale=1000.0), flush=True)
            print(
                _fmt_vec(
                    "controller_tip_minus_desired_tip_mm",
                    controller_tip_pos_w[0] - desired_tip_pos_w[0],
                    scale=1000.0,
                ),
                flush=True,
            )
        if desired_wrist_pos_w is not None:
            print(_fmt_vec("desired_wrist_pos_w_m", desired_wrist_pos_w[0]), flush=True)
            print(_fmt_vec("wrist_minus_desired_wrist_mm", wrist_pos_w[0] - desired_wrist_pos_w[0], scale=1000.0), flush=True)
        if ik_joint_ids is not None:
            print(f"reset_ik_joint_ids: {ik_joint_ids}", flush=True)
        if ik_initial_pos_err is not None:
            print(_fmt_vec("reset_ik_initial_pos_err_mm", ik_initial_pos_err[0], scale=1000.0), flush=True)
        if ik_final_pos_err is not None:
            print(_fmt_vec("reset_ik_final_pos_err_mm", ik_final_pos_err[0], scale=1000.0), flush=True)

        xy = torch.norm(delta_port[:, :2], dim=-1)
        print("[STATS]", flush=True)
        print(f"xy_mean_mm={xy.mean().item() * 1000.0:.3f}", flush=True)
        print(f"xy_min_mm={xy.min().item() * 1000.0:.3f}", flush=True)
        print(f"xy_max_mm={xy.max().item() * 1000.0:.3f}", flush=True)
        print(f"z_mean_mm={delta_port[:, 2].mean().item() * 1000.0:.3f}", flush=True)
        print(f"z_min_mm={delta_port[:, 2].min().item() * 1000.0:.3f}", flush=True)
        print(f"z_max_mm={delta_port[:, 2].max().item() * 1000.0:.3f}", flush=True)
    except BaseException:
        print("[EXCEPTION] debug_reset_geometry failed", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            print("[DEBUG] closing env", flush=True)
            env.close()
            print("[DEBUG] env closed", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
