"""
Recording script for the last-inch residual-SAC trainer.

Runs against the live AIC engine. Subscribes to `Observation` and
`/scoring/insertion_event`, runs the existing `PerceptionInsert` policy,
and writes one .npz file per episode into `--out-dir`.

Usage:

    # Terminal A — start the AIC sim (headless)
    /home/Anshul/.local/bin/distrobox enter aic_eval-latest -- bash -c \
      '/entrypoint.sh ground_truth:=false start_aic_engine:=false gazebo_gui:=false launch_rviz:=false'

    # Terminal B — start recording
    cd /home/Anshul/AIC_Phase_1/aic_0/aic
    pixi run python -m RL.record_aic \
        --port-type sc \
        --out-dir outputs/recordings \
        --n-episodes 200 \
        --image-size 32

The recorded .npz files are what `RL/train.py --recorded --dataset-dir ...`
consumes for offline training.

Caveats
-------
* port_xyz is hardcoded to (0, 0, 0). For accurate tcp_pose_err in
  the port frame, the script should look up
  `<target_module>/<port>_link_entrance` via TF. Easy to add.
* action_6d is computed as a *delta* between consecutive MotionUpdate
  target poses. The residual SAC trainer's pos_scale/rot_scale further
  normalises the delta into the action space [-1, 1].
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL.recording import EpisodeRecorder


def _record_action_delta(msg, pending: dict) -> None:
    """Compute the 6-vector action delta from a MotionUpdate message.

    MotionUpdate has a *target* pose. The residual SAC trainer expects
    the *delta* (the change the policy commanded at this step). We
    subtract the previous target pose.
    """
    p = msg.pose.position
    o = msg.pose.orientation
    new_pose = np.array([p.x, p.y, p.z, o.x, o.y, o.z], dtype=np.float32)
    if pending["prev"] is None:
        pending["delta"] = np.zeros(6, dtype=np.float32)
    else:
        pending["delta"] = new_pose - pending["prev"]
    pending["prev"] = new_pose


def _build_obs(obs_msg, target_hw):
    """Build the (image, force, pose...) tuple from a live Observation msg."""
    from RL.observation import build_obs_dict_from_ros
    # Port pose: hardcoded at world origin. For accurate port-frame
    # tcp_pose_err, add a TF lookup here. Leaving it as zeros is fine
    # for a first-pass dataset — the trainer still learns the image
    # signal which is the dominant reward.
    port_xyz = np.zeros(3, dtype=np.float32)
    port_q = np.array([1.0, 0, 0, 0], dtype=np.float32)
    try:
        d = build_obs_dict_from_ros(
            obs_msg, port_xyz, port_q, np.zeros(6, dtype=np.float32))
    except Exception:
        return None
    return {
        "image": d["image"],
        "force": d["force"],
        "tcp_xyz": d["tcp_pose"][:3],
        "tcp_q": d["tcp_pose"][3:7],
        "port_xyz": port_xyz,
        "port_q": port_q,
        "tcp_pose_err": d["tcp_pose_err"],
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Record AIC last-inch rollouts for offline SAC training")
    p.add_argument("--port-type", choices=["sc", "sfp"], required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/recordings"))
    p.add_argument("--n-episodes", type=int, default=100,
                   help="number of episodes to record (0 = run until killed)")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--policy", default="PerceptionInsert")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import rclpy
    from rclpy.node import Node
    from aic_model_interfaces.msg import Observation
    from aic_task_interfaces.action import InsertCable
    from aic_task_interfaces.msg import Task
    from std_msgs.msg import String
    from aic_control_interfaces.msg import MotionUpdate

    rclpy.init()
    node = rclpy.create_node("aic_recorder")

    insert_client = rclpy.action.ActionClient(
        node, InsertCable, "insert_cable")
    if not insert_client.wait_for_action_server(timeout_sec=10.0):
        node.get_logger().error("/insert_cable action server not available")
        return 1

    latest_obs = {"msg": None}
    latest_event = {"flag": False}

    node.create_subscription(Observation, "observations",
                            lambda msg: latest_obs.update(msg=msg), 10)
    node.create_subscription(String, "/scoring/insertion_event",
                            lambda msg: latest_event.update(flag=bool(msg.data)), 10)

    # Try to import the base policy (we don't drive it; aic_model is
    # already running it via /insert_cable). We just log which one is
    # being used.
    try:
        import importlib
        mod = importlib.import_module(f"aic_example_policies.ros.{args.policy}")
        getattr(mod, args.policy)  # sanity check the class exists
        node.get_logger().info(f"Detected base policy: {args.policy}")
    except Exception as exc:
        node.get_logger().warn(
            f"Could not import {args.policy} (will still record): {exc}")

    target_hw = (args.image_size, args.image_size)
    n_recorded = 0
    last_event_value = False

    while rclpy.ok() and (args.n_episodes == 0 or n_recorded < args.n_episodes):
        latest_event["flag"] = False
        last_event_value = False

        rec = EpisodeRecorder(
            port_type=args.port_type,
            image_h=args.image_size, image_w=args.image_size,
            n_cams=3, image_ch_per_cam=3,
        )
        pending = {"prev": None, "delta": np.zeros(6, dtype=np.float32)}

        def _action_cb(msg, pa=pending):
            _record_action_delta(msg, pa)

        action_sub = node.create_subscription(
            MotionUpdate, "/aic_controller/pose_commands", _action_cb, 10)

        target_module = (f"nic_card_mount_0" if args.port_type == "sfp"
                         else "sc_port_0")
        port_name = (f"sfp_port_0" if args.port_type == "sfp"
                     else f"sc_port_0")
        task = Task(target_module_name=target_module, port_name=port_name,
                    port_type=args.port_type)
        goal_msg = InsertCable.Goal()
        goal_msg.task = task
        send_future = insert_client.send_goal_async(goal_msg)

        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            rclpy.spin_once_until_future_complete(
                node, send_future, timeout_sec=5.0)
            goal_handle = send_future.result()
            if not goal_handle.accepted:
                node.get_logger().warn("goal rejected; skipping")
                node.destroy_subscription(action_sub)
                continue
        except Exception as exc:
            node.get_logger().warn(f"goal send failed: {exc}")
            node.destroy_subscription(action_sub)
            continue

        result_future = goal_handle.get_result_async()
        ep_step = 0
        while rclpy.ok() and ep_step < args.max_steps:
            rclpy.spin_once(node, timeout_sec=0.05)
            obs_msg = latest_obs["msg"]
            if obs_msg is None:
                continue
            obs = _build_obs(obs_msg, target_hw)
            if obs is None:
                continue
            rec.append(
                image=obs["image"],
                force_xyz=obs["force"],
                tcp_xyz=obs["tcp_xyz"],
                tcp_q_wxyz=obs["tcp_q"],
                port_xyz=obs["port_xyz"],
                port_q_wxyz=obs["port_q"],
                tcp_pose_err=obs["tcp_pose_err"],
                action_6d=pending["delta"].copy(),
                insertion_event=latest_event["flag"],
            )
            ep_step += 1
            if latest_event["flag"]:
                last_event_value = True
            if result_future.done():
                break

        out_path = args.out_dir / f"{args.port_type}_{n_recorded:04d}.npz"
        try:
            rec.save(str(out_path))
            n_recorded += 1
            node.get_logger().info(
                f"[recorder] saved {out_path} ({len(rec)} frames, "
                f"event={last_event_value})"
            )
        except Exception as exc:
            node.get_logger().warn(f"save failed: {exc}")

        node.destroy_subscription(action_sub)
        time.sleep(0.5)

    node.get_logger().info(
        f"[recorder] done. {n_recorded} episodes saved to {args.out_dir}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]