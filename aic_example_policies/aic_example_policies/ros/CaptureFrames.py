"""
CaptureFrames — AIC Policy that saves camera frames + ground truth for FoundationPose testing.

Run with ground_truth:=true so we get the scoring TFs.

Usage (Terminal 1 - eval container):
    /entrypoint.sh ground_truth:=true start_aic_engine:=true gazebo_gui:=false

Usage (Terminal 2 - host):
    pixi run ros2 run aic_model aic_model \
        --ros-args -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.CaptureFrames
"""

import json
import os
import time

import cv2
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException


class CaptureFrames(Policy):
    """Capture RGB + depth + intrinsics + ground truth from wrist cameras."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._output_dir = os.path.expanduser("~/fp_test_data")
        self._frame_count = 0
        self._trial_count = 0
        self.get_logger().info(f"CaptureFrames init | output: {self._output_dir}")

    def _save_observation(self, obs, task, trial_dir):
        """Save RGB images, depth, camera intrinsics, and ground truth from one observation."""
        cameras = [
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        ]

        for cam_name, img_msg, info_msg in cameras:
            cam_dir = os.path.join(trial_dir, cam_name)
            os.makedirs(os.path.join(cam_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(cam_dir, "masks"), exist_ok=True)

            # Save RGB
            if img_msg.encoding == "rgb8":
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                    img_msg.height, img_msg.width, 3
                )
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img_msg.encoding == "bgr8":
                img_bgr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                    img_msg.height, img_msg.width, 3
                )
            else:
                self.get_logger().warn(f"Unknown encoding: {img_msg.encoding}")
                continue

            fname = f"{self._frame_count:06d}"
            cv2.imwrite(os.path.join(cam_dir, "rgb", f"{fname}.png"), img_bgr)

            # Save camera intrinsics (3x3 K matrix)
            K = np.array(info_msg.k).reshape(3, 3)
            k_path = os.path.join(cam_dir, "cam_K.txt")
            if not os.path.exists(k_path):
                np.savetxt(k_path, K, fmt="%.6f")
                self.get_logger().info(
                    f"  Saved cam_K.txt for {cam_name}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                    f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}"
                )

            # Save camera frame_id and dimensions
            meta_path = os.path.join(cam_dir, "camera_meta.json")
            if not os.path.exists(meta_path):
                meta = {
                    "frame_id": info_msg.header.frame_id,
                    "width": info_msg.width,
                    "height": info_msg.height,
                    "distortion_model": info_msg.distortion_model,
                    "D": list(info_msg.d),
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

        self._frame_count += 1

    def _save_ground_truth(self, task, trial_dir):
        """Save ground truth TF poses for the target port."""
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        gt = {}

        for frame_name, label in [
            (port_frame, "port"),
            (cable_tip_frame, "plug_tip"),
            ("gripper/tcp", "gripper_tcp"),
        ]:
            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", frame_name, Time()
                )
                t = tf.transform.translation
                r = tf.transform.rotation
                gt[label] = {
                    "frame": frame_name,
                    "translation": {"x": t.x, "y": t.y, "z": t.z},
                    "rotation": {"x": r.x, "y": r.y, "z": r.z, "w": r.w},
                }
            except TransformException as e:
                self.get_logger().warn(f"Could not get TF for {frame_name}: {e}")

        # Also try to get the task board base and target module frames
        for extra_frame in [
            "task_board/task_board_base_link",
            f"task_board/{task.target_module_name}/nic_card_link",
            f"task_board/{task.target_module_name}/sc_port_link",
        ]:
            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", extra_frame, Time()
                )
                t = tf.transform.translation
                r = tf.transform.rotation
                label = extra_frame.replace("/", "__")
                gt[label] = {
                    "frame": extra_frame,
                    "translation": {"x": t.x, "y": t.y, "z": t.z},
                    "rotation": {"x": r.x, "y": r.y, "z": r.z, "w": r.w},
                }
            except TransformException:
                pass

        # Also save camera-to-base transforms for each camera
        for cam in ["left_camera/optical", "center_camera/optical", "right_camera/optical"]:
            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", cam, Time()
                )
                t = tf.transform.translation
                r = tf.transform.rotation
                gt[cam] = {
                    "frame": cam,
                    "translation": {"x": t.x, "y": t.y, "z": t.z},
                    "rotation": {"x": r.x, "y": r.y, "z": r.z, "w": r.w},
                }
            except TransformException:
                pass

        gt_path = os.path.join(trial_dir, f"ground_truth_{self._frame_count:06d}.json")
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)

    def _save_task_info(self, task, trial_dir):
        """Save the task parameters."""
        task_info = {
            "cable_type": task.cable_type,
            "cable_name": task.cable_name,
            "plug_type": task.plug_type,
            "plug_name": task.plug_name,
            "port_type": task.port_type,
            "port_name": task.port_name,
            "target_module_name": task.target_module_name,
            "time_limit": task.time_limit,
        }
        with open(os.path.join(trial_dir, "task_info.json"), "w") as f:
            json.dump(task_info, f, indent=2)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self._trial_count += 1
        trial_dir = os.path.join(
            self._output_dir, f"trial_{self._trial_count:03d}"
        )
        os.makedirs(trial_dir, exist_ok=True)

        self.get_logger().info(
            f"CaptureFrames trial {self._trial_count} | "
            f"{task.plug_type} → {task.port_name} on {task.target_module_name}"
        )

        # Save task info
        self._save_task_info(task, trial_dir)

        # Wait a moment for TFs to stabilize
        self.sleep_for(2.0)

        # Capture frames from the current position (home position)
        self.get_logger().info("Capturing from home position...")
        obs = get_observation()
        self._save_observation(obs, task, trial_dir)
        self._save_ground_truth(task, trial_dir)

        # Move to a few different viewpoints to get varied perspectives
        # These are small offsets from current position in TCP frame
        viewpoints = [
            (0.0, 0.0, -0.05),     # move down 5cm
            (0.03, 0.0, -0.05),    # offset right + down
            (-0.03, 0.0, -0.05),   # offset left + down
            (0.0, 0.03, -0.08),    # offset forward + down more
            (0.0, -0.03, -0.03),   # offset back + down less
        ]

        for i, (dx, dy, dz) in enumerate(viewpoints):
            self.get_logger().info(f"  Moving to viewpoint {i+1}/{len(viewpoints)}...")

            # Get current gripper pose
            try:
                gripper_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", "gripper/tcp", Time()
                )
            except TransformException as e:
                self.get_logger().warn(f"Could not get gripper TF: {e}")
                continue

            from geometry_msgs.msg import Point, Pose, Quaternion

            target_pose = Pose(
                position=Point(
                    x=gripper_tf.transform.translation.x + dx,
                    y=gripper_tf.transform.translation.y + dy,
                    z=gripper_tf.transform.translation.z + dz,
                ),
                orientation=Quaternion(
                    x=gripper_tf.transform.rotation.x,
                    y=gripper_tf.transform.rotation.y,
                    z=gripper_tf.transform.rotation.z,
                    w=gripper_tf.transform.rotation.w,
                ),
            )

            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            self.sleep_for(2.0)  # wait for arm to settle

            obs = get_observation()
            self._save_observation(obs, task, trial_dir)
            self._save_ground_truth(task, trial_dir)

        total_frames = self._frame_count
        self.get_logger().info(
            f"Trial {self._trial_count} done | {total_frames} total frames saved to {self._output_dir}"
        )

        return True