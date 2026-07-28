"""Simulator-GT collection for a physical SC front-mouth pose model.

This collector intentionally does *not* replace :mod:`DataCollectorScPoseGT`.
The deployed ``best_sc_pose.pt`` was trained on a smaller virtual rectangle;
mixing the two conventions in one output directory would produce confidently
wrong metric pose.  This writer emits a new five-keypoint dataset with the
physical 22.407 x 8.10 mm front-mouth outline and its centre.
"""

from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from .DataCollectorPose2 import (
    clamp_keypoints_to_image,
    compute_padded_bbox,
    compute_visibility_flags,
    project_keypoints,
    ros_image_to_cv2,
    tf_to_4x4,
)
from .sc_mouth_pose_geometry import (
    LOCAL_SC_FRONT_MOUTH_KPS_M,
    SC_MOUTH_KEYPOINT_COUNT,
    format_yolo_pose_label,
    split_for_trial,
    write_dataset_yaml,
)


OUTPUT_DIR = os.path.expanduser(
    os.environ.get("AIC_SC_MOUTH_POSE_OUTPUT_DIR", "~/aic_perception_data/sc_mouth_pose")
)
CAMERA_NAMES = ("left_camera", "center_camera", "right_camera")
SC_SLOTS = tuple(range(int(os.environ.get("AIC_SC_MOUTH_POSE_SLOTS", "5"))))
VIEWPOINTS_PER_TRIAL = int(os.environ.get("AIC_SC_MOUTH_POSE_VIEWPOINTS", "18"))
MIN_VISIBLE_KEYPOINTS = SC_MOUTH_KEYPOINT_COUNT
BBOX_PADDING = 0.25
SAVE_DEBUG = os.environ.get("AIC_SC_MOUTH_POSE_SAVE_DEBUG", "0") == "1"
FAR_VIEW_PROBABILITY = float(os.environ.get("AIC_SC_MOUTH_FAR_VIEW_PROB", "0.30"))


def _quat_multiply(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Compose ROS quaternions (x, y, z, w): ``q_left * q_right``."""

    x1, y1, z1, w1 = q_left
    x2, y2, z2, w2 = q_right
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + y1 * x2 - x1 * y2 + z1 * z2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _rpy_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert an intrinsic RPY perturbation to a ROS quaternion."""

    cr, cp, cy = np.cos([roll * 0.5, pitch * 0.5, yaw * 0.5])
    sr, sp, sy = np.sin([roll * 0.5, pitch * 0.5, yaw * 0.5])
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def sample_viewpoints(
    count: int, rng: random.Random
) -> list[tuple[float, float, float, float, float, float]]:
    """Sample deployment-weighted position and orientation viewpoints.

    Seventy percent stay near the mouth/approach distribution; the remainder
    covers the wider acquisition volume.  This fixes the old collector's
    zero-orientation-diversity leak without making every image extreme.
    """

    samples = []
    for _ in range(count):
        if not 0.0 <= FAR_VIEW_PROBABILITY <= 1.0:
            raise ValueError("AIC_SC_MOUTH_FAR_VIEW_PROB must be within [0, 1]")
        if rng.random() >= FAR_VIEW_PROBABILITY:
            dx = rng.uniform(-0.030, 0.030)
            dy = rng.uniform(-0.030, 0.030)
            dz = rng.uniform(0.005, 0.035)
            max_angle = np.deg2rad(12.0)
            yaw_angle = np.deg2rad(8.0)
        else:
            dx = rng.uniform(-0.060, 0.060)
            dy = rng.uniform(-0.060, 0.060)
            dz = rng.uniform(-0.010, 0.160)
            max_angle = np.deg2rad(15.0)
            yaw_angle = np.deg2rad(12.0)
        samples.append(
            (
                dx,
                dy,
                dz,
                rng.uniform(-max_angle, max_angle),
                rng.uniform(-max_angle, max_angle),
                rng.uniform(-yaw_angle, yaw_angle),
            )
        )
    return samples


class DataCollectorScMouthPoseGT(Policy):
    """Collect physical-mouth labels using timestamp-aligned entrance-frame TF."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._trial_counter = 0
        self._frame_counter = 0
        self._trial_start = int(os.environ.get("AIC_SC_MOUTH_TRIAL_START", "1"))
        self._seed = int(os.environ.get("AIC_SC_MOUTH_SEED", "20260727"))
        for subdir in (
            "images/train",
            "images/val",
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test",
            "metadata",
            "debug",
        ):
            os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
        write_dataset_yaml(OUTPUT_DIR)
        self.get_logger().info(
            "SC mouth GT collector init | "
            f"output_dir={OUTPUT_DIR} trial_start={self._trial_start} "
            f"viewpoints={VIEWPOINTS_PER_TRIAL} physical_front_mouth=22.407x8.100mm"
        )

    def _lookup_tf_at_stamp(self, target: str, source: str, stamp: Time | None):
        try:
            return self._parent_node._tf_buffer.lookup_transform(
                target, source, stamp if stamp is not None else Time(), Duration(seconds=0.20)
            )
        except TransformException:
            try:
                return self._parent_node._tf_buffer.lookup_transform(
                    target, source, Time(), Duration(seconds=0.20)
                )
            except TransformException:
                return None

    @staticmethod
    def _candidate_mouth_frames(slot_idx: int) -> list[str]:
        """Return only entrance-plane frames; never silently fall back to the seat."""

        base = f"task_board/sc_port_{slot_idx}"
        return [
            f"{base}/sc_port_base/sc_port_base_link_entrance",
            f"{base}/sc_port_base_link_entrance",
            f"{base}/sc_port_link_entrance",
        ]

    @staticmethod
    def _camera_data(obs, camera_name: str):
        images = {
            "left_camera": obs.left_image,
            "center_camera": obs.center_image,
            "right_camera": obs.right_image,
        }
        infos = {
            "left_camera": obs.left_camera_info,
            "center_camera": obs.center_camera_info,
            "right_camera": obs.right_camera_info,
        }
        image_msg, info = images[camera_name], infos[camera_name]
        if image_msg is None or info is None:
            return None
        image = ros_image_to_cv2(image_msg)
        if image is None:
            return None
        camera_matrix = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        if camera_matrix[0, 0] <= 0.0:
            return None
        try:
            stamp = Time.from_msg(image_msg.header.stamp)
        except Exception:
            stamp = Time()
        return image, camera_matrix, stamp

    @staticmethod
    def _debug_overlay(image: np.ndarray, labels: list[dict]) -> np.ndarray:
        canvas = image.copy()
        for label in labels:
            points = np.asarray(label["projected_keypoints_px"], dtype=np.float64)
            visible = np.asarray(label["visibility_flags"], dtype=np.int32)
            corners = np.rint(points[:4]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [corners], True, (0, 255, 255), 2, cv2.LINE_AA)
            for index, (point, flag) in enumerate(zip(points, visible)):
                if flag <= 0:
                    continue
                colour = (0, 255, 0) if index == 4 else (0, 128, 255)
                xy = tuple(np.rint(point).astype(int))
                cv2.circle(canvas, xy, 4, colour, -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(index),
                    (xy[0] + 5, xy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colour,
                    1,
                    cv2.LINE_AA,
                )
        return canvas

    def _capture_frame(self, obs, *, trial_index: int, split: str, viewpoint_index: int, viewpoint):
        frame_id = f"t{trial_index:05d}_v{viewpoint_index:02d}"
        metadata = {
            "frame_id": frame_id,
            "trial_index": trial_index,
            "split": split,
            "viewpoint_index": viewpoint_index,
            "viewpoint_offset_m_rpy_rad": {
                "dx": viewpoint[0],
                "dy": viewpoint[1],
                "dz": viewpoint[2],
                "roll": viewpoint[3],
                "pitch": viewpoint[4],
                "yaw": viewpoint[5],
            },
            "label_convention": "physical_front_mouth_22.407x8.100mm_plus_center",
            "cameras": {},
        }
        saved = 0
        no_gt_cameras = 0
        for camera_name in CAMERA_NAMES:
            camera_data = self._camera_data(obs, camera_name)
            if camera_data is None:
                continue
            image, camera_matrix, stamp = camera_data
            height, width = image.shape[:2]
            label_lines: list[str] = []
            label_metadata: list[dict] = []
            for slot in SC_SLOTS:
                transform = None
                frame_used = None
                for candidate in self._candidate_mouth_frames(slot):
                    transform = self._lookup_tf_at_stamp(f"{camera_name}/optical", candidate, stamp)
                    if transform is not None:
                        frame_used = candidate
                        break
                if transform is None:
                    continue

                t_camera_mouth = tf_to_4x4(transform)
                keypoints_px, in_front = project_keypoints(
                    LOCAL_SC_FRONT_MOUTH_KPS_M, t_camera_mouth, camera_matrix
                )
                flags = compute_visibility_flags(keypoints_px, in_front, width, height)
                # This model promises a physical outline.  Keep only frames in
                # which every corner and centre can actually supervise it.
                if int(np.sum(flags == 2)) < MIN_VISIBLE_KEYPOINTS:
                    continue
                bbox = compute_padded_bbox(keypoints_px, in_front, width, height, padding=BBOX_PADDING)
                if bbox is None:
                    continue
                keypoints_clamped = clamp_keypoints_to_image(keypoints_px, flags, width, height)
                try:
                    label_lines.append(
                        format_yolo_pose_label(
                            bbox,
                            keypoints_clamped,
                            flags,
                            width,
                            height,
                        )
                    )
                except ValueError:
                    continue
                label_metadata.append(
                    {
                        "slot": slot,
                        "tf_frame": frame_used,
                        "bbox_px": [float(value) for value in bbox],
                        "local_keypoints_m": LOCAL_SC_FRONT_MOUTH_KPS_M.tolist(),
                        "projected_keypoints_px": keypoints_clamped.tolist(),
                        "visibility_flags": flags.astype(int).tolist(),
                        "T_camera_mouth": t_camera_mouth.tolist(),
                    }
                )

            if not label_lines:
                no_gt_cameras += 1
                continue
            image_name = f"{frame_id}_{camera_name}.png"
            label_name = f"{frame_id}_{camera_name}.txt"
            cv2.imwrite(os.path.join(OUTPUT_DIR, "images", split, image_name), image)
            with open(os.path.join(OUTPUT_DIR, "labels", split, label_name), "w", encoding="utf-8") as stream:
                stream.write("\n".join(label_lines))
            if SAVE_DEBUG:
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, "debug", image_name),
                    self._debug_overlay(image, label_metadata),
                )
            metadata["cameras"][camera_name] = {
                "image": image_name,
                "camera_matrix": camera_matrix.tolist(),
                "labels": label_metadata,
            }
            saved += 1

        with open(os.path.join(OUTPUT_DIR, "metadata", f"{frame_id}.json"), "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
        self._frame_counter += 1
        if saved == 0:
            self.get_logger().warn(
                "SC mouth GT frame produced no labels | "
                f"trial={trial_index} viewpoint={viewpoint_index} no_gt_cameras={no_gt_cameras}"
            )
        elif no_gt_cameras == len(CAMERA_NAMES):
            self.get_logger().error(
                "SC mouth GT had images but no entrance-frame TF in every camera; "
                "skipping rather than generating pseudo-labels."
            )
        return saved

    def _move_to_viewpoint(self, move_robot, initial_tcp, viewpoint):
        dx, dy, dz, roll, pitch, yaw = viewpoint
        initial_quat = np.array(
            [
                initial_tcp.orientation.x,
                initial_tcp.orientation.y,
                initial_tcp.orientation.z,
                initial_tcp.orientation.w,
            ],
            dtype=np.float64,
        )
        quat = _quat_multiply(initial_quat, _rpy_quat(roll, pitch, yaw))
        self.set_pose_target(
            move_robot=move_robot,
            pose=Pose(
                position=Point(
                    x=initial_tcp.position.x + dx,
                    y=initial_tcp.position.y + dy,
                    z=initial_tcp.position.z + dz,
                ),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            ),
            stiffness=[95.0, 95.0, 95.0, 50.0, 50.0, 50.0],
            damping=[65.0, 65.0, 65.0, 30.0, 30.0, 30.0],
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self._trial_counter += 1
        trial_index = self._trial_start + self._trial_counter - 1
        split = split_for_trial(trial_index)
        self.get_logger().info(
            f"SC mouth GT collect trial={trial_index} split={split} "
            f"task={task.port_type}/{task.target_module_name}"
        )
        self.sleep_for(4.0)
        observation = get_observation()
        if observation is None:
            self.get_logger().error("No observation; skipping trial")
            return True

        initial_tcp = observation.controller_state.tcp_pose
        viewpoints = sample_viewpoints(
            VIEWPOINTS_PER_TRIAL, random.Random(self._seed + trial_index)
        )
        total_saved = 0
        previous_position = np.zeros(3, dtype=np.float64)
        for index, viewpoint in enumerate(viewpoints, start=1):
            self._move_to_viewpoint(move_robot, initial_tcp, viewpoint)
            position = np.asarray(viewpoint[:3], dtype=np.float64)
            self.sleep_for(max(1.8, 10.0 * float(np.linalg.norm(position - previous_position))))
            previous_position = position
            observation = get_observation()
            if observation is not None:
                total_saved += self._capture_frame(
                    observation,
                    trial_index=trial_index,
                    split=split,
                    viewpoint_index=index,
                    viewpoint=viewpoint,
                )

        self._move_to_viewpoint(move_robot, initial_tcp, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.sleep_for(2.0)
        write_dataset_yaml(OUTPUT_DIR)
        send_feedback(
            f"SC mouth GT collector saved={total_saved} images trial={trial_index} "
            f"split={split} out={OUTPUT_DIR}"
        )
        self.get_logger().info(
            f"SC MOUTH GT collection done | trial={trial_index} split={split} saved={total_saved}"
        )
        return True
