"""Simulator-ground-truth collector for a separate SC plug-pose model.

This collector is intentionally distinct from ``DataCollectorScPoseGT``, which
labels SC *ports*.  Run with ``ground_truth:=true`` and the canonical SC task
asset contract:

* cable asset ``sfp_sc_cable_reversed`` (SC plug on cable connection 0),
* task plug ``sc_tip``, port ``sc_port_base``, and target ``sc_port_0|1``,
* runtime GT frame ``cable_N/sc_tip_link``.

Each randomized trial is assigned wholly to train, validation, or test.  The
three wrist cameras and a few settled samples provide image diversity; grasp
and background pose diversity comes from ``generate_sc_plug_pose_trials.py``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model.sc_plug_pose_geometry import (
    SC_PLUG_LOCAL_KEYPOINTS_M,
    format_yolo_pose_label,
    padded_bbox,
    project_keypoints,
    visibility_flags,
    write_dataset_yaml,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from .DataCollectorPose2 import ros_image_to_cv2, tf_to_4x4


OUTPUT_DIR = Path(
    os.path.expanduser(
        os.environ.get(
            "AIC_SC_PLUG_POSE_OUTPUT_DIR",
            "~/aic_perception_data/sc_plug_pose",
        )
    )
).resolve()
CAMERA_NAMES = ("left_camera", "center_camera", "right_camera")
VIEWPOINTS_PER_TRIAL = int(os.environ.get("AIC_SC_PLUG_POSE_VIEWPOINTS", "1"))
FRAMES_PER_VIEWPOINT = int(os.environ.get("AIC_SC_PLUG_POSE_FRAMES_PER_VIEW", "3"))
SETTLE_SECONDS = float(os.environ.get("AIC_SC_PLUG_POSE_SETTLE_S", "0.6"))
FRAME_INTERVAL_SECONDS = float(
    os.environ.get("AIC_SC_PLUG_POSE_FRAME_INTERVAL_S", "0.18")
)
INITIAL_SETTLE_SECONDS = float(
    os.environ.get("AIC_SC_PLUG_POSE_INITIAL_SETTLE_S", "2.0")
)
RETURN_SETTLE_SECONDS = float(
    os.environ.get("AIC_SC_PLUG_POSE_RETURN_SETTLE_S", "1.0")
)
TRIAL_START_INDEX = int(os.environ.get("AIC_SC_PLUG_POSE_TRIAL_START", "1"))
MIN_VISIBLE_KEYPOINTS = 6
BBOX_PADDING = 0.28


def _split_for_trial(global_trial_index: int) -> str:
    """Stable 80/10/10 split from the generated trial index."""

    cycle = (int(global_trial_index) - 1) % 10
    if cycle == 9:
        return "test"
    if cycle == 8:
        return "val"
    return "train"


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(left, dtype=np.float64).reshape(4)
    x2, y2, z2, w2 = np.asarray(right, dtype=np.float64).reshape(4)
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _euler_xyz_to_quat_xyzw(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, cy, cz = np.cos(np.array([rx, ry, rz], dtype=np.float64) * 0.5)
    sx, sy, sz = np.sin(np.array([rx, ry, rz], dtype=np.float64) * 0.5)
    return np.array(
        [
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ],
        dtype=np.float64,
    )


def sample_viewpoints(count: int, rng: random.Random) -> list[dict[str, float]]:
    """Sample small, safe wrist motions around the actual handoff pose."""

    canonical = [
        {"dx": 0.0, "dy": 0.0, "dz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        {"dx": 0.018, "dy": 0.0, "dz": 0.020, "rx": 0.08, "ry": 0.0, "rz": 0.0},
        {"dx": -0.018, "dy": 0.0, "dz": 0.020, "rx": -0.08, "ry": 0.0, "rz": 0.0},
        {"dx": 0.0, "dy": 0.018, "dz": 0.025, "rx": 0.0, "ry": 0.08, "rz": 0.0},
    ]
    viewpoints = canonical[: max(0, min(len(canonical), count))]
    while len(viewpoints) < count:
        viewpoints.append(
            {
                "dx": rng.uniform(-0.03, 0.03),
                "dy": rng.uniform(-0.03, 0.03),
                "dz": rng.uniform(-0.004, 0.055),
                "rx": rng.uniform(-0.16, 0.16),
                "ry": rng.uniform(-0.16, 0.16),
                "rz": rng.uniform(-0.12, 0.12),
            }
        )
    rng.shuffle(viewpoints)
    return viewpoints


class DataCollectorScPlugPoseGT(Policy):
    """Collect RGB/YOLO labels from the canonical SC tip ground-truth TF."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._run_number = self._next_run_number()
        self._local_trial_number = 0
        self._rng = random.Random(2017 * self._run_number + TRIAL_START_INDEX)
        self._prepare_directories()
        write_dataset_yaml(OUTPUT_DIR)
        self.get_logger().info(
            "SC plug GT collector initialized | "
            f"run={self._run_number} global_trial_start={TRIAL_START_INDEX} "
            f"split=stable-trial-80/10/10 views={VIEWPOINTS_PER_TRIAL} "
            f"frames/view={FRAMES_PER_VIEWPOINT} output={OUTPUT_DIR}"
        )

    def _prepare_directories(self):
        for split in ("train", "val", "test"):
            (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
        if os.environ.get("AIC_SC_PLUG_POSE_SAVE_DEBUG", "0") == "1":
            (OUTPUT_DIR / "debug").mkdir(parents=True, exist_ok=True)

    def _next_run_number(self) -> int:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        counter = OUTPUT_DIR / ".run_counter"
        try:
            value = int(counter.read_text(encoding="utf-8").strip()) + 1
        except (OSError, ValueError):
            value = 1
        counter.write_text(str(value), encoding="utf-8")
        return value

    def _lookup_at_stamp(self, target: str, source: str, stamp: Time):
        try:
            return self._parent_node._tf_buffer.lookup_transform(
                target, source, stamp, Duration(seconds=0.25)
            )
        except TransformException:
            return None

    @staticmethod
    def _stamp_from_image(image_msg: Any) -> Time:
        try:
            return Time.from_msg(image_msg.header.stamp)
        except Exception:
            return Time()

    @staticmethod
    def _stamp_dict(image_msg: Any) -> dict[str, int]:
        stamp = image_msg.header.stamp
        return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}

    @staticmethod
    def _camera_messages(obs, camera_name: str):
        image_map = {
            "left_camera": obs.left_image,
            "center_camera": obs.center_image,
            "right_camera": obs.right_image,
        }
        info_map = {
            "left_camera": obs.left_camera_info,
            "center_camera": obs.center_camera_info,
            "right_camera": obs.right_camera_info,
        }
        return image_map.get(camera_name), info_map.get(camera_name)

    @staticmethod
    def _tip_frame_candidates(task: Task) -> list[str]:
        requested = f"{task.cable_name}/{task.plug_name}_link"
        candidates = [
            requested,
            f"{task.cable_name}/sc_tip_link",
            "cable_0/sc_tip_link",
            "sc_tip_link",
        ]
        return list(dict.fromkeys(candidates))

    def _resolve_tip_frame(self, camera_frame: str, candidates: list[str], stamp: Time):
        for frame in candidates:
            transform = self._lookup_at_stamp(camera_frame, frame, stamp)
            if transform is not None:
                return frame, transform
        return None, None

    def _capture_observation(
        self,
        obs,
        task: Task,
        global_trial_index: int,
        viewpoint_index: int,
        sample_index: int,
        viewpoint: dict[str, float],
    ) -> int:
        frame_id = (
            f"trial_{global_trial_index:04d}_v{viewpoint_index:03d}_"
            f"s{sample_index:02d}"
        )
        split = _split_for_trial(global_trial_index)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "frame_id": frame_id,
            "run": self._run_number,
            "global_trial_index": global_trial_index,
            "generated_trial_name": f"trial_{global_trial_index:04d}",
            "split": split,
            "asset_contract": {
                "cable_asset": "sfp_sc_cable_reversed",
                "plug_asset": "SC Plug",
                "plug_frame": "sc_tip_link",
                "port_asset": "SC Port",
                "port_frame": "sc_port_base_link",
            },
            "tip_frame_requested": f"{task.cable_name}/{task.plug_name}_link",
            "viewpoint": viewpoint,
            "local_keypoints_m": SC_PLUG_LOCAL_KEYPOINTS_M.tolist(),
            "cameras": {},
        }
        saved = 0
        candidates = self._tip_frame_candidates(task)
        for camera_name in CAMERA_NAMES:
            image_msg, camera_info = self._camera_messages(obs, camera_name)
            if image_msg is None or camera_info is None:
                continue
            image = ros_image_to_cv2(image_msg)
            if image is None:
                continue
            K = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
            if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
                continue
            stamp = self._stamp_from_image(image_msg)
            camera_frame = f"{camera_name}/optical"
            tip_frame, camera_from_tip_msg = self._resolve_tip_frame(
                camera_frame, candidates, stamp
            )
            if camera_from_tip_msg is None:
                continue
            world_from_camera_msg = self._lookup_at_stamp(
                "base_link", camera_frame, stamp
            )
            world_from_tip_msg = self._lookup_at_stamp("base_link", tip_frame, stamp)
            if world_from_camera_msg is None or world_from_tip_msg is None:
                continue

            camera_from_tip = tf_to_4x4(camera_from_tip_msg)
            pixels, in_front = project_keypoints(
                SC_PLUG_LOCAL_KEYPOINTS_M, camera_from_tip, K
            )
            height, width = image.shape[:2]
            flags = visibility_flags(pixels, in_front, width, height)
            if int(np.count_nonzero(flags == 2)) < MIN_VISIBLE_KEYPOINTS:
                continue
            bbox = padded_bbox(
                pixels, in_front, width, height, padding=BBOX_PADDING
            )
            if bbox is None:
                continue
            label = format_yolo_pose_label(bbox, pixels, flags, width, height)
            image_name = f"{frame_id}_{camera_name}.png"
            label_name = f"{frame_id}_{camera_name}.txt"
            image_path = OUTPUT_DIR / "images" / split / image_name
            label_path = OUTPUT_DIR / "labels" / split / label_name
            if not cv2.imwrite(str(image_path), image):
                continue
            label_path.write_text(label + "\n", encoding="utf-8")

            if os.environ.get("AIC_SC_PLUG_POSE_SAVE_DEBUG", "0") == "1":
                debug = image.copy()
                for index, point in enumerate(pixels):
                    if flags[index] == 2:
                        xy = tuple(np.rint(point).astype(int).tolist())
                        cv2.circle(debug, xy, 4, (0, 255, 255), -1)
                        cv2.putText(
                            debug,
                            str(index),
                            (xy[0] + 4, xy[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 0),
                            2,
                        )
                cv2.imwrite(str(OUTPUT_DIR / "debug" / image_name), debug)

            metadata["cameras"][camera_name] = {
                "image": str(image_path.relative_to(OUTPUT_DIR)),
                "label": str(label_path.relative_to(OUTPUT_DIR)),
                "stamp": self._stamp_dict(image_msg),
                "camera_frame": camera_frame,
                "tip_frame": tip_frame,
                "image_width": width,
                "image_height": height,
                "K": K.tolist(),
                "T_camera_from_tip": camera_from_tip.tolist(),
                "T_world_from_camera": tf_to_4x4(world_from_camera_msg).tolist(),
                "T_world_from_tip": tf_to_4x4(world_from_tip_msg).tolist(),
                "keypoints_px": pixels.tolist(),
                "visibility": flags.tolist(),
                "bbox_xyxy": list(map(float, bbox)),
            }
            saved += 1

        if saved:
            metadata_path = OUTPUT_DIR / "metadata" / f"{frame_id}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return saved

    def _move_to_viewpoint(self, move_robot, initial_pose: Pose, viewpoint: dict[str, float]):
        initial_q = np.array(
            [
                initial_pose.orientation.x,
                initial_pose.orientation.y,
                initial_pose.orientation.z,
                initial_pose.orientation.w,
            ],
            dtype=np.float64,
        )
        delta_q = _euler_xyz_to_quat_xyzw(
            viewpoint["rx"], viewpoint["ry"], viewpoint["rz"]
        )
        target_q = _quat_multiply_xyzw(initial_q, delta_q)
        self.set_pose_target(
            move_robot=move_robot,
            pose=Pose(
                position=Point(
                    x=initial_pose.position.x + viewpoint["dx"],
                    y=initial_pose.position.y + viewpoint["dy"],
                    z=initial_pose.position.z + viewpoint["dz"],
                ),
                orientation=Quaternion(
                    x=float(target_q[0]),
                    y=float(target_q[1]),
                    z=float(target_q[2]),
                    w=float(target_q[3]),
                ),
            ),
            stiffness=[100.0, 100.0, 100.0, 55.0, 55.0, 55.0],
            damping=[65.0, 65.0, 65.0, 32.0, 32.0, 32.0],
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        if (
            str(task.port_type).lower() != "sc"
            or str(task.plug_type).lower() != "sc"
            or str(task.plug_name).lower() != "sc_tip"
            or str(task.port_name).lower() != "sc_port_base"
            or str(task.target_module_name).lower()
            not in {"sc_port_0", "sc_port_1"}
        ):
            self.get_logger().info(
                "Skipping noncanonical SC plug task while collecting: "
                f"plug={task.plug_type}/{task.plug_name} "
                f"port={task.port_type}/{task.port_name} "
                f"target={task.target_module_name}"
            )
            return True

        self._local_trial_number += 1
        global_trial_index = TRIAL_START_INDEX + self._local_trial_number - 1
        split = _split_for_trial(global_trial_index)
        self.sleep_for(INITIAL_SETTLE_SECONDS)
        initial_observation = get_observation()
        if initial_observation is None:
            self.get_logger().error("No initial observation; SC plug collection skipped")
            return False
        initial_pose = initial_observation.controller_state.tcp_pose
        viewpoints = sample_viewpoints(VIEWPOINTS_PER_TRIAL, self._rng)
        saved = 0
        for viewpoint_index, viewpoint in enumerate(viewpoints):
            self._move_to_viewpoint(move_robot, initial_pose, viewpoint)
            rotation_size = math.sqrt(
                viewpoint["rx"] ** 2
                + viewpoint["ry"] ** 2
                + viewpoint["rz"] ** 2
            )
            translation_size = math.sqrt(
                viewpoint["dx"] ** 2
                + viewpoint["dy"] ** 2
                + viewpoint["dz"] ** 2
            )
            self.sleep_for(
                max(SETTLE_SECONDS, 7.0 * translation_size + 1.2 * rotation_size)
            )
            for sample_index in range(FRAMES_PER_VIEWPOINT):
                observation = get_observation()
                if observation is not None:
                    saved += self._capture_observation(
                        observation,
                        task,
                        global_trial_index,
                        viewpoint_index,
                        sample_index,
                        viewpoint,
                    )
                if sample_index + 1 < FRAMES_PER_VIEWPOINT:
                    self.sleep_for(FRAME_INTERVAL_SECONDS)
            self.get_logger().info(
                f"SC plug GT trial={global_trial_index:04d} "
                f"view={viewpoint_index + 1}/{len(viewpoints)} saved={saved}"
            )

        self._move_to_viewpoint(
            move_robot,
            initial_pose,
            {"dx": 0.0, "dy": 0.0, "dz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
        self.sleep_for(RETURN_SETTLE_SECONDS)
        yaml_path = write_dataset_yaml(OUTPUT_DIR)
        message = (
            f"SC plug GT collection complete: trial={global_trial_index:04d} "
            f"saved={saved} split={split} dataset={yaml_path}"
        )
        send_feedback(message)
        self.get_logger().info(message)
        return saved >= 2
