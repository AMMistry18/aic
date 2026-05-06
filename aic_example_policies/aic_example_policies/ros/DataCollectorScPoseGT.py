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
from .perception_core import PerceptionCore

OUTPUT_DIR = os.path.expanduser(
    os.environ.get("AIC_SC_POSE_OUTPUT_DIR", "~/aic_perception_data/pose_sc_gt_raw")
)
CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]

VIEWPOINTS_PER_TRIAL = 18
VAL_EVERY_N_RUNS = 5

SC_HALF_WIDTH_M = 0.0044
SC_HALF_HEIGHT_M = 0.0030
MIN_VISIBLE_KEYPOINTS = 2
BBOX_PADDING = 0.30

LOCAL_SC_PORT_KPS = np.array(
    [
        [SC_HALF_WIDTH_M, SC_HALF_HEIGHT_M, 0.0],
        [-SC_HALF_WIDTH_M, SC_HALF_HEIGHT_M, 0.0],
        [-SC_HALF_WIDTH_M, -SC_HALF_HEIGHT_M, 0.0],
        [SC_HALF_WIDTH_M, -SC_HALF_HEIGHT_M, 0.0],
    ],
    dtype=np.float32,
)

CLASS_NAMES = ["sc_port"]
FLIP_IDX = [1, 0, 3, 2]


def format_sc_pose_label(bbox, kps_clamped, flags, img_w: int, img_h: int, class_id: int = 0):
    if kps_clamped.shape[0] != 4 or len(flags) != 4:
        return None
    x_min, y_min, x_max, y_max = bbox
    cx = ((x_min + x_max) / 2.0) / img_w
    cy = ((y_min + y_max) / 2.0) / img_h
    nw = (x_max - x_min) / img_w
    nh = (y_max - y_min) / img_h
    kp_tokens = []
    for pt, vis in zip(kps_clamped, flags):
        px = pt[0] / img_w
        py = pt[1] / img_h
        kp_tokens.append(f"{px:.6f} {py:.6f} {int(vis)}")
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kp_tokens)


def sample_viewpoints(n: int) -> list[tuple[float, float, float]]:
    viewpoints = []
    for _ in range(n):
        viewpoints.append(
            (
                random.uniform(-0.06, 0.06),
                random.uniform(-0.06, 0.06),
                random.uniform(-0.01, 0.16),
            )
        )
    return viewpoints


class DataCollectorScPoseGT(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._frame_counter = 0
        self._trial_counter = 0
        self._run_counter = self._load_run_counter()
        self._split = "val" if (self._run_counter % VAL_EVERY_N_RUNS == 0) else "train"
        self._fallback_detector = PerceptionCore()

        for sub in [
            "images/train",
            "images/val",
            "labels/train",
            "labels/val",
            "metadata",
            "debug",
        ]:
            os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
        self.get_logger().info(
            f"SC GT collector init | run={self._run_counter} split={self._split} "
            f"viewpoints={VIEWPOINTS_PER_TRIAL} output_dir={OUTPUT_DIR}"
        )

    def _load_run_counter(self) -> int:
        counter_file = os.path.join(OUTPUT_DIR, ".run_counter")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            with open(counter_file, "r", encoding="utf-8") as f:
                value = int(f.read().strip()) + 1
        except Exception:
            value = 1
        with open(counter_file, "w", encoding="utf-8") as f:
            f.write(str(value))
        return value

    def _lookup_tf_at_stamp(self, target: str, source: str, stamp: Time | None):
        # First try timestamp-aligned lookup for geometric accuracy.
        try:
            tf_time = stamp if stamp is not None else Time()
            return self._parent_node._tf_buffer.lookup_transform(
                target, source, tf_time, Duration(seconds=0.20)
            )
        except TransformException:
            pass
        # Fall back to latest TF so collection still proceeds when exact-time
        # transforms are unavailable in the current buffer window.
        try:
            return self._parent_node._tf_buffer.lookup_transform(
                target, source, Time(), Duration(seconds=0.20)
            )
        except TransformException:
            return None

    def _get_cam_data(self, obs, cam_name: str):
        img_map = {
            "left_camera": obs.left_image,
            "center_camera": obs.center_image,
            "right_camera": obs.right_image,
        }
        info_map = {
            "left_camera": obs.left_camera_info,
            "center_camera": obs.center_camera_info,
            "right_camera": obs.right_camera_info,
        }
        msg = img_map.get(cam_name)
        info = info_map.get(cam_name)
        if msg is None or info is None:
            return None

        img = ros_image_to_cv2(msg)
        if img is None:
            return None
        K = np.array(info.k).reshape(3, 3)
        if K[0, 0] == 0:
            return None
        stamp = Time()
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            try:
                stamp = Time.from_msg(msg.header.stamp)
            except Exception:
                stamp = Time()
        return img, K, stamp

    def _candidate_sc_frames(self, slot_idx: int) -> list[str]:
        base = f"task_board/sc_port_{slot_idx}"
        return [
            f"{base}/sc_port_base/sc_port_base_link_entrance",
            f"{base}/sc_port_base_link_entrance",
            f"{base}/sc_port_base/sc_port_base_link",
            f"{base}/sc_port_link_entrance",
            f"{base}/sc_port_link",
        ]

    def _capture_frame(self, obs, viewpoint_idx: int, vp_xyz):
        frame_id = f"{self._run_counter:04d}_{self._trial_counter:02d}_{self._frame_counter:03d}"
        metadata = {
            "frame_id": frame_id,
            "run": self._run_counter,
            "trial": self._trial_counter,
            "split": self._split,
            "viewpoint_idx": viewpoint_idx,
            "viewpoint_offset": {"dx": vp_xyz[0], "dy": vp_xyz[1], "dz": vp_xyz[2]},
            "cameras": {},
        }
        saved = 0
        labeled = 0
        debug_counts = {
            "camera_missing": 0,
            "tf_miss": 0,
            "kpt_visibility_drop": 0,
            "bbox_drop": 0,
            "format_drop": 0,
        }

        for cam in CAMERA_NAMES:
            cam_data = self._get_cam_data(obs, cam)
            if cam_data is None:
                debug_counts["camera_missing"] += 1
                continue
            img, K, stamp = cam_data
            h, w = img.shape[:2]
            cam_frame = f"{cam}/optical"

            label_lines = []
            cam_meta_labels = []
            for slot in (0, 1):
                tf_port = None
                frame_used = None
                for frame in self._candidate_sc_frames(slot):
                    tf_port = self._lookup_tf_at_stamp(cam_frame, frame, stamp)
                    if tf_port is not None:
                        frame_used = frame
                        break
                if tf_port is None:
                    debug_counts["tf_miss"] += 1
                    continue

                T_cam_port = tf_to_4x4(tf_port)
                kps2d, in_front = project_keypoints(LOCAL_SC_PORT_KPS, T_cam_port, K)
                flags = compute_visibility_flags(kps2d, in_front, w, h)
                if int(np.sum(flags == 2)) < MIN_VISIBLE_KEYPOINTS:
                    debug_counts["kpt_visibility_drop"] += 1
                    continue

                bbox = compute_padded_bbox(kps2d, in_front, w, h, padding=BBOX_PADDING)
                if bbox is None:
                    debug_counts["bbox_drop"] += 1
                    continue
                kps_clamped = clamp_keypoints_to_image(kps2d, flags, w, h)
                line = format_sc_pose_label(
                    bbox=bbox,
                    kps_clamped=kps_clamped,
                    flags=flags,
                    img_w=w,
                    img_h=h,
                    class_id=0,
                )
                if line is None:
                    debug_counts["format_drop"] += 1
                    continue
                label_lines.append(line)
                cam_meta_labels.append(
                    {
                        "slot": slot,
                        "tf_frame": frame_used,
                        "visible_keypoints": int(np.sum(flags == 2)),
                        "bbox_px": [float(x) for x in bbox],
                    }
                )

            if not label_lines:
                # Fallback for environments where SC GT TF frames are unavailable.
                # Use the current SC color detector output as pseudo-labels.
                for det in self._fallback_detector.detect_sc(img)[:2]:
                    corners = np.asarray(det.get("corners", []), dtype=np.float32)
                    if corners.shape != (4, 2):
                        continue
                    x, y, bw, bh = det["bbox"]
                    bbox = (float(x), float(y), float(x + bw), float(y + bh))
                    flags = np.array([2, 2, 2, 2], dtype=np.int32)
                    kps_clamped = clamp_keypoints_to_image(corners, flags, w, h)
                    line = format_sc_pose_label(
                        bbox=bbox,
                        kps_clamped=kps_clamped,
                        flags=flags,
                        img_w=w,
                        img_h=h,
                        class_id=0,
                    )
                    if line is None:
                        continue
                    label_lines.append(line)
                    cam_meta_labels.append(
                        {
                            "slot": -1,
                            "tf_frame": "fallback_color_filter",
                            "visible_keypoints": 4,
                            "bbox_px": [float(v) for v in bbox],
                        }
                    )

            if not label_lines:
                continue

            img_name = f"{frame_id}_{cam}.png"
            lbl_name = f"{frame_id}_{cam}.txt"
            cv2.imwrite(os.path.join(OUTPUT_DIR, "images", self._split, img_name), img)
            # Keep a debug copy of every captured image for quick visual checks.
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "debug", img_name),
                img,
            )
            with open(
                os.path.join(OUTPUT_DIR, "labels", self._split, lbl_name),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("\n".join(label_lines))

            metadata["cameras"][cam] = {
                "image": img_name,
                "label_count": len(label_lines),
                "labels": cam_meta_labels,
            }
            saved += 1
            if label_lines:
                labeled += 1

        with open(
            os.path.join(OUTPUT_DIR, "metadata", f"{frame_id}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

        self._frame_counter += 1
        if saved == 0:
            self.get_logger().warn(
                "SC GT frame produced no saved labels | "
                f"tf_miss={debug_counts['tf_miss']} "
                f"vis_drop={debug_counts['kpt_visibility_drop']} "
                f"bbox_drop={debug_counts['bbox_drop']} "
                f"fmt_drop={debug_counts['format_drop']} "
                f"cam_missing={debug_counts['camera_missing']}"
            )
        return saved, labeled

    def _move_to_offset(self, move_robot, initial_tcp, dx, dy, dz):
        self.set_pose_target(
            move_robot=move_robot,
            pose=Pose(
                position=Point(
                    x=initial_tcp.position.x + dx,
                    y=initial_tcp.position.y + dy,
                    z=initial_tcp.position.z + dz,
                ),
                orientation=Quaternion(
                    x=initial_tcp.orientation.x,
                    y=initial_tcp.orientation.y,
                    z=initial_tcp.orientation.z,
                    w=initial_tcp.orientation.w,
                ),
            ),
            stiffness=[95.0, 95.0, 95.0, 50.0, 50.0, 50.0],
            damping=[65.0, 65.0, 65.0, 30.0, 30.0, 30.0],
        )

    def _write_dataset_yaml(self):
        train_dir = os.path.join(OUTPUT_DIR, "images", "train")
        val_dir = os.path.join(OUTPUT_DIR, "images", "val")
        train_count = len([f for f in os.listdir(train_dir) if f.endswith(".png")])
        val_count = len([f for f in os.listdir(val_dir) if f.endswith(".png")])
        content = f"""# SC port pose dataset from simulator TF ground truth.
# Run-level split every {VAL_EVERY_N_RUNS}th run to keep board states disjoint.
# Train images: {train_count}
# Val images:   {val_count}
path: {OUTPUT_DIR}
train: images/train
val: images/val
nc: 1
names: {CLASS_NAMES}
kpt_shape: [4, 3]
flip_idx: {FLIP_IDX}
"""
        with open(os.path.join(OUTPUT_DIR, "aic_sc_pose_raw.yaml"), "w", encoding="utf-8") as f:
            f.write(content)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self._trial_counter += 1
        self.get_logger().info(
            f"SC GT collect trial={self._trial_counter} task={task.port_type}/{task.target_module_name}"
        )
        self.sleep_for(4.0)
        obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation; skipping")
            return True

        initial_tcp = obs.controller_state.tcp_pose
        viewpoints = sample_viewpoints(VIEWPOINTS_PER_TRIAL)
        total_saved = 0
        total_labeled = 0
        prev = (0.0, 0.0, 0.0)
        for i, (dx, dy, dz) in enumerate(viewpoints):
            self._move_to_offset(move_robot, initial_tcp, dx, dy, dz)
            dist = np.linalg.norm(np.array([dx, dy, dz]) - np.array(prev))
            self.sleep_for(max(1.8, 10.0 * float(dist)))
            prev = (dx, dy, dz)
            obs = get_observation()
            if obs is None:
                continue
            n_saved, n_labeled = self._capture_frame(obs, i, (dx, dy, dz))
            total_saved += n_saved
            total_labeled += n_labeled
            self.get_logger().info(
                f"  vp {i+1}/{len(viewpoints)} saved={n_saved} labeled={n_labeled} "
                f"total_saved={total_saved} total_labeled={total_labeled}"
            )

        self._move_to_offset(move_robot, initial_tcp, 0.0, 0.0, 0.0)
        self.sleep_for(2.0)
        self._write_dataset_yaml()
        send_feedback(
            f"SC GT collector saved={total_saved} images labeled={total_labeled} "
            f"out={OUTPUT_DIR}"
        )
        self.get_logger().info(
            f"SC GT collection done | saved={total_saved} labeled={total_labeled}"
        )
        return True
