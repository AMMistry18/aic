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

OUTPUT_DIR = os.path.expanduser(
    os.environ.get("AIC_SC_POSE_OUTPUT_DIR", "~/aic_perception_data/pose_sc_gt_raw")
)
CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]

# Every SC slot that may exist on the board.  This loop used to be hardcoded to
# (0, 1), which matched the qualification xacro.  task_board.urdf.xacro now
# declares sc_port_0..4 (3 on rail 0, 2 on rail 1), and an image containing five
# SC ports with only two of them labelled does not merely lose data -- it teaches
# the detector that the other three are background.  That is the same class of
# silent poisoning as the pseudo-label path removed below, so keep this range at
# or above the number of slots the board can carry.  Slots that are not present
# in a given trial simply fail their TF lookup and are skipped.
SC_SLOTS = tuple(range(int(os.environ.get("AIC_SC_POSE_SLOTS", "5"))))

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
        # No fallback detector by design -- see _capture_frame.  This collector
        # emits projected-TF labels or nothing.

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
        """Frames that are the port MOUTH, and nothing else.

        Every entry here must be the entrance plane, because the caller takes the
        first one that resolves and labels against it.  This list used to end
        with ``sc_port_base_link`` and ``sc_port_link``:

          * ``sc_port_base_link`` is the SEAT -- 15.64 mm deeper than the mouth
            (see the module-level ground truth in sc_controller.py).
          * ``sc_port_link`` is the port origin, another 2 mm off.

        If the entrance frames were not being published during a collection run,
        the loop fell through to those silently and every label in the run was
        systematically 15.64 mm too deep, with nothing in the output to say so
        beyond a ``tf_frame`` string nobody reads.  A dataset that is wrong by
        more than the entire insertion depth is worse than no dataset: it trains
        a model to aim at the back wall.  Fail the frame instead.
        """
        base = f"task_board/sc_port_{slot_idx}"
        return [
            f"{base}/sc_port_base/sc_port_base_link_entrance",
            f"{base}/sc_port_base_link_entrance",
            f"{base}/sc_port_link_entrance",
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
            # Camera had an image but no entrance-frame GT for either slot, so it
            # produced no label.  If this is nonzero for a whole run the entrance
            # frames are not being published and the run is worthless -- check it
            # before training on the output.
            "no_gt_label": 0,
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
            for slot in SC_SLOTS:
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
                # There used to be an HSV blue-blob fallback here that wrote
                # pseudo-labels from the colour detector whenever TF ground truth
                # was unavailable.  It is deliberately gone.
                #
                # Those corners are not a projected rectangle at all -- they are
                # whatever the blob's minimum-area box happened to be -- so any
                # frame produced that way followed a DIFFERENT convention from
                # the projected 8.8 x 6.0 mm labels, inside the same dataset,
                # tagged only by a "fallback_color_filter" string in the
                # per-sample JSON.  Two conventions mixed silently in one
                # training set is unrecoverable after the fact: you cannot tell
                # from the weights which frames poisoned them.
                #
                # If TF is unavailable the correct outcome is fewer frames, not
                # worse ones.  A run that collects nothing is a loud, fixable
                # problem; a run that collects garbage is a silent one.
                debug_counts["no_gt_label"] += 1
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
                f"no_gt_label={debug_counts['no_gt_label']} "
                f"vis_drop={debug_counts['kpt_visibility_drop']} "
                f"bbox_drop={debug_counts['bbox_drop']} "
                f"fmt_drop={debug_counts['format_drop']} "
                f"cam_missing={debug_counts['camera_missing']}"
            )
        # A frame that saw cameras but never resolved an entrance frame is the
        # signature of TF not publishing the mouth at all.  Say so once per
        # frame at error level: this used to be papered over by pseudo-labels,
        # so the failure mode has no history of being noticed.
        # Absent slots miss TF by design now that SC_SLOTS covers the whole board,
        # so key this on EVERY camera failing rather than on any tf_miss at all.
        if debug_counts["no_gt_label"] >= len(CAMERA_NAMES):
            self.get_logger().error(
                f"SC GT: all {debug_counts['no_gt_label']} camera(s) had images but no "
                "entrance-frame TF, so they were skipped rather than pseudo-labelled. "
                "Check that sc_port_base_link_entrance is being published -- until it "
                "is, this run collects nothing usable."
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
