import json
import os
import random
import time

import cv2
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException


# ─── Output Config ───────────────────────────────────────────────────────────

import shutil
from datetime import datetime

OUTPUT_DIR = os.path.expanduser(
    os.environ.get("AIC_SC_COLLECT_OUTPUT_DIR", "~/aic_perception_data/pose_sc")
)

def clear_pose_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Grab the parent folder (~/aic_perception_data) and put the backup right inside it!
    parent_dir = os.path.dirname(OUTPUT_DIR)
    old_pose_dir = os.path.join(parent_dir, f"pose_sc_old_{timestamp}")

    # 1. Check if 'pose' exists and actually has files inside it
    if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        # 2. Rename the current 'pose' folder to our new old folder name
        os.rename(OUTPUT_DIR, old_pose_dir)
        print(f"[INFO] Moved existing data to: {old_pose_dir}")

    # 3. Create a brand new, completely empty 'pose' folder for this run
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Ready to save new data into: {OUTPUT_DIR}")

# Call this function right before your script starts collecting data!
# USE THIS FORT TESTING DATA COLLECTOR 
clear_pose_folder()


# How many randomly sampled viewpoints per trial
VIEWPOINTS_PER_TRIAL = 15

# Val split: every Nth run goes entirely to val
# Ensures val images come from completely different board configs
VAL_EVERY_N_RUNS = 5

# Minimum bounding box side length (pixels) to include a label
MIN_BBOX_PX = 20

# Minimum number of keypoints with flag=2 (truly visible) to include a label.
MIN_VISIBLE_KEYPOINTS = 3

# Bounding-box padding — fraction added on EACH side of the raw keypoint extent.
BBOX_PADDING = 0.30


# ─── Monte Carlo Pose Space ───────────────────────────────────────────────────

# Bin definitions: (dz_min, dz_max, fraction_of_viewpoints)
DISTANCE_BINS = [
    (-0.01, 0.04, 0.2),   # Very close:  port fills 30-50 % of frame
    ( 0.04, 0.09, 0.4),   # Mid-close:   port fills 15-30 %
    ( 0.09, 0.14, 0.3),   # Mid-far:     port fills  5-15 %
    ( 0.14, 0.18, 0.1),   # Far:         full board visible
]

LATERAL_SCALE_NEAR = 0.02   # m at dz=0
LATERAL_SCALE_FAR  = 0.08   # m at dz=0.18
DZ_RANGE = (DISTANCE_BINS[0][0], DISTANCE_BINS[-1][1])

ROT_RANGE = {
    "rx": (-0.20, 0.20),
    "ry": (-0.20, 0.20),
    "rz": (-0.12, 0.12),
}

CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]

# Debug: draw projected TF frame axes on the saved images.
# This helps diagnose TF↔image timestamp mismatch and frame orientation issues.
DEBUG_DRAW_PORT_FRAMES = False
DEBUG_AXIS_LEN_M = 0.02  # 2 cm in the port frame

# Heuristic occlusion handling (no depth/segmentation available in Observation).
# If a projected keypoint lands on a very dark pixel (often robot arm / deep shadow),
# downgrade visibility from 2→1. If too few keypoints remain visible, the label is dropped.
OCCLUSION_DARK_PIXEL_THRESH = 35  # 0-255 grayscale
OCCLUSION_DARK_BORDER = 0         # px: ignore samples too close to image border


# ─── YOLO-Pose Keypoint Geometry (SC receptacle opening) ─────────────────────
#
# 4 corners in the resolved SC port entrance link frame (opening plane z=0).
# Local x/y:
#   KP0 (+W,+H) ─── KP1 (-W,+H)
#     │                   │
#   KP3 (+W,-H) ─── KP2 (-W,-H)

# Measured opening extents (meters): width × length → halves for ±XY corners at entrance plane z=0.
SC_FULL_WIDTH_M = 0.02578
SC_FULL_LENGTH_M = 0.00927
SC_HALF_WIDTH_M = SC_FULL_WIDTH_M * 0.5   # 12.890 mm
SC_HALF_HEIGHT_M = SC_FULL_LENGTH_M * 0.5  # 4.635 mm

LOCAL_SC_PORT_KPS = np.array(
    [
        [SC_HALF_WIDTH_M, SC_HALF_HEIGHT_M, 0.0],
        [-SC_HALF_WIDTH_M, SC_HALF_HEIGHT_M, 0.0],
        [-SC_HALF_WIDTH_M, -SC_HALF_HEIGHT_M, 0.0],
        [SC_HALF_WIDTH_M, -SC_HALF_HEIGHT_M, 0.0],
    ],
    dtype=np.float32,
)

NUM_KEYPOINTS = 4

# Horizontal flip: mirror corners L↔R within the quad.
FLIP_IDX = [1, 0, 3, 2]

CLASS_NAMES = ["sc_port"]

SC_SLOT_INDICES = (0, 1)


def candidate_sc_port_frames(slot_idx: int) -> list[str]:
    """TF frame names to try for task_board/sc_port_{slot}; first resolve wins."""
    base = f"task_board/sc_port_{slot_idx}"
    return [
        f"{base}/sc_port_base/sc_port_base_link_entrance",
        f"{base}/sc_port_base_link_entrance",
        f"{base}/sc_port_base/sc_port_base_link",
        f"{base}/sc_port_link_entrance",
        f"{base}/sc_port_link",
    ]

# ─── Math Helpers ────────────────────────────────────────────────────────────

def tf_to_4x4(tf_msg) -> np.ndarray:
    """Convert a ROS Transform (or TransformStamped) message to a 4×4 homogeneous matrix."""
    # Dig into the stamped message to get the actual transform data
    if hasattr(tf_msg, 'transform'):
        tf_msg = tf_msg.transform
        
    t = tf_msg.translation
    q = tf_msg.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [t.x, t.y, t.z]
    
    return T


def euler_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """ZYX Euler angles → quaternion (x, y, z, w)."""
    cx, cy, cz = np.cos([rx/2, ry/2, rz/2])
    sx, sy, sz = np.sin([rx/2, ry/2, rz/2])
    return np.array([
        sx*cy*cz - cx*sy*sz,
        cx*sy*cz + sx*cy*sz,
        cx*cy*sz - sx*sy*cz,
        cx*cy*cz + sx*sy*sz,
    ])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def ros_image_to_cv2(img_msg) -> np.ndarray | None:
    """Convert a ROS Image message to a BGR OpenCV array."""
    enc = img_msg.encoding
    try:
        arr = np.frombuffer(img_msg.data, dtype=np.uint8)
        if enc == "mono8":
            arr = arr.reshape(img_msg.height, img_msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        arr = arr.reshape(img_msg.height, img_msg.width, 3)
        if enc == "bgr8":
            return arr.copy()
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)   # rgb8 and fallback
    except Exception:
        return None

# ─── Keypoint Projection ─────────────────────────────────────────────────────

def project_keypoints(
    kps_3d:    np.ndarray,   # (N, 3)  object-frame 3-D points
    T_obj_cam: np.ndarray,   # (4, 4)  object → camera transform
    K:         np.ndarray,   # (3, 3)  camera intrinsic matrix
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3-D keypoints into image space WITHOUT clamping.

    Returns
    -------
    pts_2d : (N, 2)  float  raw pixel coords (may be outside image bounds)
    in_front : (N,)  bool   True when the point is in front of the camera (z > 0)
    """
    ones   = np.ones((kps_3d.shape[0], 1))
    pts_h  = np.hstack([kps_3d, ones])                    # (N, 4)
    pts_c  = (T_obj_cam @ pts_h.T).T[:, :3]              # (N, 3)  camera frame
    in_front = pts_c[:, 2] > 0.01

    # For behind-camera points use a large dummy depth so the division is safe;
    # the in_front mask will discard them from every meaningful calculation.
    safe_z = np.where(in_front, pts_c[:, 2], 1.0)
    pts_2d_h = (K @ pts_c.T).T                            # (N, 3)
    pts_2d   = pts_2d_h[:, :2] / safe_z[:, None]         # (N, 2)

    return pts_2d, in_front


def compute_visibility_flags(
    pts_2d:   np.ndarray,   # (N, 2)  raw (unclamped) pixel coords
    in_front: np.ndarray,   # (N,)    bool
    img_w:    int,
    img_h:    int,
) -> np.ndarray:
    """
    Assign YOLO-Pose visibility flags.

    Flag values
    -----------
    0  Point behind camera — genuinely unlabeled, YOLO ignores this in all losses.
    1  Point in front of camera but outside image bounds (off-screen / truncated).
       YOLO uses for regression loss but not keypoint visibility loss.
    2  Point in front of camera AND within image bounds — fully visible.
       YOLO uses for all losses.
    """
    flags = np.zeros(len(pts_2d), dtype=np.int32)
    for i, (pt, front) in enumerate(zip(pts_2d, in_front)):
        if not front:
            flags[i] = 0   # behind camera
        elif 0 <= pt[0] < img_w and 0 <= pt[1] < img_h:
            flags[i] = 2   # visible
        else:
            flags[i] = 1   # off-screen but in-front
    return flags


def clamp_keypoints_to_image(
    pts_2d: np.ndarray,   # (N, 2)  raw pixel coords
    flags:  np.ndarray,   # (N,)    visibility flags
    img_w:  int,
    img_h:  int,
) -> np.ndarray:
    """
    Clamp keypoint pixel coordinates to image bounds for writing to the label file.

    Points with flag=0 (behind camera) are zeroed — YOLO ignores their coordinates.
    Points with flag=1 (off-screen) are clamped to the nearest edge pixel.
    Points with flag=2 are already in-bounds (no-op clamp).
    """
    clamped = pts_2d.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0.0, img_w - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0.0, img_h - 1)
    clamped[flags == 0] = 0.0   # behind-camera: zero out coords
    return clamped


def downgrade_visibility_on_dark_pixels(
    gray: np.ndarray,        # (H, W) uint8
    pts_2d: np.ndarray,      # (N, 2) float
    flags: np.ndarray,       # (N,) int visibility flags (modified in-place)
    dark_thresh: int = OCCLUSION_DARK_PIXEL_THRESH,
    border: int = OCCLUSION_DARK_BORDER,
) -> None:
    """
    Heuristic: downgrade keypoints that land on very dark pixels.

    Approximates occlusion without depth/masks. Only points already marked
    as visible (flag=2) are candidates for downgrade.
    """
    h, w = gray.shape[:2]
    for i, (pt, vis) in enumerate(zip(pts_2d, flags)):
        if vis != 2:
            continue
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if x < border or y < border or x >= (w - border) or y >= (h - border):
            continue
        if int(gray[y, x]) <= dark_thresh:
            flags[i] = 1


def compute_padded_bbox(
    pts_2d:   np.ndarray,   # (N, 2)  raw (unclamped) pixel coords
    in_front: np.ndarray,   # (N,)    bool
    img_w:    int,
    img_h:    int,
    padding:  float = BBOX_PADDING,
) -> tuple[float, float, float, float] | None:
    """
    Compute a padded bounding box that encloses all in-front keypoints.

    Padding is added BEFORE clipping to image bounds so the box includes
    some context around the small SC opening.
    Uses raw (unclamped) projections so off-screen points pull the box
    out to the true object extent.

    Returns
    -------
    (x_min, y_min, x_max, y_max)  in pixels, clipped to image bounds,
    or None if no in-front points exist or the resulting box is too small.
    """
    if not np.any(in_front):
        return None

    pts = pts_2d[in_front]
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    bw = x_max - x_min
    bh = y_max - y_min

    # Add symmetric padding BEFORE clipping
    x_min -= bw * padding
    x_max += bw * padding
    y_min -= bh * padding
    y_max += bh * padding

    # Now clip to image bounds
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(float(img_w - 1), x_max)
    y_max = min(float(img_h - 1), y_max)

    if (x_max - x_min) < MIN_BBOX_PX or (y_max - y_min) < MIN_BBOX_PX:
        return None

    return x_min, y_min, x_max, y_max


def format_yolo_pose_label(
    bbox:       tuple[float, float, float, float],   # (x_min, y_min, x_max, y_max) px
    kps_clamped: np.ndarray,                          # (N, 2)  clamped pixel coords
    flags:      np.ndarray,                           # (N,)    visibility flags
    img_w:      int,
    img_h:      int,
    class_id:   int = 0,
) -> str | None:
    """
    Serialise one detection into the YOLO-Pose label format:

        <class_id> <cx> <cy> <w> <h>  <px0> <py0> <v0> ... <px{N-1}> <py{N-1}> <v{N-1}>

    All spatial values are normalised to [0, 1].  Visibility flags are integers.
    """
    if kps_clamped.shape[0] != NUM_KEYPOINTS or len(flags) != NUM_KEYPOINTS:
        return None

    x_min, y_min, x_max, y_max = bbox
    cx = ((x_min + x_max) / 2.0) / img_w
    cy = ((y_min + y_max) / 2.0) / img_h
    nw = (x_max - x_min) / img_w
    nh = (y_max - y_min) / img_h

    kp_tokens = []
    for pt, v in zip(kps_clamped, flags):
        px = pt[0] / img_w
        py = pt[1] / img_h
        kp_tokens.append(f"{px:.6f} {py:.6f} {int(v)}")

    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kp_tokens)


# ─── Monte Carlo Sampling ────────────────────────────────────────────────────

def sample_viewpoints(n: int) -> list[tuple[float, ...]]:
    """
    Stratified Monte Carlo sampling across distance bins.
    Returns a shuffled list of (dx, dy, dz, drx, dry, drz) tuples.
    """
    viewpoints = []
    remaining  = n

    bin_counts = []
    for i, (dz_min, dz_max, frac) in enumerate(DISTANCE_BINS):
        count = remaining if i == len(DISTANCE_BINS) - 1 else max(1, round(n * frac))
        remaining -= count
        bin_counts.append(count)

    for (dz_min, dz_max, _), count in zip(DISTANCE_BINS, bin_counts):
        for _ in range(count):
            dz = random.uniform(dz_min, dz_max)
            t  = (dz - DZ_RANGE[0]) / (DZ_RANGE[1] - DZ_RANGE[0])
            t  = max(0.0, min(1.0, t))
            lat = LATERAL_SCALE_NEAR + t * (LATERAL_SCALE_FAR - LATERAL_SCALE_NEAR)

            dx  = random.uniform(-lat,  lat)
            dy  = random.uniform(-lat,  lat)
            drx = random.uniform(*ROT_RANGE["rx"])
            dry = random.uniform(*ROT_RANGE["ry"])
            drz = random.uniform(*ROT_RANGE["rz"])

            viewpoints.append((dx, dy, dz, drx, dry, drz))

    random.shuffle(viewpoints)
    return viewpoints


def compute_settle_time(dx: float, dy: float, dz: float) -> float:
    """Estimate settle time (seconds) based on move distance in the slow sim."""
    return max(2.0, np.sqrt(dx**2 + dy**2 + dz**2) * 15.0)

# ─── The Policy ──────────────────────────────────────────────────────────────

class DataCollectorPoseSC(Policy):
    """
    Monte Carlo perception data collector for YOLO-Pose SC port detection.

    Produces labels with one class ``sc_port`` and 4 corners per detected slot
    (task_board/sc_port_0 and sc_port_1). Absent rails / missing TF omit that line.

    Uses the same Monte Carlo viewpoint sampling as the SFP collector policy.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        self._frame_counter = 0
        self._trial_counter = 0
        self._run_counter   = self._load_run_counter()

        for subdir in ["images/train", "images/val",
                       "labels/train", "labels/val", "metadata"]:
            os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

        self._split = "val" if (self._run_counter % VAL_EVERY_N_RUNS == 0) else "train"

        self.get_logger().info(
            f"DataCollectorPoseSC init | run={self._run_counter} "
            f"split={self._split} | {VIEWPOINTS_PER_TRIAL} viewpoints/trial | "
            f"YOLO-Pose: class={CLASS_NAMES[0]} kpts={NUM_KEYPOINTS}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_run_counter(self) -> int:
        """Persist run counter across restarts so the val-split stays consistent."""
        counter_file = os.path.join(OUTPUT_DIR, ".run_counter")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            with open(counter_file) as f:
                count = int(f.read().strip()) + 1
        except Exception:
            count = 1
        with open(counter_file, "w") as f:
            f.write(str(count))
        return count

    def _lookup_tf(self, target: str, source: str):
        try:
            return self._parent_node._tf_buffer.lookup_transform(
                target, source, Time(), Duration(seconds=0.05))
        except TransformException:
            return None

    def _lookup_tf_at_stamp(self, target: str, source: str, stamp: Time | None):
        """
        Lookup TF at a specific timestamp.

        Using the image timestamp here is critical; using "latest" (Time())
        can produce large intermittent projection errors when TF and images
        are not perfectly time-aligned.
        """
        try:
            tf_time = stamp if stamp is not None else Time()
            return self._parent_node._tf_buffer.lookup_transform(
                target, source, tf_time, Duration(seconds=0.20)
            )
        except TransformException:
            return None

    def _get_camera_intrinsics(self, obs, camera_name: str) -> np.ndarray | None:
        info_map = {
            "left_camera":   obs.left_camera_info,
            "center_camera": obs.center_camera_info,
            "right_camera":  obs.right_camera_info,
        }
        info = info_map.get(camera_name)
        if info is None:
            return None
        K = np.array(info.k).reshape(3, 3)
        return K if K[0, 0] != 0 else None

    def _get_camera_image(self, obs, camera_name: str) -> tuple[np.ndarray, Time] | None:
        img_map = {
            "left_camera":   obs.left_image,
            "center_camera": obs.center_image,
            "right_camera":  obs.right_image,
        }
        msg = img_map.get(camera_name)
        if msg is None:
            return None
        img = ros_image_to_cv2(msg)
        if img is None:
            return None
        # Best-effort: use the image message timestamp for TF.
        stamp = Time()
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            try:
                stamp = Time.from_msg(msg.header.stamp)
            except Exception:
                stamp = Time()
        return img, stamp

    def _draw_projected_axes(
        self,
        image: np.ndarray,
        K: np.ndarray,
        T_obj_cam: np.ndarray,
        label: str,
        axis_len_m: float = DEBUG_AXIS_LEN_M,
    ) -> None:
        """Overlay projected XYZ axes for a frame (object→camera transform)."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_len_m, 0.0, 0.0],
                [0.0, axis_len_m, 0.0],
                [0.0, 0.0, axis_len_m],
            ],
            dtype=np.float32,
        )
        pts2d, in_front = project_keypoints(pts, T_obj_cam, K)
        if not np.all(in_front):
            return

        o = tuple(np.round(pts2d[0]).astype(int))
        x = tuple(np.round(pts2d[1]).astype(int))
        y = tuple(np.round(pts2d[2]).astype(int))
        z = tuple(np.round(pts2d[3]).astype(int))

        # X=red, Y=green, Z=blue (BGR in OpenCV)
        cv2.line(image, o, x, (0, 0, 255), 2)
        cv2.line(image, o, y, (0, 255, 0), 2)
        cv2.line(image, o, z, (255, 0, 0), 2)
        cv2.putText(
            image,
            label,
            (o[0] + 4, o[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ── Core capture logic ────────────────────────────────────────────────────

    def _capture_frame(
        self,
        obs,
        vp_idx:    int,
        vp_params: tuple[float, ...],
    ) -> int:
        """
        For every camera: project SC port corners per slot, compute padded bbox,
        assign visibility flags, and write YOLO-Pose label files (one line per port).

        Returns the number of images saved in this frame.
        """
        frame_id = (
            f"{self._run_counter:04d}_"
            f"{self._trial_counter:02d}_"
            f"{self._frame_counter:03d}"
        )
        dx, dy, dz, drx, dry, drz = vp_params
        metadata: dict = {
            "frame_id":         frame_id,
            "trial":            self._trial_counter,
            "run":              self._run_counter,
            "split":            self._split,
            "viewpoint_idx":    vp_idx,
            "viewpoint_offset": dict(dx=dx, dy=dy, dz=dz, drx=drx, dry=dry, drz=drz),
            "cameras":          {},
        }
        saved = 0

        for cam_name in CAMERA_NAMES:
            img_and_stamp = self._get_camera_image(obs, cam_name)
            if img_and_stamp is None:
                continue
            image, img_stamp = img_and_stamp
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            K = self._get_camera_intrinsics(obs, cam_name)
            if K is None:
                continue

            img_h, img_w = image.shape[:2]
            cam_frame = f"{cam_name}/optical"

            labels: list[str] = []
            ports_meta: list[dict] = []

            # ── One label line per SC slot with valid TF ─────────────────────────
            for slot_idx in SC_SLOT_INDICES:
                tf_msg = None
                frame_used = None
                for src in candidate_sc_port_frames(slot_idx):
                    tf_msg = self._lookup_tf_at_stamp(cam_frame, src, img_stamp)
                    if tf_msg is not None:
                        frame_used = src
                        break
                if tf_msg is None:
                    continue

                T_cam_entrance = tf_to_4x4(tf_msg)

                if DEBUG_DRAW_PORT_FRAMES:
                    self._draw_projected_axes(
                        image,
                        K,
                        T_cam_entrance,
                        f"sc_port_{slot_idx}",
                    )

                kps2d_all, in_front_all = project_keypoints(LOCAL_SC_PORT_KPS, T_cam_entrance, K)
                flags = compute_visibility_flags(kps2d_all, in_front_all, img_w, img_h)
                downgrade_visibility_on_dark_pixels(gray, kps2d_all, flags)

                visible_count = int(np.sum(flags == 2))
                if visible_count < MIN_VISIBLE_KEYPOINTS:
                    continue

                bbox = compute_padded_bbox(
                    kps2d_all, in_front_all, img_w, img_h, BBOX_PADDING
                )
                if bbox is None:
                    continue
                box_min_x, box_min_y, box_max_x, box_max_y = bbox

                kps_clamped = clamp_keypoints_to_image(kps2d_all, flags, img_w, img_h)

                label_line = format_yolo_pose_label(
                    bbox, kps_clamped, flags, img_w, img_h, class_id=0
                )
                if label_line is None:
                    continue
                labels.append(label_line)

                ports_meta.append(
                    {
                        "sc_slot": int(slot_idx),
                        "tf_frame": frame_used,
                        "class_id": 0,
                        "bbox": [
                            float(box_min_x),
                            float(box_min_y),
                            float(box_max_x),
                            float(box_max_y),
                        ],
                        "keypoints_visible": visible_count,
                    }
                )

            if not labels:
                continue   # No SC ports visible in this camera — skip image

            # ── Write image and label ────────────────────────────────────────
            img_fn = f"{frame_id}_{cam_name}.png"
            lbl_fn = f"{frame_id}_{cam_name}.txt"

            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "images", self._split, img_fn), image)
            with open(
                os.path.join(OUTPUT_DIR, "labels", self._split, lbl_fn), "w"
            ) as f:
                f.write("\n".join(labels))

            metadata["cameras"][cam_name] = {
                "image": img_fn,
                "labels": len(labels),
                "sc_ports": ports_meta,
            }
            saved += 1

        # ── Write metadata JSON ──────────────────────────────────────────────
        meta_path = os.path.join(OUTPUT_DIR, "metadata", f"{frame_id}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._frame_counter += 1
        return saved

    # ── Robot movement ────────────────────────────────────────────────────────

    def _move_to_offset(
        self,
        move_robot,
        initial_tcp,
        dx: float, dy: float, dz: float,
        drx: float, dry: float, drz: float,
    ):
        """Move robot to initial_tcp + 6-DOF offset."""
        q      = initial_tcp.orientation
        base_q = np.array([q.x, q.y, q.z, q.w])

        if abs(drx) + abs(dry) + abs(drz) > 1e-6:
            delta_q = euler_to_quat(drx, dry, drz)
            new_q   = quat_multiply(base_q, delta_q)
            new_q  /= np.linalg.norm(new_q)
            orient  = Quaternion(x=new_q[0], y=new_q[1], z=new_q[2], w=new_q[3])
        else:
            orient = initial_tcp.orientation

        self.set_pose_target(
            move_robot=move_robot,
            pose=Pose(
                position=Point(
                    x=initial_tcp.position.x + dx,
                    y=initial_tcp.position.y + dy,
                    z=initial_tcp.position.z + dz,
                ),
                orientation=orient,
            ),
            stiffness=[100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
            damping=  [ 70.0,  70.0,  70.0, 35.0, 35.0, 35.0],
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def insert_cable(
        self,
        task:            Task,
        get_observation: GetObservationCallback,
        move_robot:      MoveRobotCallback,
        send_feedback:   SendFeedbackCallback,
    ):
        self._trial_counter += 1
        self.get_logger().info(
            f"Trial {self._trial_counter} | {task.plug_type} → {task.port_name} "
            f"on {task.target_module_name} | run={self._run_counter} "
            f"split={self._split}"
        )
        send_feedback(f"Trial {self._trial_counter} starting (run {self._run_counter})")

        # Wait for TF tree to stabilise after board spawn
        self.sleep_for(6.0)

        obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation at trial start — skipping")
            return True

        initial_tcp = obs.controller_state.tcp_pose

        # Sample viewpoints for this trial
        viewpoints = sample_viewpoints(VIEWPOINTS_PER_TRIAL)
        self.get_logger().info(
            f"Sampled {len(viewpoints)} Monte Carlo viewpoints "
            f"for trial {self._trial_counter}"
        )

        trial_saved = 0
        prev_dx, prev_dy, prev_dz = 0.0, 0.0, 0.0

        for vp_idx, (dx, dy, dz, drx, dry, drz) in enumerate(viewpoints):
            self._move_to_offset(
                move_robot, initial_tcp, dx, dy, dz, drx, dry, drz)

            # Settle time based on actual delta from previous viewpoint
            settle = compute_settle_time(
                dx - prev_dx, dy - prev_dy, dz - prev_dz)
            self.get_logger().info(
                f"  VP {vp_idx+1}/{len(viewpoints)} "
                f"dz={dz:.3f}m  settle={settle:.2f}s"
            )
            self.sleep_for(settle)
            prev_dx, prev_dy, prev_dz = dx, dy, dz

            obs = get_observation()
            if obs is not None:
                n = self._capture_frame(obs, vp_idx, (dx, dy, dz, drx, dry, drz))
                trial_saved += n
                self.get_logger().info(
                    f"  VP {vp_idx+1}: saved {n} imgs "
                    f"(trial total: {trial_saved})"
                )
            else:
                self.get_logger().warn(f"  VP {vp_idx+1}: no observation")

        # Return to start pose
        self._move_to_offset(move_robot, initial_tcp, 0, 0, 0, 0, 0, 0)
        self.sleep_for(2.0)

        train_n = len([
            f for f in os.listdir(os.path.join(OUTPUT_DIR, "images", "train"))
            if f.endswith(".png")
        ])
        val_n = len([
            f for f in os.listdir(os.path.join(OUTPUT_DIR, "images", "val"))
            if f.endswith(".png")
        ])

        self.get_logger().info(
            f"Trial {self._trial_counter} done | "
            f"saved {trial_saved} | train={train_n} val={val_n}"
        )
        send_feedback(
            f"Trial {self._trial_counter} done. {trial_saved} images saved. "
            f"train={train_n} val={val_n}"
        )

        self._write_yolo_config(train_n, val_n)
        return True

    # ── YAML config ───────────────────────────────────────────────────────────

    def _write_yolo_config(self, train_n: int, val_n: int):
        """
        Write the YOLO-Pose dataset YAML.

        Key additions vs. detection YAML:
          kpt_shape  — [N_keypoints, 3]  (x, y, visibility)
          flip_idx   — keypoint remapping for horizontal flip augmentation
        """
        flip_idx_str = str(FLIP_IDX)
        yaml_content = f"""# AIC SC port YOLO-Pose dataset (DataCollectorPoseSC)
# Single class ``sc_port``; 4 keypoints = opening quad in entrance frame projection.
# One label row per visible sc_port slot (task_board/sc_port_0 | sc_port_1).
#
# Val split: every {VAL_EVERY_N_RUNS}th run → val (distinct board configs from train)
# Train: {train_n} images | Val: {val_n} images

path: {OUTPUT_DIR}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}

kpt_shape: [{NUM_KEYPOINTS}, 3]

# Horizontal flip mirrors corners L↔R within each quad (see FLIP_IDX in source).
flip_idx: {flip_idx_str}
"""
        with open(os.path.join(OUTPUT_DIR, "aic_sc_ports_pose.yaml"), "w") as f:
            f.write(yaml_content)