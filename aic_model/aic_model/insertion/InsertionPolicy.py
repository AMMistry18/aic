#
# InsertionPolicy -- perception-guided SFP and SC insertion policy.
#
# The policy owns camera perception and dispatches to the deterministic,
# force-limited SFP and SC controllers. Learned last-inch control is disabled
# in the active deployment.
#
# Motion contract (per team decision):
#   * Perceive the target SFP port pose in base_link from the cameras.
#   * From WHEREVER the upstream macro handed the plug off, descend straight
#     down the perceived port insertion axis into the port.  No retract, no
#     "up then back down", no separate approach file.
#   * Object poses come only from perception + the robot's own TCP TF composed
#     with the fixed measured SFP-tip<-TCP transform (contract).
#
# Module path for aic_model:  -p policy:=aic_model.insertion.InsertionPolicy
#
# Key imports outside this file are:
#   * PerceptionCore -- the raw YOLO wrapper (best.pt / best_sc_mouth_pose.pt)
#   * contract -- shared calibration and port-frame geometry.
#
import itertools
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from aic_example_policies.ros.perception_core import PerceptionCore
from aic_model.policy import Policy
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from .visual_gap_recovery import VisualGapRecoveryMixin
from .sfp_controller import (
    configure_sfp_controller,
    prime_sfp_plug_pose,
    run_sfp_insertion,
    sfp_tcp_pose_for_tip,
    sfp_tip_from_tcp,
)
from .contract import (
    SFP_TIP_IN_TCP_POS,
    SFP_TIP_IN_TCP_QUAT,
    port_frame,
)
# Weighted DLT triangulation, reused as-is for the confidence-weighted port
# corners below -- it is pure numpy/ROS-free, already audited for the SFP
# plug-pose path, and must not be forked.
from .sfp_plug_pose import triangulate_dlt

# ----------------------------- configuration -------------------------------
# Perception weights.  best.pt = SFP YOLO-pose keypoints (the one loaded "in the
# form"); best_sc_mouth_pose.pt = physical SC front-mouth pose.
# Default to the weights dir that ships next to PerceptionCore -- the base image
# already bakes both deployed pose weights there, so this resolves correctly both
# in the repo and inside the installed site-packages layout.
import aic_example_policies.ros.perception_core as _pc_mod
_WEIGHTS_DIR = Path(
    os.environ.get(
        "AIC_PERCEPTION_WEIGHTS_DIR",
        str(Path(_pc_mod.__file__).resolve().parent / "weights"),
    )
)
NIC_WEIGHTS = os.environ.get("AIC_NIC_WEIGHTS", str(_WEIGHTS_DIR / "best.pt"))
SC_WEIGHTS = os.environ.get(
    "AIC_SC_POSE_WEIGHTS", str(_WEIGHTS_DIR / "best_sc_mouth_pose.pt")
)

CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]
MAX_PORT_REPROJ_PX = float(os.environ.get("RL_INSERT_MAX_REPROJ_PX", "25.0"))
# Perception robustness (2026-07-12): a single-frame perceive occasionally locks
# onto the WRONG NIC/port (~40 mm off) with clean reproj, and wanders ~1 mm
# frame-to-frame. Sample several frames and require a MEDIAN cluster to agree
# before trusting the pose; separately reject any candidate implausibly far from
# the plug tip (the macro hands off ~9 mm out, so a 40 mm pick is a wrong port).
PERCEPT_SAMPLES = int(os.environ.get("RL_INSERT_PERCEPT_SAMPLES", "7"))
PERCEPT_MIN_AGREE = int(os.environ.get("RL_INSERT_PERCEPT_MIN_AGREE", "3"))
PERCEPT_AGREE_TOL_M = float(os.environ.get("RL_INSERT_PERCEPT_AGREE_TOL_M", "0.004"))
PERCEPT_SAMPLE_DT = float(os.environ.get("RL_INSERT_PERCEPT_SAMPLE_DT", "0.15"))
MAX_HANDOFF_SELECT_M = float(os.environ.get("RL_INSERT_MAX_HANDOFF_SELECT_M", "0.020"))
# Selection-time reproj gate (2026-07-24): the multiview product also builds
# cross-matched ghost candidates (e.g. plug-body picks paired across cameras)
# that triangulate NEAR the tip with ~50px+ residuals. Nearest-tip selection
# would prefer such a ghost over a sub-1px true port that sits farther out, and
# the consensus 25px gate then rejects every frame. Gate candidates on reproj
# BEFORE the nearest-tip pick; the true port runs well under 1px.
MAX_SELECT_REPROJ_PX = float(os.environ.get("RL_INSERT_MAX_SELECT_REPROJ_PX", "5.0"))

# Confidence-weighted port triangulation (2026-07-25): detect_nic's per-keypoint
# confidence lets a per-corner DLT drop an occluded/unreliable camera-corner
# pair instead of triangulating through it, and the known port rectangle is
# then fit rigidly to whichever corners survive rather than averaged/measured
# with the flat midpoint math. Kill switch below restores the exact prior
# behavior when unset.
PORT_KP_CONF_MIN = float(os.environ.get("RL_INSERT_PORT_KP_CONF_MIN", "0.2"))
PORT_RIGID_FIT_ENABLE = os.environ.get(
    "RL_INSERT_SFP_PORT_RIGID_FIT", "1").strip().lower() in ("1", "true", "yes")

# This is the controller budget for both SFP and SC. Keep it strictly below
# the engine's per-task time limit so the controller can hold and return before
# the engine hard-cuts a run. The current pair is 720/780 seconds, leaving 60
# seconds for perception and approach before insertion starts.
ACTION_TIME_BUDGET_S = float(
    os.environ.get("RL_INSERT_ACTION_TIME_BUDGET_S", "720.0")
)
# Restore the bounded visual board-framing pass from b269872.  It is opt-in for
# normal insertion tasks; the reserved task ID runs only the board search.
BOARD_SEARCH = os.environ.get("RL_INSERT_BOARD_SEARCH", "0") == "1"
BOARD_SEARCH_ONLY_TASK_ID = os.environ.get(
    "RL_INSERT_BOARD_SEARCH_ONLY_TASK_ID", "board_search_only"
).strip()

# The upstream macro should hand off within the last-inch envelope.
HANDOFF_MAX_DIST = float(os.environ.get("RL_INSERT_HANDOFF_MAX_DIST", "0.12"))  # m

# One-shot grasp calibration dump: when set, at handoff log the TCP pose and any
# available ground-truth plug/tip TF frames (all in base_link), so the true
# tip-relative-to-TCP transform can be solved offline and SFP_TIP_IN_TCP_QUAT
# recalibrated to the actual grasp (not the wedge-inferred guess). Logs only;
# does not change control.
CALIB_DUMP = os.environ.get("RL_INSERT_CALIB_DUMP", "0").strip().lower() in ("1", "true", "yes")
# Candidate TF frame names the sim MIGHT publish for the true plug/tip pose. We
# probe each; whichever resolves gives ground truth. Extend via env (comma-sep).
CALIB_PLUG_FRAMES = [
    f.strip() for f in os.environ.get(
        "RL_INSERT_CALIB_PLUG_FRAMES",
        "sfp_tip,sfp_plug,plug,cable_0,sfp,gripper/sfp_tip,gripper/plug,tool/tip",
    ).split(",") if f.strip()
]
# SFP corner keypoints in the port entrance frame; matches YOLO-pose kp order.
LOCAL_SFP_PORT_KPS = np.array([
    [0.00685, 0.0043, 0.0],    # KP0: top-left
    [-0.00685, 0.0043, 0.0],   # KP1: top-right
    [-0.00685, -0.0043, 0.0],  # KP2: bottom-right
    [0.00685, -0.0043, 0.0],   # KP3: bottom-left
], dtype=np.float64)
# ---------------------------------------------------------------------------


# --------------------------- small math helpers ----------------------------
def _q_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-12
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_axis_angle(R):
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= (2.0 * np.sin(angle) + 1e-12)
    return axis * angle


def _axis_angle_to_R(v):
    """Convert an axis-angle vector in radians to a rotation matrix."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-10:
        return np.eye(3)
    k = np.asarray(v, dtype=np.float64) / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotmat_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        q = np.array([w, x, y, z])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        v = np.zeros(3)
        v[i] = s / 4.0
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = np.array([(R[k, j] - R[j, k]) / s, *v])
    return q / (np.linalg.norm(q) + 1e-12)


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return None if n < eps else v / n


def _weighted_kabsch_fit(local_pts, world_pts, weights):
    """Weighted Kabsch/SVD rigid fit mapping local_pts onto world_pts.

    Same algorithm as sfp_plug_pose.fit_rigid_transform, but that helper
    floors at >= 4 paired points (tuned for the 8-keypoint plug fit); the SFP
    port rectangle only ever has 3-4 corners, so this floors at 3 instead of
    forking the plug's stricter gate. Returns (rotation, translation) such
    that rotation @ local + translation ~= world.
    """
    source = np.asarray(local_pts, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("port rigid fit needs at least three paired 3D points")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) != len(source) or np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("port rigid-fit weights must be finite, positive, and match point count")
    w = w / np.sum(w)
    source_center = np.sum(source * w[:, None], axis=0)
    target_center = np.sum(target * w[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * w[:, None]).T @ target_zero
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _ros_image_to_cv2(img_msg):
    arr = np.frombuffer(img_msg.data, dtype=np.uint8)
    if img_msg.encoding == "mono8":
        return cv2.cvtColor(arr.reshape(img_msg.height, img_msg.width), cv2.COLOR_GRAY2BGR)
    arr = arr.reshape(img_msg.height, img_msg.width, 3)
    return arr.copy() if img_msg.encoding == "bgr8" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _tf_to_4x4(tf_msg):
    if hasattr(tf_msg, "transform"):
        tf_msg = tf_msg.transform
    t, q = tf_msg.translation, tf_msg.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class InsertionPolicy(VisualGapRecoveryMixin, Policy):
    """Perception-guided, force-limited SFP and SC insertion policy."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        log = self.get_logger()
        self._task = None
        self._wrench_baseline = np.zeros(6, dtype=np.float64)
        self._last_port_quat_wxyz = None
        self._last_port_reproj_px = None

        # ---- perception weights (SFP + physical SC mouth) ----
        sc_weights = SC_WEIGHTS if os.path.exists(SC_WEIGHTS) else None
        if not os.path.exists(NIC_WEIGHTS):
            raise FileNotFoundError(
                f"[rl] SFP YOLO weights not found at {NIC_WEIGHTS}; set AIC_NIC_WEIGHTS")
        log.info(f"[rl] loading SFP YOLO-pose weights: {NIC_WEIGHTS}")
        if sc_weights is None:
            log.warn(f"[rl] SC pose weights absent at {SC_WEIGHTS} (SFP unaffected)")
        else:
            log.info(f"[rl] SC pose weights available: {sc_weights}")
        self._pc = PerceptionCore(nic_weights=NIC_WEIGHTS, sc_weights=sc_weights)
        # SFP control requires a fresh, directly observed plug pose. Missing
        # plug weights fail lifecycle configuration instead of silently using
        # a fixed grasp transform.
        configure_sfp_controller(self)

    # ------------------------------------------------------------------ util
    def _lookup_transform(self, target_frame, source_frame, timeout_sec=0.2):
        return self._parent_node._tf_buffer.lookup_transform(
            target_frame, source_frame, Time(), Duration(seconds=timeout_sec))

    def _wait_for_transform(self, target_frame, source_frame, timeout_sec=8.0):
        deadline = time.monotonic() + timeout_sec
        last_error = None
        while time.monotonic() < deadline:
            try:
                return self._lookup_transform(target_frame, source_frame, timeout_sec=0.2)
            except TransformException as e:
                last_error = e
                self.sleep_for(0.1)
        if last_error is not None:
            raise last_error
        raise TransformException(f"Timed out waiting for {target_frame} <- {source_frame}")

    def _wait_for_stable_clock(self, timeout_sec=8.0, samples=4):
        deadline = time.monotonic() + timeout_sec
        last_ns = None
        stable = 0
        while time.monotonic() < deadline:
            now_ns = self._parent_node.get_clock().now().nanoseconds
            if now_ns <= 0:
                stable = 0
            elif last_ns is None or now_ns >= last_ns:
                stable += 1
                if stable >= samples:
                    return True
            else:
                stable = 0
            last_ns = now_ns
            time.sleep(0.1)
        self.get_logger().warn("[rl] timed out waiting for stable sim time; continuing")
        return False

    def _tcp(self):
        t = self._lookup_transform("base_link", "gripper/tcp")
        tr, ro = t.transform.translation, t.transform.rotation
        return (np.array([tr.x, tr.y, tr.z]), np.array([ro.w, ro.x, ro.y, ro.z]))  # wxyz

    def _tip_from_tcp(self, tcp_pos, tcp_quat):
        """Plug tip from the fresh per-run visual grasp transform."""
        return sfp_tip_from_tcp(self, tcp_pos, tcp_quat)

    def _dump_grasp_calibration(self):
        """One-shot: log everything needed to recalibrate SFP_TIP_IN_TCP_QUAT.

        Dumps (a) the TCP pose in base_link, (b) the tip pose the CURRENT
        (possibly wrong) transform produces, and (c) any ground-truth plug/tip TF
        frame that resolves -- all in base_link. From (a) + a true tip pose the
        exact tip-relative-to-TCP transform is R_tcp^T @ R_tip_true (right axis),
        which we then paste into contract. Logs only; no motion change.
        """
        log = self.get_logger()
        try:
            tcp_pos, tcp_quat = self._tcp()
        except Exception as ex:
            log.error(f"[calib] cannot read TCP: {ex}")
            return
        tip_pos, R_tip = self._tip_from_tcp(tcp_pos, tcp_quat)
        q_tip_assumed = _rotmat_to_quat_wxyz(R_tip)
        log.info("[calib] === GRASP CALIBRATION DUMP (base_link) ===")
        log.info(f"[calib] TCP pos={np.round(tcp_pos,6).tolist()} "
                 f"quat_wxyz={np.round(tcp_quat,6).tolist()}")
        log.info(f"[calib] ASSUMED tip (current transform) pos={np.round(tip_pos,6).tolist()} "
                 f"quat_wxyz={np.round(q_tip_assumed,6).tolist()}")
        log.info(f"[calib] current SFP_TIP_IN_TCP_QUAT={SFP_TIP_IN_TCP_QUAT.tolist()} "
                 f"POS={SFP_TIP_IN_TCP_POS.tolist()}")
        found_any = False
        for frame in CALIB_PLUG_FRAMES:
            try:
                tf = self._lookup_transform("base_link", frame, timeout_sec=0.3)
            except Exception:
                continue
            tr, ro = tf.transform.translation, tf.transform.rotation
            gt_pos = np.array([tr.x, tr.y, tr.z])
            gt_quat = np.array([ro.w, ro.x, ro.y, ro.z])  # wxyz
            # True tip-in-TCP transform: T_tcp^-1 @ T_true. Compute the rotation
            # part R_tcp^T @ R_true and the position part R_tcp^T @ (p_true-p_tcp).
            R_tcp = _q_to_R(*tcp_quat)
            R_true = _q_to_R(*gt_quat)
            R_rel = R_tcp.T @ R_true
            q_rel = _rotmat_to_quat_wxyz(R_rel)
            p_rel = R_tcp.T @ (gt_pos - tcp_pos)
            found_any = True
            log.info(f"[calib] GROUND-TRUTH frame '{frame}' RESOLVED: "
                     f"pos={np.round(gt_pos,6).tolist()} quat_wxyz={np.round(gt_quat,6).tolist()}")
            log.info(f"[calib]   >>> SOLVED SFP_TIP_IN_TCP_QUAT={np.round(q_rel,10).tolist()}")
            log.info(f"[calib]   >>> SOLVED SFP_TIP_IN_TCP_POS ={np.round(p_rel,10).tolist()}  "
                     f"(paste BOTH into contract.py if this frame is the true tip)")
        if not found_any:
            log.warn(f"[calib] no ground-truth plug/tip frame resolved from "
                     f"{CALIB_PLUG_FRAMES}. Set RL_INSERT_CALIB_PLUG_FRAMES to the "
                     f"correct frame name, or provide TCP + true-tip poses another way.")
        log.info("[calib] === END DUMP ===")

    def _tcp_target_for_tip(self, tip_pos, R_tip):
        tcp_pos, q_tcp = sfp_tcp_pose_for_tip(self, tip_pos, R_tip)
        return Pose(
            position=Point(x=float(tcp_pos[0]), y=float(tcp_pos[1]), z=float(tcp_pos[2])),
            orientation=Quaternion(
                w=float(q_tcp[0]), x=float(q_tcp[1]),
                y=float(q_tcp[2]), z=float(q_tcp[3])),
        )

    # --------------------------------------------------------- perception ---
    def _get_cam_data(self, obs, cam_name):
        img_map = {"left_camera": obs.left_image, "center_camera": obs.center_image,
                   "right_camera": obs.right_image}
        info_map = {"left_camera": obs.left_camera_info, "center_camera": obs.center_camera_info,
                    "right_camera": obs.right_camera_info}
        img_msg, info_msg = img_map.get(cam_name), info_map.get(cam_name)
        if img_msg is None or info_msg is None:
            return None
        K = np.array(info_msg.k).reshape(3, 3)
        if K[0, 0] == 0:
            return None
        try:
            bgr = _ros_image_to_cv2(img_msg)
        except Exception:
            return None
        return bgr, K

    def _lookup_cam_from_base(self, cam_name):
        try:
            tf = self._lookup_transform(f"{cam_name}/optical", "base_link")
        except TransformException as e:
            self.get_logger().warn(f"{cam_name}: TF lookup failed: {e}")
            return None
        return _tf_to_4x4(tf)

    def _build_views(self, obs):
        views = {}
        for cam in CAMERA_NAMES:
            d = self._get_cam_data(obs, cam)
            if d is None:
                continue
            bgr, K = d
            T = self._lookup_cam_from_base(cam)
            if T is None:
                continue
            views[cam] = (bgr, K, T)
        return views

    def _reproject_error_px(self, X, K, T_cam_from_base, uv):
        P = self._pc.build_projection_matrix(K, T_cam_from_base)
        x = P @ np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
        if x[2] <= 1e-6:
            return None
        uv_hat = np.array([x[0] / x[2], x[1] / x[2]], dtype=np.float64)
        return float(np.linalg.norm(uv_hat - np.array(uv, dtype=np.float64)))

    def _estimate_sfp_port_orientation(self, kp_3d):
        """Entrance-frame quat/yaw from the four triangulated SFP corners.

        In-plane +X = midpoint(KP0,KP3) - midpoint(KP1,KP2), projected to the
        board plane; insertion axis is world -Z (downward).
        """
        if kp_3d.shape != (4, 3):
            return None, None
        x_axis = ((kp_3d[0] + kp_3d[3]) * 0.5) - ((kp_3d[1] + kp_3d[2]) * 0.5)
        x_axis[2] = 0.0
        x_axis = _normalize(x_axis)
        if x_axis is None:
            return None, None
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        y_axis = _normalize(np.cross(z_axis, x_axis))
        if y_axis is None:
            return None, None
        x_axis = _normalize(np.cross(y_axis, z_axis))
        R_tip = np.column_stack([x_axis, y_axis, z_axis])
        yaw = float(np.arctan2(R_tip[1, 0], R_tip[0, 0]))
        return _rotmat_to_quat_wxyz(R_tip), yaw

    def _log_port_pose_input(self, cam, nics):
        """Per-camera port detection line, style-matched to SFP's plug-pose
        PLUG_POSE_INPUT log: one compact line per camera per sampled frame,
        using the top (highest box-confidence) detection.
        """
        if not nics:
            return
        top = nics[0]
        kp_conf = top.get("kp_conf")
        kp_conf = np.ones(8, dtype=np.float64) if kp_conf is None else np.asarray(kp_conf, dtype=np.float64)
        self.get_logger().info(
            f"[sfp] PORT_POSE_INPUT camera={cam} box_conf={top['conf']:.3f} "
            f"kp_conf_mean={float(np.mean(kp_conf)):.3f} kp_conf_min={float(np.min(kp_conf)):.3f}"
        )

    def _make_sfp_multiview_candidates(self, per_cam):
        cams = [c for c, cand in per_cam.items() if cand]
        if len(cams) < 2:
            return []
        for c in cams:
            per_cam[c] = per_cam[c][:5]

        candidates = []
        for picks in itertools.product(*[per_cam[c] for c in cams]):
            if not PORT_RIGID_FIT_ENABLE:
                kp_3d = []
                try:
                    for i in range(4):
                        pts_2d = [tuple(p["kps"][i]) for p in picks]
                        Ps = [p["P"] for p in picks]
                        kp_3d.append(self._pc.triangulate(pts_2d, Ps))
                except Exception:
                    continue
                kp_3d = np.array(kp_3d, dtype=np.float64)
                X = kp_3d.mean(axis=0)
                if X[2] < -0.05 or X[2] > 0.25:
                    continue

                q_wxyz, yaw = self._estimate_sfp_port_orientation(kp_3d)
                if q_wxyz is None:
                    continue

                width = np.linalg.norm(((kp_3d[0] + kp_3d[3]) * 0.5) - ((kp_3d[1] + kp_3d[2]) * 0.5))
                height = np.linalg.norm(((kp_3d[0] + kp_3d[1]) * 0.5) - ((kp_3d[2] + kp_3d[3]) * 0.5))
                if not (0.006 <= width <= 0.030 and 0.004 <= height <= 0.025):
                    continue

                errors = []
                for p in picks:
                    for i in range(4):
                        err = self._reproject_error_px(kp_3d[i], p["K"], p["T"], p["kps"][i])
                        if err is not None:
                            errors.append(err)
                if not errors:
                    continue
                reproj = float(np.mean(errors))
                shape_penalty = abs(width - 0.0137) * 250.0 + abs(height - 0.0086) * 250.0
                score = reproj + shape_penalty - 0.02 * float(np.mean([p.get("conf", 0.0) for p in picks]))
                candidates.append({
                    "X": X, "kp_3d": kp_3d, "q_wxyz": q_wxyz, "yaw": yaw,
                    "score": float(score), "reproj_px": reproj,
                    "width": float(width), "height": float(height), "port_slot": None,
                })
                continue

            # Confidence-weighted per-corner DLT: a view's corner contributes
            # only when its keypoint confidence clears PORT_KP_CONF_MIN, and a
            # corner still needs >= 2 surviving views to triangulate at all --
            # fewer than that is a failed/missing corner, same as today.
            tri_idx, tri_pts, tri_weight = [], [], []
            for i in range(4):
                pts_2d, Ps, weights = [], [], []
                for p in picks:
                    kp_conf = p.get("kp_conf")
                    conf = 1.0 if kp_conf is None else float(kp_conf[i])
                    if conf < PORT_KP_CONF_MIN:
                        continue
                    pts_2d.append(tuple(p["kps"][i]))
                    Ps.append(p["P"])
                    weights.append(conf)
                if len(pts_2d) < 2:
                    continue
                try:
                    tri_pts.append(triangulate_dlt(pts_2d, Ps, weights))
                except Exception:
                    continue
                tri_idx.append(i)
                tri_weight.append(float(np.sum(weights)))

            # < 3 usable corners is a failed candidate exactly like today (one
            # missing corner used to abort the whole 4-corner triangulation).
            if len(tri_idx) < 3:
                continue
            raw_pts = np.array(tri_pts, dtype=np.float64)
            X_raw = raw_pts.mean(axis=0)
            if X_raw[2] < -0.05 or X_raw[2] > 0.25:
                continue

            # Rigid rectangle fit (weighted Kabsch/SVD, weights = summed view
            # confidence per corner): recovers the true port center/in-plane
            # orientation even with a corner missing, unlike the raw
            # mean/midpoint math above which only works with all 4 present.
            try:
                R_fit, t_fit = _weighted_kabsch_fit(
                    LOCAL_SFP_PORT_KPS[tri_idx], raw_pts, tri_weight)
            except Exception:
                continue
            # _estimate_sfp_port_orientation orthonormalizes against world -Z
            # the same way regardless of input; feeding it the idealized
            # fitted rectangle (not the raw corners) is what makes the
            # reported orientation the FITTED one, per the insertion-axis
            # convention that function already enforces.
            fitted_kp_3d = (R_fit @ LOCAL_SFP_PORT_KPS.T).T + t_fit
            q_wxyz, yaw = self._estimate_sfp_port_orientation(fitted_kp_3d)
            if q_wxyz is None:
                continue
            X = t_fit

            # Shape gate stays on the RAW corners exactly as today; the
            # diagonal-midpoint formula needs all 4, so it is skipped (not
            # relaxed) when only 3 triangulated -- the reprojection gate below
            # still screens those candidates.
            width, height, shape_penalty = float("nan"), float("nan"), 0.0
            if len(tri_idx) == 4:
                width = np.linalg.norm(((raw_pts[0] + raw_pts[3]) * 0.5) - ((raw_pts[1] + raw_pts[2]) * 0.5))
                height = np.linalg.norm(((raw_pts[0] + raw_pts[1]) * 0.5) - ((raw_pts[2] + raw_pts[3]) * 0.5))
                if not (0.006 <= width <= 0.030 and 0.004 <= height <= 0.025):
                    continue
                shape_penalty = abs(width - 0.0137) * 250.0 + abs(height - 0.0086) * 250.0

            # Reprojection gate measures the RAW triangulated corners against
            # every view, same formula as today -- the 25px / 5px consensus
            # gates downstream keep reading this field with unchanged meaning.
            errors = []
            for p in picks:
                for pos, i in enumerate(tri_idx):
                    err = self._reproject_error_px(raw_pts[pos], p["K"], p["T"], p["kps"][i])
                    if err is not None:
                        errors.append(err)
            if not errors:
                continue
            reproj = float(np.mean(errors))
            score = reproj + shape_penalty - 0.02 * float(np.mean([p.get("conf", 0.0) for p in picks]))
            candidates.append({
                "X": X, "kp_3d": raw_pts, "q_wxyz": q_wxyz, "yaw": yaw,
                "score": float(score), "reproj_px": reproj,
                "width": float(width), "height": float(height), "port_slot": None,
            })

        candidates.sort(key=lambda c: c["score"])
        return candidates

    def _estimate_sfp_port_pose_single_view(self, kps_2d, K, T_cam_from_base, cam_name):
        """Fallback pose from one camera via PnP on the known SFP rectangle."""
        img_pts = np.asarray(kps_2d, dtype=np.float64).reshape(-1, 2)
        if img_pts.shape != (4, 2) or not np.all(np.isfinite(img_pts)):
            return None
        dist = np.zeros((5, 1), dtype=np.float64)
        flags = cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(LOCAL_SFP_PORT_KPS, img_pts, K.astype(np.float64), dist, flags=flags)
        if not ok and flags != cv2.SOLVEPNP_ITERATIVE:
            ok, rvec, tvec = cv2.solvePnP(
                LOCAL_SFP_PORT_KPS, img_pts, K.astype(np.float64), dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        reproj, _ = cv2.projectPoints(LOCAL_SFP_PORT_KPS, rvec, tvec, K.astype(np.float64), dist)
        reproj_error = float(np.mean(np.linalg.norm(reproj.reshape(-1, 2) - img_pts, axis=1)))
        if reproj_error > MAX_PORT_REPROJ_PX:
            return None
        R_cam_port, _ = cv2.Rodrigues(rvec)
        port_cam = tvec.reshape(3)
        if port_cam[2] <= 0.0:
            return None
        T_base_from_cam = self._pc.invert_transform(T_cam_from_base)
        X = (T_base_from_cam @ np.array([port_cam[0], port_cam[1], port_cam[2], 1.0]))[:3]
        R_base_port = T_base_from_cam[:3, :3] @ R_cam_port
        kp_3d = (R_base_port @ LOCAL_SFP_PORT_KPS.T).T + X
        q_wxyz, yaw = self._estimate_sfp_port_orientation(kp_3d)
        return X, kp_3d, q_wxyz, yaw, reproj_error

    def _extract_trailing_index(self, name, prefix):
        if not name or not name.startswith(prefix):
            return None
        try:
            return int(name[len(prefix):].split("_")[0])
        except (TypeError, ValueError):
            return None

    def _select_sfp_candidate(self, candidates, target_idx, label):
        """Pick the target port: nearest candidate to the current plug tip.

        The task board translates/yaws between trials and not every slot is
        populated, so we choose the perceived port physically closest to our own
        tip rather than trusting a fixed slot index.
        """
        if not candidates:
            return None
        # Reproj gate first: nearest-tip must only choose among geometrically
        # consistent candidates, or a cross-matched ghost near the tip wins.
        clean = [c for c in candidates if c["reproj_px"] <= MAX_SELECT_REPROJ_PX]
        if not clean:
            best = min(candidates, key=lambda c: c["reproj_px"])
            self.get_logger().warn(
                f"{label}: no candidate under {MAX_SELECT_REPROJ_PX:.1f}px select "
                f"gate (best reproj {best['reproj_px']:.1f}px) -- rejecting frame")
            return None
        candidates = clean
        try:
            tcp_pos, tcp_quat = self._tcp()
            tip_pos, _ = self._tip_from_tcp(tcp_pos, tcp_quat)
        except Exception:
            return candidates[0]
        # Distance gate: the macro hands off ~9 mm from the target mouth, so a
        # candidate far beyond that is a different (wrong) NIC/port picked up by
        # nearest-tip. Drop those before selecting so we don't commit to a
        # cleanly-detected-but-wrong port.
        in_range = [c for c in candidates
                    if float(np.linalg.norm(c["X"] - tip_pos)) <= MAX_HANDOFF_SELECT_M]
        if not in_range:
            nearest = min(candidates,
                          key=lambda c: float(np.linalg.norm(c["X"] - tip_pos)))
            self.get_logger().warn(
                f"{label}: all candidates beyond {MAX_HANDOFF_SELECT_M*1000:.0f}mm "
                f"handoff gate (nearest {np.linalg.norm(nearest['X']-tip_pos)*1000:.1f}mm) "
                "-- rejecting as wrong port")
            return None
        chosen = min(in_range, key=lambda c: float(np.linalg.norm(c["X"] - tip_pos)))
        self.get_logger().info(
            f"{label}: nearest-tip selected X={np.round(chosen['X'], 5).tolist()} "
            f"dist={np.linalg.norm(chosen['X'] - tip_pos) * 1000:.1f}mm "
            f"reproj={chosen['reproj_px']:.1f}px (requested_slot={target_idx})")
        return chosen

    def perceive_port_pose_consensus(self, task, get_observation):
        """Robust port pose: sample several frames and return the median of a
        cluster that agrees, rejecting single-frame wrong-port / noisy picks.

        Returns (port_pos(3), port_quat_wxyz(4), reproj_px) or None.
        """
        log = self.get_logger()
        samples = []  # list of (X, q_wxyz, reproj)
        for _ in range(PERCEPT_SAMPLES):
            obs = get_observation()
            if obs is not None:
                res = self.perceive_port_pose(task, obs)
                if res is not None:
                    X, q, reproj = res
                    if np.isfinite(reproj) and reproj <= MAX_PORT_REPROJ_PX:
                        samples.append((np.asarray(X, float),
                                        np.asarray(q, float), float(reproj)))
            self.sleep_for(PERCEPT_SAMPLE_DT)

        if len(samples) < PERCEPT_MIN_AGREE:
            log.error(f"[rl] perception consensus failed: only {len(samples)}/"
                      f"{PERCEPT_SAMPLES} frames passed reproj (need "
                      f"{PERCEPT_MIN_AGREE})")
            return None

        # Cluster around the median position (robust to a wrong-port outlier).
        positions = np.array([s[0] for s in samples])
        med = np.median(positions, axis=0)
        keep = [s for s in samples
                if float(np.linalg.norm(s[0] - med)) <= PERCEPT_AGREE_TOL_M]
        if len(keep) < PERCEPT_MIN_AGREE:
            spread = float(np.max(np.linalg.norm(positions - med, axis=1))) * 1000
            log.error(f"[rl] perception consensus failed: {len(keep)}/"
                      f"{len(samples)} frames agree within "
                      f"{PERCEPT_AGREE_TOL_M*1000:.1f}mm (spread={spread:.1f}mm) "
                      "-- unstable / conflicting port detections")
            return None

        kept_pos = np.array([s[0] for s in keep])
        port_pos = np.median(kept_pos, axis=0)
        kept_reproj = float(np.median([s[2] for s in keep]))
        # Quaternion from the kept sample closest to the median position (avoids
        # naive quaternion averaging / sign issues).
        best = min(keep, key=lambda s: float(np.linalg.norm(s[0] - port_pos)))
        port_quat = best[1]
        log.info(f"[rl] perception consensus: {len(keep)}/{len(samples)} agree, "
                 f"port={np.round(port_pos, 5).tolist()} reproj={kept_reproj:.2f}px")
        self._last_port_quat_wxyz = port_quat
        self._last_port_reproj_px = kept_reproj
        return port_pos, port_quat, kept_reproj

    def perceive_port_pose(self, task, obs):
        """Return (port_pos(3), port_quat_wxyz(4), reproj_px) or None (SFP)."""
        self._last_port_quat_wxyz = None
        self._last_port_reproj_px = None
        views = self._build_views(obs)
        if len(views) < 2:
            # Single-view PnP fallback so one usable camera still yields a pose.
            if len(views) == 1:
                cam, (bgr, K, T) = next(iter(views.items()))
                nics = self._pc.detect_nic(bgr, conf_thresh=0.2)
                self._log_port_pose_input(cam, nics)
                best = None
                for det in nics[:5]:
                    for slot in (0, 1):
                        kps = np.asarray(det["kps"])[slice(4 * slot, 4 * (slot + 1))]
                        if kps.shape != (4, 2):
                            continue
                        res = self._estimate_sfp_port_pose_single_view(kps, K, T, cam)
                        if res is None:
                            continue
                        X, _, q_wxyz, _, reproj = res
                        if q_wxyz is None:
                            continue
                        if best is None or reproj < best[2]:
                            best = (X, q_wxyz, reproj)
                if best is None:
                    self.get_logger().error("[rl] single-view SFP PnP produced no pose")
                    return None
                X, q_wxyz, reproj = best
                self._last_port_quat_wxyz = q_wxyz
                self._last_port_reproj_px = float(reproj)
                return np.asarray(X, dtype=np.float64), np.asarray(q_wxyz, dtype=np.float64), float(reproj)
            self.get_logger().error(f"[rl] only {len(views)} cam views usable")
            return None

        # Multiview: collect per-camera SFP keypoint detections for both slots.
        per_slot_cam = {0: {}, 1: {}}
        for cam, (bgr, K, T) in views.items():
            nics = self._pc.detect_nic(bgr, conf_thresh=0.2)
            self._log_port_pose_input(cam, nics)
            if not nics:
                self.get_logger().warn(f"{cam}: no NIC")
                continue
            for slot in (0, 1):
                kp_slice = slice(4 * slot, 4 * (slot + 1))
                dets = []
                for det in nics[:5]:
                    kps = np.asarray(det["kps"])[kp_slice]
                    if kps.shape != (4, 2):
                        continue
                    # kp_conf may be absent on old weights -- ones fallback
                    # keeps the confidence-weighted fit a no-op degrade.
                    kp_conf_full = det.get("kp_conf")
                    kp_conf_full = (
                        np.ones(8, dtype=np.float64) if kp_conf_full is None
                        else np.asarray(kp_conf_full, dtype=np.float64)
                    )
                    dets.append({
                        "kps": kps, "conf": det["conf"], "kp_conf": kp_conf_full[kp_slice],
                        "K": K, "T": T,
                        "P": self._pc.build_projection_matrix(K, T),
                    })
                if dets:
                    per_slot_cam[slot][cam] = dets

        candidates = []
        for slot, slot_cams in per_slot_cam.items():
            for cand in self._make_sfp_multiview_candidates(slot_cams):
                cand["port_slot"] = slot
                candidates.append(cand)
        candidates.sort(key=lambda c: c["score"])
        if not candidates:
            self.get_logger().warn("[rl] SFP multiview matching found no port candidates")
            return None

        target_idx = self._extract_trailing_index(task.target_module_name, "nic_card_mount_")
        chosen = self._select_sfp_candidate(candidates, target_idx, "SFP target")
        if chosen is None or chosen["q_wxyz"] is None:
            return None
        self._last_port_quat_wxyz = np.asarray(chosen["q_wxyz"], dtype=np.float64)
        self._last_port_reproj_px = float(chosen["reproj_px"])
        return (np.asarray(chosen["X"], dtype=np.float64),
                self._last_port_quat_wxyz, float(chosen["reproj_px"]))

    # ---------------------------------------------------- controller support
    @staticmethod
    def _wrench_vector(obs_msg):
        w = obs_msg.wrist_wrench.wrench
        return np.array([w.force.x, w.force.y, w.force.z,
                         w.torque.x, w.torque.y, w.torque.z], dtype=np.float64)

    # ----------------------------------------------------------------- main
    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback):
        log = self.get_logger()
        deadline_start_wall = time.monotonic()
        self._action_deadline_wall = deadline_start_wall + ACTION_TIME_BUDGET_S
        self._action_deadline_start_wall = deadline_start_wall
        log.info(
            f"[rl] ACTION_DEADLINE_START budget_s={ACTION_TIME_BUDGET_S:.1f} "
            "clock=wall"
        )
        try:
            return self._run(task, get_observation, move_robot, send_feedback)
        except TimeoutError as exc:
            log.error(f"[rl] INSERTION_ABORT reason=action_deadline detail={exc}")
            send_feedback("insert_cable timed out -- holding current pose")
            return False
        except Exception:
            log.error("[rl] insert_cable crashed:\n" + traceback.format_exc())
            send_feedback("insert_cable crashed -- see model log")
            return False

    def _enforce_action_deadline(self, move_robot):
        """Hold the TCP and stop insertion once its wall-clock budget expires."""
        deadline = getattr(self, "_action_deadline_wall", None)
        if deadline is None or time.monotonic() <= deadline:
            return
        now_wall = time.monotonic()
        start_wall = float(getattr(self, "_action_deadline_start_wall", deadline))
        task = getattr(self, "_task", None)
        plug_type = getattr(task, "plug_type", "unknown") if task is not None else "unknown"
        self.get_logger().error(
            f"[rl] ACTION_DEADLINE_EXCEEDED plug_type={plug_type} "
            f"elapsed_wall_s={now_wall - start_wall:.3f} "
            f"budget_s={ACTION_TIME_BUDGET_S:.3f} action=hold_current_tcp"
        )
        try:
            tcp_pos, tcp_quat = self._tcp()
            self.set_pose_target(
                move_robot,
                Pose(
                    position=Point(
                        x=float(tcp_pos[0]),
                        y=float(tcp_pos[1]),
                        z=float(tcp_pos[2]),
                    ),
                    orientation=Quaternion(
                        w=float(tcp_quat[0]),
                        x=float(tcp_quat[1]),
                        y=float(tcp_quat[2]),
                        z=float(tcp_quat[3]),
                    ),
                ),
                stiffness=[90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
                damping=[50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
            )
        finally:
            raise TimeoutError(
                f"insertion exceeded {ACTION_TIME_BUDGET_S:.1f}s wall-clock budget"
            )

    def _run(self, task, get_observation, move_robot, send_feedback):
        log = self.get_logger()
        self._task = task
        plug_type = (task.plug_type or "sfp").lower()
        log.info(f"[rl] task: cable={task.cable_name} plug={task.plug_name} "
                 f"({plug_type}) port={task.port_name} module={task.target_module_name}")
        if plug_type not in ("sfp", "sc"):
            log.error(f"[rl] unsupported plug type '{plug_type}' (expected sfp or sc)")
            return False

        self._wait_for_stable_clock()
        self._wait_for_transform("base_link", "gripper/tcp", timeout_sec=8.0)
        self.sleep_for(1.0)

        # --- wait for a first observation (cameras + robot state)
        obs = None
        t0 = time.monotonic()
        while obs is None and time.monotonic() - t0 < 10.0:
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.2)
        if obs is None:
            log.error("[rl] no Observation in 10 s -- zenoh wiring broken")
            return False

        # The Flowstate insert-cable skill exposes Task.id. Reserving one ID
        # provides a board-framing-only step without another ROS policy node.
        if task.id == BOARD_SEARCH_ONLY_TASK_ID:
            from .board_search import BoardSearch
            ok = BoardSearch(self).run(get_observation, move_robot)
            log.info(f"[board_search] board-only task complete: {ok}")
            return ok

        if BOARD_SEARCH:
            from .board_search import BoardSearch
            ok = BoardSearch(self).run(get_observation, move_robot)
            log.info(f"[board_search] whole board in view: {ok}")

        # SC branches to its own scripted controller: different bore depth,
        # different force ladder, and its own per-run measured grasp transform
        # (prime_sc_plug_pose inside run_sc_insertion). It must branch before
        # SFP-only plug priming below.
        if plug_type == "sc":
            from .sc_controller import run_sc_insertion
            return run_sc_insertion(self, task, get_observation, move_robot, send_feedback)

        # Prime direct SFP plug vision before port selection. The selector can
        # then rank candidate cages against the measured plug geometry.
        if not prime_sfp_plug_pose(self, get_observation, move_robot):
            send_feedback("fresh plug pose unavailable -- insertion aborted")
            return False

        # --- perceive the SFP port pose (multi-frame consensus so one bad frame
        #     or a wrong-port pick does not commit us to the wrong cage)
        perceived = self.perceive_port_pose_consensus(task, get_observation)
        if perceived is None:
            log.error("[rl] perception failed to produce an SFP port pose")
            return False
        port_pos, port_quat, reproj_px = perceived
        port_quat = np.asarray(port_quat, dtype=np.float64)
        port_quat /= max(float(np.linalg.norm(port_quat)), 1e-12)
        if not np.isfinite(reproj_px) or reproj_px > MAX_PORT_REPROJ_PX:
            log.error(f"[rl] rejecting SFP pose: reproj={reproj_px}px limit={MAX_PORT_REPROJ_PX}px")
            return False
        log.info(f"[rl] perceived port p={np.round(port_pos, 5).tolist()} "
                 f"q_wxyz={np.round(port_quat, 5).tolist()} reproj={reproj_px:.2f}px")

        # --- wrench baseline captured HERE, at the current handoff pose. We do
        #     NOT move up/away first, so this is a valid contact-force baseline.
        self._wrench_baseline = self._wrench_vector(obs)
        log.info(
            f"[rl] handoff wrench baseline="
            f"{np.round(self._wrench_baseline, 4).tolist()} "
            "controller=deterministic_sfp"
        )

        # One-shot grasp-calibration dump (set RL_INSERT_CALIB_DUMP=1). Logs the
        # TCP + any ground-truth tip frame so SFP_TIP_IN_TCP_QUAT can be re-solved.
        if CALIB_DUMP:
            self._dump_grasp_calibration()

        # --- handoff sanity: are we within the last-inch envelope of the mouth?
        Rp = port_frame(port_quat)
        tcp_pos, tcp_quat = self._tcp()
        tip_pos, R_tip = self._tip_from_tcp(tcp_pos, tcp_quat)
        dist = float(np.linalg.norm(tip_pos - port_pos))
        handoff_delta = Rp.T @ (tip_pos - port_pos)
        handoff_rot = _R_to_axis_angle(Rp.T @ R_tip)
        log.info(f"[rl] handoff check: |tip-mouth|={dist*1000:.1f}mm "
                 f"delta_port_mm={(handoff_delta*1000).round(2).tolist()} "
                 f"rot_err_deg={np.degrees(handoff_rot).round(2).tolist()}")
        if dist > HANDOFF_MAX_DIST:
            log.error(f"[rl] tip is {dist*1000:.0f} mm from the mouth -- outside the "
                      "last-inch envelope; the upstream macro must hand off closer. Aborting.")
            return False

        # Hand off to the canonical direct plug-to-port controller.
        return run_sfp_insertion(
            self, task, get_observation, move_robot, send_feedback,
            port_pos=port_pos, port_quat=port_quat, Rp=Rp)
