#
# RLInsert -- self-contained aic_model Policy for last-inch SFP insertion.
#
# This is a single-file policy: it does its OWN perception (YOLO-pose on the
# three cameras via PerceptionCore + multiview triangulation -> SFP port pose)
# and then runs the last-inch RL/guided insertion network (TorchScript, 69-dim
# observation -> 6-dim tanh cartesian residual).  It deliberately does NOT
# inherit from PerceptionInsert; everything needed lives here.
#
# Motion contract (per team decision):
#   * Perceive the target SFP port pose in base_link from the cameras.
#   * From WHEREVER the upstream macro handed the plug off, descend straight
#     down the perceived port insertion axis into the port.  No retract, no
#     "up then back down", no separate approach file.
#   * Object poses come only from perception + the robot's own TCP TF composed
#     with the fixed measured SFP-tip<-TCP transform (rl_insert_contract).
#
# Module path for aic_model:  -p policy:=aic_model.RLInsert
#
# The only imports outside this file are:
#   * PerceptionCore  -- the raw YOLO wrapper (weights load: best.pt / best_sc_pose.pt)
#   * rl_insert_contract -- the 69-dim obs + action math SHARED WITH TRAINING
#     (must stay identical to the MuJoCo side; do not fork it here).
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

from .rl_insert_contract import (
    HOME_QPOS as AIC_HOME_QPOS,
    SFP_TIP_IN_TCP_POS,
    SFP_TIP_IN_TCP_QUAT,
    build_observation69,
    guided_tip_target,
    port_frame,
    sfp_tip_pose_from_tcp,
    tcp_pose_for_sfp_tip,
)

# ----------------------------- configuration -------------------------------
MODEL_PATH = os.environ.get("RL_INSERT_MODEL", "/models/final_insert_sfp_flowstate_v1.ts")

# Perception weights.  best.pt = SFP YOLO-pose keypoints (the one loaded "in the
# form"); best_sc_pose.pt = SC pose (kept wired for later, not used for SFP).
# Default to the weights dir that ships next to PerceptionCore -- the base image
# already bakes best.pt / best_sc_pose.pt there, so this resolves correctly both
# in the repo and inside the installed site-packages layout.
import aic_example_policies.ros.perception_core as _pc_mod
_WEIGHTS_DIR = Path(
    os.environ.get(
        "AIC_PERCEPTION_WEIGHTS_DIR",
        str(Path(_pc_mod.__file__).resolve().parent / "weights"),
    )
)
NIC_WEIGHTS = os.environ.get("AIC_NIC_WEIGHTS", str(_WEIGHTS_DIR / "best.pt"))
SC_WEIGHTS = os.environ.get("AIC_SC_POSE_WEIGHTS", str(_WEIGHTS_DIR / "best_sc_pose.pt"))

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

# raw: original Contract-A behavior; zero: isolate pose/action contract;
# baseline: subtract the six-axis wrench observed before the handoff.
WRENCH_MODE = os.environ.get("RL_INSERT_WRENCH_MODE", "baseline").strip().lower()
CONTROL_MODE = os.environ.get("RL_INSERT_CONTROL_MODE", "guided").strip().lower()

STEP_DT = float(os.environ.get("RL_INSERT_STEP_DT", "0.10"))
MAX_RL_STEPS = int(os.environ.get("RL_INSERT_STEPS", "400"))
INSERT_DEPTH = 0.045    # m, mouth -> seated
# From the handoff, the tip may legitimately start a bit outside the mouth.
# This only guards against a totally wrong perception / handoff (macro failed).
HANDOFF_MAX_DIST = float(os.environ.get("RL_INSERT_HANDOFF_MAX_DIST", "0.12"))  # m

# Deploy action adapter (kept from the contract; identity by default).
ACTION_SIGN = np.array([
    float(v) for v in os.environ.get("RL_INSERT_ACTION_SIGN", "1,1,1,1,1,1").split(",")
], dtype=np.float64)
if ACTION_SIGN.shape != (6,) or not np.all(np.isin(ACTION_SIGN, (-1.0, 1.0))):
    raise ValueError("RL_INSERT_ACTION_SIGN must contain six comma-separated +/-1 values")
ACTION_GAIN = np.array([
    float(v) for v in os.environ.get("RL_INSERT_ACTION_GAIN", "1,1,1,1,1,1").split(",")
], dtype=np.float64)
if ACTION_GAIN.shape != (6,) or not np.all(np.isfinite(ACTION_GAIN)):
    raise ValueError("RL_INSERT_ACTION_GAIN must contain six finite comma-separated values")

SAFETY_MAX_LATERAL_M = float(os.environ.get("RL_INSERT_SAFETY_MAX_LATERAL_M", "0.006"))
SAFETY_MAX_RETREAT_M = float(os.environ.get("RL_INSERT_SAFETY_MAX_RETREAT_M", "0.015"))
SAFETY_MAX_ROTATION_RAD = float(os.environ.get("RL_INSERT_SAFETY_MAX_ROTATION_RAD", "0.26"))
# Force-reactive advance freeze (rl mode): once the plug touches the chamfer,
# pushing deeper while off-square levers the tip sideways and cocks it (the
# lateral+rotation+force-all-rising jam seen at seat).  Above this contact-force
# threshold, suppress the INWARD component of the commanded position delta so the
# impedance controller can re-square before advancing; lateral/rotation
# correction and retreat still pass through.  Set threshold high to disable.
ADVANCE_FREEZE_FORCE_N = float(os.environ.get("RL_INSERT_ADVANCE_FREEZE_FORCE_N", "1.5"))
# Grace window: the macro hands off with real lateral/rotation offset (~6 mm /
# ~8.6 deg) that the policy is meant to correct.  Enforcing the steady-state
# lateral/rotation limits on the raw handoff pose aborts on step 0 before the
# policy can act.  During the grace window we only abort if the error grows
# meaningfully WORSE than the handoff (true runaway drift), not for the
# pre-existing offset; after it, the hard limits apply.  The retreat and force
# guards stay live from step 0 regardless.
SAFETY_GRACE_STEPS = int(os.environ.get("RL_INSERT_SAFETY_GRACE_STEPS", "40"))
SAFETY_GRACE_LATERAL_SLACK_M = float(
    os.environ.get("RL_INSERT_SAFETY_GRACE_LATERAL_SLACK_M", "0.004"))
SAFETY_GRACE_ROTATION_SLACK_RAD = float(
    os.environ.get("RL_INSERT_SAFETY_GRACE_ROTATION_SLACK_RAD", "0.087"))

GUIDED_OUTSIDE_STEP_M = float(os.environ.get("RL_INSERT_GUIDED_OUTSIDE_STEP_M", "0.0015"))
GUIDED_INSIDE_STEP_M = float(os.environ.get("RL_INSERT_GUIDED_INSIDE_STEP_M", "0.00075"))

# --------------------- scripted align-then-descend mode ---------------------
# CONTROL_MODE="script": a pure-geometry two-phase controller with NO RL.
#   Phase 1 (align): hold the current standoff depth and cancel the perceived
#     lateral + rotation error until it is within tolerance (or a step budget).
#   Phase 2 (descend): step straight down the perceived port axis in small
#     increments that get PROGRESSIVELY SLOWER the deeper we go, so if the
#     alignment was slightly off we contact the mouth gently instead of ramming.
#   Force-reactive: above SCRIPT_CONTACT_FORCE_N, pause the descent (hold depth)
#     to let the impedance controller settle; sustained high force aborts.
# This is the align-first "script does alignment + gentle descent" test bed.
SCRIPT_ALIGN_LATERAL_TOL_M = float(os.environ.get("RL_INSERT_SCRIPT_ALIGN_LATERAL_TOL_M", "0.0010"))
SCRIPT_ALIGN_ROTATION_TOL_RAD = float(
    os.environ.get("RL_INSERT_SCRIPT_ALIGN_ROTATION_TOL_RAD", "0.026"))  # ~1.5 deg
SCRIPT_ALIGN_MAX_STEPS = int(os.environ.get("RL_INSERT_SCRIPT_ALIGN_MAX_STEPS", "80"))
SCRIPT_ALIGN_LATERAL_GAIN = float(os.environ.get("RL_INSERT_SCRIPT_ALIGN_LATERAL_GAIN", "0.6"))
SCRIPT_ALIGN_ROTATION_GAIN = float(os.environ.get("RL_INSERT_SCRIPT_ALIGN_ROTATION_GAIN", "0.6"))
SCRIPT_ALIGN_MAX_LATERAL_STEP_M = float(
    os.environ.get("RL_INSERT_SCRIPT_ALIGN_MAX_LATERAL_STEP_M", "0.0015"))
SCRIPT_ALIGN_MAX_ROTATION_STEP_RAD = float(
    os.environ.get("RL_INSERT_SCRIPT_ALIGN_MAX_ROTATION_STEP_RAD", "0.026"))
# Progressive descent: step size shrinks linearly from FAR to NEAR as depth goes
# from the mouth (0) to seated (INSERT_DEPTH). Far steps move quickly through
# free space; near steps creep so a missed alignment taps rather than rams.
SCRIPT_DESCEND_STEP_FAR_M = float(os.environ.get("RL_INSERT_SCRIPT_DESCEND_STEP_FAR_M", "0.0020"))
SCRIPT_DESCEND_STEP_NEAR_M = float(os.environ.get("RL_INSERT_SCRIPT_DESCEND_STEP_NEAR_M", "0.0004"))
# Standoff the tip holds during phase-1 alignment (m along the port axis; the
# mouth is depth=0, negative is above/outside). Aligns just above the mouth.
SCRIPT_ALIGN_STANDOFF_M = float(os.environ.get("RL_INSERT_SCRIPT_ALIGN_STANDOFF_M", "-0.004"))
# Post-align pre-engage bias: after Phase-1 alignment converges (perception read +
# lateral/rotation cancelled), apply ONE fixed offset before Phase-2 descent, to
# pre-seat the plug against the (near-constant) wedge location so the seat starts
# from a better spot. Both are in the PERCEIVED PORT frame:
#   BIAS_X_M  = shift along port X (lateral, along the pin row; signed)
#   BIAS_Y_M  = shift along port Y (signed)
#   BIAS_RX_RAD = tilt about port X (signed; negative tips one way, positive the other)
# Set to 0 to disable (no bias, exactly the previous behavior). Tune per rebuild
# via RL_INSERT_SCRIPT_BIAS_{X_M,Y_M,RX_RAD} (no runtime override in Flowstate).
SCRIPT_BIAS_X_M = float(os.environ.get("RL_INSERT_SCRIPT_BIAS_X_M", "-0.001"))      # -1.0 mm (-X)
SCRIPT_BIAS_Y_M = float(os.environ.get("RL_INSERT_SCRIPT_BIAS_Y_M", "0.010"))       # +10.0 mm (+Y)
SCRIPT_BIAS_RX_RAD = float(os.environ.get("RL_INSERT_SCRIPT_BIAS_RX_RAD", "-0.122173"))  # -7.0 deg
# Contact-force handling during descent (uses the tared wrench, like rl mode).
# Above CONTACT_FORCE we are touching the mouth: switch from fast free-space
# descent to a SLOW force-limited seat push (keep advancing gently, do NOT freeze)
# until force reaches the SEAT_FORCE cap, at which point we hold that one step.
# This is the scripted force-reactive seat: square plug pressing gently home.
SCRIPT_CONTACT_FORCE_N = float(os.environ.get("RL_INSERT_SCRIPT_CONTACT_FORCE_N", "5.0"))
SCRIPT_SEAT_FORCE_N = float(os.environ.get("RL_INSERT_SCRIPT_SEAT_FORCE_N", "12.0"))
SCRIPT_SEAT_STEP_M = float(os.environ.get("RL_INSERT_SCRIPT_SEAT_STEP_M", "0.0003"))
# Wedge is a GEOMETRIC bind: depth stalled AND lateral blown out. Steady small
# lateral at a stall is normal contact (keep seating), NOT a wedge. Only abort as
# a wedge if lateral exceeds this while stalled.
SCRIPT_WEDGE_LATERAL_M = float(os.environ.get("RL_INSERT_SCRIPT_WEDGE_LATERAL_M", "0.0015"))
# Re-align if lateral/rotation drifts back out of tolerance mid-descent.
SCRIPT_REALIGN_LATERAL_M = float(os.environ.get("RL_INSERT_SCRIPT_REALIGN_LATERAL_M", "0.0020"))
# Wedge/stall detection: if we command inward motion but depth does not progress
# for this many steps while force stays BELOW the hard-jam abort (i.e. it is a
# geometric bind, not an axial crush), declare wedged and abort instead of
# grinding to the step budget. A wedge with a repeatable stall depth across runs
# indicates a FIXED grasp mis-calibration; a varying stall depth indicates the
# grasp angle varies run-to-run (the kinematic tip cannot be trusted either way).
SCRIPT_STALL_STEPS = int(os.environ.get("RL_INSERT_SCRIPT_STALL_STEPS", "40"))
SCRIPT_STALL_PROGRESS_M = float(os.environ.get("RL_INSERT_SCRIPT_STALL_PROGRESS_M", "0.0008"))
# Hand-off gate: once the plug is pre-engaged (tilted + biased into the mouth), the
# "lateral distance from the perceived mouth" reading is high by construction (the
# tip is deep and cocked, not wandering), so it is NOT a valid wedge signal. Instead
# keep descending through a depth-stall until FORCE builds -- a real jam -- then hold
# for the RL hand-off. This lets the script pre-engage the plug as deep as it can
# GENTLY before handing off, rather than quitting early on a false lateral reading.
SCRIPT_HANDOFF_FORCE_N = float(os.environ.get("RL_INSERT_SCRIPT_HANDOFF_FORCE_N", "10.0"))

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
GUIDED_MAX_LEAD_M = float(os.environ.get("RL_INSERT_GUIDED_MAX_LEAD_M", "0.004"))
GUIDED_CONTACT_DEPTH_M = float(os.environ.get("RL_INSERT_GUIDED_CONTACT_DEPTH_M", "-0.008"))
GUIDED_CONTACT_MAX_LEAD_M = float(os.environ.get("RL_INSERT_GUIDED_CONTACT_MAX_LEAD_M", "0.020"))

# Impedance mirroring training (CartesianImpedanceAction, kp~100):
STIFFNESS = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
GUIDED_STIFFNESS = [500.0, 500.0, 500.0, 60.0, 60.0, 60.0]
GUIDED_DAMPING = [100.0, 100.0, 100.0, 25.0, 25.0, 25.0]

HOME_QPOS_ENV = os.environ.get("RL_INSERT_HOME_QPOS", "")

# VERIFY: MuJoCo joint order used in training.
JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Success/abort thresholds for the insertion loop tail.
SEAT_DEPTH_TOL = 0.002    # m
SEAT_LATERAL_MAX = 0.005  # m
FORCE_ABORT_N = float(os.environ.get("RL_INSERT_FORCE_ABORT_N", "18.0"))
FORCE_ABORT_STEPS = 8     # consecutive steps above FORCE_ABORT_N => abort

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
    """Rodrigues: axis-angle vector (rad) -> rotation matrix. Inverse of
    _R_to_axis_angle."""
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


class RLInsert(Policy):
    """Single-file SFP last-inch insertion policy (perception + RL)."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        log = self.get_logger()
        if WRENCH_MODE not in ("raw", "zero", "baseline"):
            raise ValueError("RL_INSERT_WRENCH_MODE must be raw, zero, or baseline")
        if CONTROL_MODE not in ("rl", "guided", "script"):
            raise ValueError("RL_INSERT_CONTROL_MODE must be rl, guided, or script")

        self._task = None
        self._wrench_baseline = np.zeros(6, dtype=np.float64)
        self._last_port_quat_wxyz = None
        self._last_port_reproj_px = None

        # ---- perception weights (best.pt / best_sc_pose.pt) ----
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

        # ---- final-insert RL/guided network ----
        import torch
        log.info(f"[rl] torch {torch.__version__}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"[rl] model not found at {MODEL_PATH}; COPY it into the image "
                "and/or set RL_INSERT_MODEL")
        self._torch = torch
        self._model = torch.jit.load(MODEL_PATH, map_location="cpu")
        self._model.eval()
        probe = self._forward(np.zeros(69, dtype=np.float32))
        log.info(f"[rl] model loaded from {MODEL_PATH}; probe={np.round(probe, 4)}")

        if HOME_QPOS_ENV:
            self._home_qpos = np.array([float(v) for v in HOME_QPOS_ENV.split(",")])
            assert self._home_qpos.shape == (6,)
            log.info(f"[rl] home qpos from env: {self._home_qpos}")
        else:
            self._home_qpos = AIC_HOME_QPOS.copy()
            log.info(f"[rl] using training home qpos: {self._home_qpos}")

    # ------------------------------------------------------------------ util
    def _forward(self, obs69):
        with self._torch.no_grad():
            out = self._model(self._torch.from_numpy(obs69.astype(np.float32)))
        return np.clip(out.detach().cpu().numpy().reshape(-1), -1.0, 1.0)

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
        """SFP plug tip pose from TCP + the measured SFP-tip<-TCP transform."""
        return sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)

    def _dump_grasp_calibration(self):
        """One-shot: log everything needed to recalibrate SFP_TIP_IN_TCP_QUAT.

        Dumps (a) the TCP pose in base_link, (b) the tip pose the CURRENT
        (possibly wrong) transform produces, and (c) any ground-truth plug/tip TF
        frame that resolves -- all in base_link. From (a) + a true tip pose the
        exact tip-relative-to-TCP transform is R_tcp^T @ R_tip_true (right axis),
        which we then paste into rl_insert_contract. Logs only; no motion change.
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
                     f"(paste BOTH into rl_insert_contract.py if this frame is the true tip)")
        if not found_any:
            log.warn(f"[calib] no ground-truth plug/tip frame resolved from "
                     f"{CALIB_PLUG_FRAMES}. Set RL_INSERT_CALIB_PLUG_FRAMES to the "
                     f"correct frame name, or provide TCP + true-tip poses another way.")
        log.info("[calib] === END DUMP ===")

    def _tcp_target_for_tip(self, tip_pos, R_tip):
        tcp_pos, q_tcp = tcp_pose_for_sfp_tip(tip_pos, R_tip)
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

    def _make_sfp_multiview_candidates(self, per_cam):
        cams = [c for c, cand in per_cam.items() if cand]
        if len(cams) < 2:
            return []
        for c in cams:
            per_cam[c] = per_cam[c][:5]

        candidates = []
        for picks in itertools.product(*[per_cam[c] for c in cams]):
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
                    dets.append({
                        "kps": kps, "conf": det["conf"], "K": K, "T": T,
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

    # ------------------------------------------------------- obs construction
    def _joint_vec(self, joint_state, attr):
        vals = dict(zip(joint_state.name, getattr(joint_state, attr)))
        try:
            return np.array([vals[j] for j in JOINT_ORDER])
        except KeyError:
            arr = np.array(list(getattr(joint_state, attr)))
            return arr[:6] if arr.size >= 6 else np.zeros(6)

    @staticmethod
    def _wrench_vector(obs_msg):
        w = obs_msg.wrist_wrench.wrench
        return np.array([w.force.x, w.force.y, w.force.z,
                         w.torque.x, w.torque.y, w.torque.z], dtype=np.float64)

    def _build_obs(self, obs_msg, tcp_pos, tcp_quat, tip_pos, R_tip,
                   port_pos, port_quat, prev_tcp, dt, last_action):
        jp = self._joint_vec(obs_msg.joint_states, "position")
        jv = self._joint_vec(obs_msg.joint_states, "velocity")
        v_lin = np.zeros(3, dtype=np.float64)
        v_ang = np.zeros(3, dtype=np.float64)
        if prev_tcp is not None and dt > 1e-4:
            v_lin = (tcp_pos - prev_tcp[0]) / dt
            R_prev = _q_to_R(*prev_tcp[1])
            R_cur = _q_to_R(*tcp_quat)
            v_ang = _R_to_axis_angle(R_cur @ R_prev.T) / dt
        return build_observation69(
            joint_pos=jp, joint_vel=jv,
            tcp_pos=tcp_pos, tcp_quat=tcp_quat,
            tcp_linear_velocity_world=v_lin, tcp_angular_velocity_world=v_ang,
            port_pos=port_pos, port_quat=port_quat,
            tip_pos=tip_pos, tip_rotation=R_tip,
            wrench=self._wrench_vector(obs_msg), last_action=last_action,
            home_qpos=self._home_qpos, wrench_mode=WRENCH_MODE,
            wrench_baseline=self._wrench_baseline,
        )

    # ----------------------------------------------------------------- main
    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback):
        log = self.get_logger()
        try:
            return self._run(task, get_observation, move_robot, send_feedback)
        except Exception:
            log.error("[rl] insert_cable crashed:\n" + traceback.format_exc())
            send_feedback("insert_cable crashed -- see model log")
            return False

    def _run(self, task, get_observation, move_robot, send_feedback):
        log = self.get_logger()
        self._task = task
        plug_type = (task.plug_type or "sfp").lower()
        log.info(f"[rl] task: cable={task.cable_name} plug={task.plug_name} "
                 f"({plug_type}) port={task.port_name} module={task.target_module_name}")
        if plug_type != "sfp":
            log.error("[rl] this single-file policy is SFP-only")
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
        log.info(f"[rl] handoff wrench baseline mode={WRENCH_MODE} "
                 f"baseline={np.round(self._wrench_baseline, 4).tolist()} control={CONTROL_MODE}")

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

        # Scripted (no-RL) align-then-progressive-descent test path.
        if CONTROL_MODE == "script":
            return self._run_script(
                get_observation, move_robot, send_feedback,
                port_pos=port_pos, port_quat=port_quat, Rp=Rp)

        # ------------------------------ insertion loop -----------------------
        # Straight descent from the CURRENT pose down the perceived port axis.
        # No preposition / retract / re-approach.
        send_feedback("rl last-inch (straight descent)")
        insert_axis = Rp[:, 2]
        last_action = np.zeros(6, dtype=np.float32)
        prev_tcp = None
        prev_t = None
        cmd_pos, cmd_R = tcp_pos.copy(), _q_to_R(*tcp_quat)
        force_hot_steps = 0
        depth = float("nan")
        handoff_depth = float(handoff_delta[2])
        # Baselines for the grace-window safety gate: the lateral/rotation error
        # present at handoff, which the policy is expected to reduce.
        handoff_lateral = float(np.linalg.norm(handoff_delta[:2]))
        handoff_rotation = float(np.linalg.norm(handoff_rot))
        planned_depth = handoff_depth

        for step in range(MAX_RL_STEPS):
            obs = get_observation()
            if obs is None:
                self.sleep_for(STEP_DT)
                continue
            try:
                tcp_pos, tcp_quat = self._tcp()
            except TransformException as ex:
                log.warn(f"[rl] tcp TF miss: {ex}")
                self.sleep_for(STEP_DT)
                continue
            tip_pos, R_tip = self._tip_from_tcp(tcp_pos, tcp_quat)

            now = time.monotonic()
            dt = (now - prev_t) if prev_t else STEP_DT
            o = self._build_obs(obs, tcp_pos, tcp_quat, tip_pos, R_tip,
                                port_pos, port_quat, prev_tcp, dt, last_action)
            raw_act = self._forward(o)
            act = np.clip(raw_act * ACTION_SIGN * ACTION_GAIN, -1.0, 1.0)
            if step == 0 or step % 50 == 0:
                log.info("[rl] obs69=" + np.array2string(
                    o, precision=5, max_line_width=1000, separator=","))
            last_action = raw_act
            prev_tcp = (tcp_pos, tcp_quat)
            prev_t = now

            depth_vec = Rp.T @ (tip_pos - port_pos)
            depth = float(depth_vec[2])
            lateral = float(np.linalg.norm(depth_vec[:2]))
            rotation_error = _R_to_axis_angle(Rp.T @ R_tip)
            rotation_error_norm = float(np.linalg.norm(rotation_error))
            wrench_delta = self._wrench_vector(obs) - self._wrench_baseline
            f_mag = float(np.linalg.norm(wrench_delta[:3]))
            guided_max_lead = float("nan")
            target_depth = float("nan")

            if CONTROL_MODE == "guided":
                guided_max_lead = (
                    GUIDED_CONTACT_MAX_LEAD_M if depth >= GUIDED_CONTACT_DEPTH_M
                    else GUIDED_MAX_LEAD_M)
                target_tip, target_R, planned_depth, target_depth = guided_tip_target(
                    port_pos=port_pos, port_quat=port_quat,
                    current_depth=depth, planned_depth=planned_depth,
                    insert_depth=INSERT_DEPTH,
                    outside_step=GUIDED_OUTSIDE_STEP_M, inside_step=GUIDED_INSIDE_STEP_M,
                    max_lead=guided_max_lead)
                target_pose = self._tcp_target_for_tip(target_tip, target_R)
                command_delta_port = Rp.T @ (target_tip - tip_pos)
                self.set_pose_target(move_robot, target_pose,
                                     stiffness=GUIDED_STIFFNESS, damping=GUIDED_DAMPING)
            else:
                from .rl_insert_contract import deploy_action_delta
                dp, drot_world = deploy_action_delta(act, port_quat)
                # Force-reactive advance freeze: if we're in contact (force above
                # threshold) and the policy commands further inward motion, drop
                # only the inward component so it stops driving into a jam and
                # lets the controller re-square.  Lateral correction and any
                # retreat (negative inward) pass through unchanged.
                if f_mag > ADVANCE_FREEZE_FORCE_N:
                    inward = float(dp @ insert_axis)
                    if inward > 0.0:
                        dp = dp - inward * insert_axis
                        if step % 10 == 0:
                            log.info(f"[rl] advance freeze: force={f_mag:.2f}N > "
                                     f"{ADVANCE_FREEZE_FORCE_N:.1f}N, held inward "
                                     f"{inward*1000:.2f}mm")
                angle = float(np.linalg.norm(drot_world))
                if angle < 1e-10:
                    dR = np.eye(3)
                else:
                    k = drot_world / angle
                    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                    dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                cmd_pos = cmd_pos + dp
                cmd_R = dR @ cmd_R
                q = _rotmat_to_quat_wxyz(cmd_R)
                command_delta_port = Rp.T @ dp
                self.set_pose_target(
                    move_robot,
                    Pose(position=Point(x=cmd_pos[0], y=cmd_pos[1], z=cmd_pos[2]),
                         orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])),
                    stiffness=STIFFNESS, damping=DAMPING)

            if step % 10 == 0:
                log.info(f"[rl] step {step}: depth={depth*1000:.1f}/{INSERT_DEPTH*1000:.0f}mm "
                         f"lateral={lateral*1000:.1f}mm rot_err_deg={np.degrees(rotation_error_norm):.1f} "
                         f"force_delta_N={f_mag:.2f} control={CONTROL_MODE} "
                         f"target_depth_mm={target_depth*1000:.1f} "
                         f"raw_act={np.round(raw_act, 3).tolist()} "
                         f"cmd_delta_port_mm={(command_delta_port*1000).round(3).tolist()}")

            # Retreat guard is always live: pulling away from the board is never
            # acceptable, even during the grace window.
            retreated = depth < handoff_depth - SAFETY_MAX_RETREAT_M
            if step < SAFETY_GRACE_STEPS:
                # Give the policy room to correct the handoff offset: only abort
                # if lateral/rotation grows meaningfully worse than at handoff.
                lateral_limit = max(SAFETY_MAX_LATERAL_M,
                                    handoff_lateral + SAFETY_GRACE_LATERAL_SLACK_M)
                rotation_limit = max(SAFETY_MAX_ROTATION_RAD,
                                     handoff_rotation + SAFETY_GRACE_ROTATION_SLACK_RAD)
            else:
                lateral_limit = SAFETY_MAX_LATERAL_M
                rotation_limit = SAFETY_MAX_ROTATION_RAD
            if (lateral > lateral_limit
                    or retreated
                    or rotation_error_norm > rotation_limit):
                log.error(f"[rl] contract safety abort | step={step} "
                          f"depth_mm={depth*1000:.1f} "
                          f"initial_mm={handoff_depth*1000:.1f} lateral_mm={lateral*1000:.1f} "
                          f"(limit={lateral_limit*1000:.1f}) "
                          f"rot_err_deg={np.degrees(rotation_error_norm):.1f} "
                          f"(limit={np.degrees(rotation_limit):.1f})")
                send_feedback("rl contract safety abort")
                return False

            if depth >= INSERT_DEPTH - SEAT_DEPTH_TOL and lateral < SEAT_LATERAL_MAX:
                log.info(f"[rl] seated: depth={depth*1000:.1f}mm lateral={lateral*1000:.1f}mm "
                         f"after {step + 1} steps")
                send_feedback("rl insert seated")
                return True

            force_hot_steps = force_hot_steps + 1 if f_mag > FORCE_ABORT_N else 0
            if force_hot_steps >= FORCE_ABORT_STEPS:
                log.error(f"[rl] sustained wrist force {f_mag:.1f} N over {FORCE_ABORT_STEPS} "
                          f"steps at depth={depth*1000:.1f}mm -- jammed; retreating and aborting.")
                retreat = cmd_pos - insert_axis * 0.010
                q = _rotmat_to_quat_wxyz(cmd_R)
                self.set_pose_target(
                    move_robot,
                    Pose(position=Point(x=retreat[0], y=retreat[1], z=retreat[2]),
                         orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])),
                    stiffness=STIFFNESS, damping=DAMPING)
                self.sleep_for(1.0)
                send_feedback("rl insert jammed -- aborted under force budget")
                return False

            self.sleep_for(STEP_DT)

        log.error(f"[rl] step budget ({MAX_RL_STEPS}) exhausted; "
                  f"last depth={depth*1000:.1f}/{INSERT_DEPTH*1000:.0f} mm")
        send_feedback("rl insert timed out")
        return False

    # --------------------------------------------------- scripted controller ---
    def _script_errors(self, Rp, port_pos):
        """Current (depth, lateral, rotation-error-vec) of the plug tip in the
        perceived port frame, from live TCP TF. depth: +inward, 0 at the mouth."""
        tcp_pos, tcp_quat = self._tcp()
        tip_pos, R_tip = self._tip_from_tcp(tcp_pos, tcp_quat)
        d = Rp.T @ (tip_pos - port_pos)
        rot_err = _R_to_axis_angle(Rp.T @ R_tip)
        return (float(d[2]), d[:2].copy(), rot_err, tip_pos, R_tip)

    def _run_script(self, get_observation, move_robot, send_feedback, *,
                    port_pos, port_quat, Rp):
        """Pure-geometry align-then-progressive-descent. No RL, no network.

        Phase 1: hold a standoff depth and cancel perceived lateral + rotation
        error to tolerance.  Phase 2: creep straight down the perceived port
        axis, step size shrinking with depth so a slightly-missed alignment taps
        the mouth gently rather than ramming.  Force pauses the descent; sustained
        high force retreats and aborts.  Uses the tared wrench from _run().
        """
        log = self.get_logger()
        insert_axis = Rp[:, 2]                       # unit port axis in base_link
        R_port = Rp                                  # target tip orientation == port frame
        send_feedback("script align+descend (no RL)")

        # Align at WHERE we were handed off (do not force a descent to a fixed
        # standoff first). This lets a deliberately-higher handoff -- e.g. +5mm so
        # the cameras see the port clearer -- simply align higher, then descend
        # progressively. Never align DEEPER than SCRIPT_ALIGN_STANDOFF_M (if handed
        # off already close/below, hold there). The 120mm HANDOFF_MAX_DIST guard
        # (checked in _run before we get here) already permits a higher start.
        handoff_depth0, _, _, _, _ = self._script_errors(Rp, port_pos)
        align_standoff = min(handoff_depth0, SCRIPT_ALIGN_STANDOFF_M)
        log.info(f"[script] align standoff = {align_standoff*1000:.1f}mm "
                 f"(handoff depth {handoff_depth0*1000:.1f}mm)")

        # ---- Phase 1: align at standoff -------------------------------------
        aligned = False
        for astep in range(SCRIPT_ALIGN_MAX_STEPS):
            depth, lat_vec, rot_err, tip_pos, R_tip = self._script_errors(Rp, port_pos)
            lateral = float(np.linalg.norm(lat_vec))
            rot_norm = float(np.linalg.norm(rot_err))
            if lateral <= SCRIPT_ALIGN_LATERAL_TOL_M and rot_norm <= SCRIPT_ALIGN_ROTATION_TOL_RAD:
                aligned = True
                log.info(f"[script] aligned after {astep} steps: "
                         f"lateral={lateral*1000:.2f}mm rot_err_deg={np.degrees(rot_norm):.2f}")
                break
            # Command: cancel lateral (bounded), pull depth toward the standoff,
            # and square orientation to the port frame. All in base_link.
            lat_cmd = -SCRIPT_ALIGN_LATERAL_GAIN * lat_vec
            lat_step = float(np.linalg.norm(lat_cmd))
            if lat_step > SCRIPT_ALIGN_MAX_LATERAL_STEP_M:
                lat_cmd *= SCRIPT_ALIGN_MAX_LATERAL_STEP_M / lat_step
            # lat_vec is expressed in the port XY plane -> lift to base_link
            lat_world = Rp[:, 0] * lat_cmd[0] + Rp[:, 1] * lat_cmd[1]
            depth_err = align_standoff - depth                   # toward standoff
            depth_world = insert_axis * float(np.clip(depth_err, -0.002, 0.002))
            target_tip = tip_pos + lat_world + depth_world
            # Orientation goes toward the port frame, rate-limited. rot_err is the
            # tip-vs-port error as an axis-angle in the PORT frame (E = Rp^T @ R_tip).
            # Keep the fraction of the error that a single capped step leaves:
            #   target_R = Rp @ axis_angle_to_R((1 - move_frac) * rot_err)
            if rot_norm > SCRIPT_ALIGN_MAX_ROTATION_STEP_RAD:
                move_frac = SCRIPT_ALIGN_MAX_ROTATION_STEP_RAD / rot_norm
                target_R = R_port @ _axis_angle_to_R((1.0 - move_frac) * rot_err)
            else:
                target_R = R_port
            self.set_pose_target(
                move_robot, self._tcp_target_for_tip(target_tip, target_R),
                stiffness=GUIDED_STIFFNESS, damping=GUIDED_DAMPING)
            if astep % 10 == 0:
                log.info(f"[script] align step {astep}: lateral={lateral*1000:.2f}mm "
                         f"rot_err_deg={np.degrees(rot_norm):.2f} depth_mm={depth*1000:.1f}")
            self.sleep_for(STEP_DT)

        if not aligned:
            log.error(f"[script] failed to align within {SCRIPT_ALIGN_MAX_STEPS} steps; "
                      f"aborting before descent (last lateral={lateral*1000:.2f}mm "
                      f"rot_err_deg={np.degrees(rot_norm):.2f}).")
            send_feedback("script align failed -- aborted before descent")
            return False

        # ---- Post-align pre-engage bias -------------------------------------
        # After alignment, nudge the plug to the (near-constant) wedge spot so the
        # seat starts pre-engaged. Fixed offset in the perceived port frame: shift
        # along port Y by SCRIPT_BIAS_Y_M and tilt about port X by SCRIPT_BIAS_RX_RAD.
        # CRITICAL: the tilt must PERSIST through descent -- R_seat below replaces
        # R_port as the descent orientation. Otherwise Phase 2 re-commands the square
        # R_port every step and erases the tilt instantly (why 2.5 and 6 deg looked
        # identical). The positional bias (bias_world) is applied ONCE here as a
        # one-shot pre-engage move; the descent below commands pure inward motion and
        # must NOT re-add it (re-adding to the live tip each step walks the plug off).
        # No-op when both are 0 (R_seat == R_port, bias_world == 0).
        R_seat = R_port @ _axis_angle_to_R(
            np.array([SCRIPT_BIAS_RX_RAD, 0.0, 0.0], dtype=float))
        bias_world = (Rp[:, 0] * SCRIPT_BIAS_X_M              # port X (lateral) offset
                      + Rp[:, 1] * SCRIPT_BIAS_Y_M)          # port Y offset, both signed
        if SCRIPT_BIAS_X_M != 0.0 or SCRIPT_BIAS_Y_M != 0.0 or SCRIPT_BIAS_RX_RAD != 0.0:
            depth, lat_vec, rot_err, tip_pos, R_tip = self._script_errors(Rp, port_pos)
            log.info(f"[script] pre-engage bias (persists in descent): "
                     f"dX={SCRIPT_BIAS_X_M*1000:.2f}mm dY={SCRIPT_BIAS_Y_M*1000:.2f}mm "
                     f"rX={np.degrees(SCRIPT_BIAS_RX_RAD):.2f}deg")
            self.set_pose_target(
                move_robot, self._tcp_target_for_tip(tip_pos + bias_world, R_seat),
                stiffness=GUIDED_STIFFNESS, damping=GUIDED_DAMPING)
            self.sleep_for(STEP_DT)
            send_feedback("script pre-engage bias applied")

        # ---- Phase 2: progressive slow descent ------------------------------
        force_hot_steps = 0
        # Wedge/stall tracking: deepest depth reached, and how long we have been
        # commanding inward without beating it. Also record the lateral swing at
        # stall (the wedge signature) for the fixed-vs-variable-grasp diagnosis.
        deepest = -np.inf
        stall_steps = 0
        stall_depth = float("nan")
        stall_lat_min = np.inf
        stall_lat_max = -np.inf
        for dstep in range(MAX_RL_STEPS):
            obs = get_observation()
            depth, lat_vec, rot_err, tip_pos, R_tip = self._script_errors(Rp, port_pos)
            lateral = float(np.linalg.norm(lat_vec))
            rot_norm = float(np.linalg.norm(rot_err))
            f_mag = float("nan")
            if obs is not None:
                wrench_delta = self._wrench_vector(obs) - self._wrench_baseline
                f_mag = float(np.linalg.norm(wrench_delta[:3]))

            # Seated?
            if depth >= INSERT_DEPTH - SEAT_DEPTH_TOL and lateral < SEAT_LATERAL_MAX:
                log.info(f"[script] seated: depth={depth*1000:.1f}mm lateral={lateral*1000:.1f}mm "
                         f"after {dstep} descent steps")
                send_feedback("script insert seated")
                return True

            # Sustained high force -> jam: HOLD (do not retreat, per instruction).
            if np.isfinite(f_mag) and f_mag > FORCE_ABORT_N:
                force_hot_steps += 1
            else:
                force_hot_steps = 0
            if force_hot_steps >= FORCE_ABORT_STEPS:
                log.error(f"[script] sustained force {f_mag:.1f}N over {FORCE_ABORT_STEPS} steps "
                          f"at depth={depth*1000:.1f}mm -- HOLDING position (no retreat), stopping.")
                self.set_pose_target(
                    move_robot, self._tcp_target_for_tip(tip_pos, R_seat),
                    stiffness=STIFFNESS, damping=DAMPING)
                send_feedback("script high force -- holding (no retreat)")
                return False

            # Track depth stall for wedge detection. A WEDGE is a GEOMETRIC bind:
            # depth stalled AND lateral blown out (the plug levering sideways
            # because it entered cocked). Steady SMALL lateral at a stall is just
            # normal contact at the mouth -- keep seating, do NOT call it a wedge.
            if depth > deepest + SCRIPT_STALL_PROGRESS_M:
                deepest = depth
                stall_steps = 0
                stall_depth = float("nan")
                stall_lat_min, stall_lat_max = np.inf, -np.inf
            else:
                stall_steps += 1
                if np.isnan(stall_depth):
                    stall_depth = depth
                stall_lat_min = min(stall_lat_min, lateral)
                stall_lat_max = max(stall_lat_max, lateral)
            # Hand off to RL when the plug is pre-engaged and DEPTH stalls, OR force
            # builds. NOTE: we do NOT use lateral distance here -- once pre-engaged
            # (tilted/biased into the mouth) the lateral vs the perceived mouth is
            # high by construction and is NOT a wedge signal. The wedge we care about
            # is a LOW-FORCE geometric catch: depth stuck for many steps at modest
            # force. So depth-stall alone is a valid hand-off; force is an additional
            # immediate trigger for a harder jam.
            force_jam = np.isfinite(f_mag) and f_mag >= SCRIPT_HANDOFF_FORCE_N
            depth_stalled = stall_steps >= SCRIPT_STALL_STEPS
            if depth_stalled or force_jam:
                trigger = "force-jam" if force_jam else "depth-stall"
                log.error(
                    f"[script] HANDOFF ({trigger}) at depth {stall_depth*1000:.1f}mm, "
                    f"{stall_steps} stalled steps, force {f_mag:.2f}N, "
                    f"lateral {lateral*1000:.2f}mm. "
                    f"Pre-engaged -- HOLDING position (no retreat), stopping for RL.")
                log.error(
                    f"[script] STALL-SIGNATURE stall_depth_mm={stall_depth*1000:.2f} "
                    f"lat_swing_mm={(stall_lat_max-stall_lat_min)*1000:.2f} "
                    f"lat_mm={lateral*1000:.2f} force_N={f_mag:.2f}")
                # Do NOT retreat on collision (per instruction). Hold the current
                # contact pose so the plug stays pre-engaged in the mouth -- ready for
                # the force-reactive RL to take over from here.
                self.set_pose_target(
                    move_robot, self._tcp_target_for_tip(tip_pos, R_seat),
                    stiffness=STIFFNESS, damping=DAMPING)
                send_feedback("script pre-engaged -- holding for RL (no retreat)")
                return False

            # Step sizing:
            #  - free space (low force): progressive fast descent, shrinking with depth.
            #  - in contact (force > CONTACT): SLOW force-limited seat push -- keep
            #    advancing gently (this is the scripted force-reactive seat), and
            #    only hold for one step once we hit the SEAT_FORCE cap.
            if np.isfinite(f_mag) and f_mag > SCRIPT_CONTACT_FORCE_N:
                if f_mag >= SCRIPT_SEAT_FORCE_N:
                    step_m = 0.0            # at the seat-force cap: hold this step
                else:
                    step_m = SCRIPT_SEAT_STEP_M   # gentle push toward seated
                if dstep % 10 == 0:
                    log.info(f"[script] seating: force={f_mag:.2f}N depth={depth*1000:.1f}mm "
                             f"step_mm={step_m*1000:.2f} lateral={lateral*1000:.2f}mm")
            else:
                frac = float(np.clip(depth / max(INSERT_DEPTH, 1e-6), 0.0, 1.0))
                step_m = (SCRIPT_DESCEND_STEP_FAR_M
                          + frac * (SCRIPT_DESCEND_STEP_NEAR_M - SCRIPT_DESCEND_STEP_FAR_M))

            # Keep lateral squared during descent; re-tighten if it drifts out.
            # BUT when a pre-engage bias is active, the plug is DELIBERATELY offset
            # from the perceived mouth (tilted/biased in), so the mouth-relative
            # lateral is high by design -- re-aligning would fight the bias and undo
            # the pre-engagement. Only re-align when NO bias is set.
            bias_active = (SCRIPT_BIAS_X_M != 0.0 or SCRIPT_BIAS_Y_M != 0.0
                           or SCRIPT_BIAS_RX_RAD != 0.0)
            lat_world = np.zeros(3)
            if not bias_active and lateral > SCRIPT_REALIGN_LATERAL_M:
                lat_cmd = -SCRIPT_ALIGN_LATERAL_GAIN * lat_vec
                lat_step = float(np.linalg.norm(lat_cmd))
                if lat_step > SCRIPT_ALIGN_MAX_LATERAL_STEP_M:
                    lat_cmd *= SCRIPT_ALIGN_MAX_LATERAL_STEP_M / lat_step
                lat_world = Rp[:, 0] * lat_cmd[0] + Rp[:, 1] * lat_cmd[1]

            # In contact use the softer (compliant) stiffness so the plug can give
            # against the mouth as it seats; use the stiff gains in free space.
            in_contact = np.isfinite(f_mag) and f_mag > SCRIPT_CONTACT_FORCE_N
            stiff = STIFFNESS if in_contact else GUIDED_STIFFNESS
            damp = DAMPING if in_contact else GUIDED_DAMPING
            # Descent commands PURE inward motion from the live tip position. The
            # positional bias (bias_world) was applied ONCE at pre-engage above and
            # must NOT be re-added here -- adding it every step to the live tip_pos
            # is a runaway integrator that walks the plug off the board (observed
            # lateral drifting to 80+ mm). Orientation R_seat is absolute (a target
            # pose, not an increment), so it correctly persists every step.
            target_tip = tip_pos + insert_axis * step_m + lat_world
            self.set_pose_target(
                move_robot, self._tcp_target_for_tip(target_tip, R_seat),
                stiffness=stiff, damping=damp)

            if dstep % 10 == 0:
                log.info(f"[script] descend step {dstep}: depth={depth*1000:.1f}/"
                         f"{INSERT_DEPTH*1000:.0f}mm step_mm={step_m*1000:.2f} "
                         f"lateral={lateral*1000:.2f}mm rot_err_deg={np.degrees(rot_norm):.2f} "
                         f"force_N={f_mag:.2f}")
            self.sleep_for(STEP_DT)

        log.error(f"[script] descent step budget ({MAX_RL_STEPS}) exhausted; "
                  f"last depth={depth*1000:.1f}/{INSERT_DEPTH*1000:.0f}mm")
        send_feedback("script insert timed out")
        return False
