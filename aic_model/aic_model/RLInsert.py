#
# RLInsert -- aic_model Policy wrapping the Contract-A last-inch insertion RL
# network (TorchScript, 69-dim observation -> 6-dim tanh cartesian residual).
#
# Deployment assumptions (per team decision):
#   * The process hands off at the START OF THE LAST-INCH ENVELOPE (~72 mm
#     retract span, tip ~26 mm outside the port mouth) -- no approach phase.
#   * NO ground-truth TF for objects. Object poses come from:
#       - port pose:  PERCEPTION, as geometry_msgs/PoseStamped in base_link on
#         a topic (default /aic_perception/port_pose, env RL_INSERT_PORT_TOPIC)
#         or a static env override RL_INSERT_PORT_POSE="x,y,z,qw,qx,qy,qz".
#       - plug tip pose: computed from the robot's own gripper/tcp TF composed
#         with the fixed grasp transform published in the task description
#         ("Reference Grasp Poses": gripper/tcp relative to the plug link).
#
# Module path for aic_model:  -p policy:=aic_model.RLInsert
#
# --------------------------------------------------------------------------
# Items tagged VERIFY must be diffed against RL/teacher/student_env_a.py
# before trusting a scored run -- the known Gazebo "lateral jam" bug lives in
# exactly these conventions.
# --------------------------------------------------------------------------
import os
import time
import traceback

import numpy as np

from aic_example_policies.ros.PerceptionInsert import PerceptionInsert
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.time import Time
from tf2_ros import TransformException

from .rl_insert_contract import (
    DEPLOY_POS_SCALE as POS_SCALE,
    DEPLOY_ROT_SCALE as ROT_SCALE,
    HOME_QPOS as AIC_HOME_QPOS,
    build_observation69,
    deploy_action_delta,
    port_frame,
    sfp_tip_pose_from_tcp,
    tcp_pose_for_sfp_tip,
)

# ----------------------------- configuration -------------------------------
MODEL_PATH = os.environ.get("RL_INSERT_MODEL", "/models/final_insert_sfp_contractA_v1.ts")
PORT_POSE_TOPIC = os.environ.get("RL_INSERT_PORT_TOPIC", "/aic_perception/port_pose")
STATIC_PORT_POSE = os.environ.get("RL_INSERT_PORT_POSE", "")  # "x,y,z,qw,qx,qy,qz"
PORT_SOURCE = os.environ.get("RL_INSERT_PORT_SOURCE", "perception").strip().lower()
PORT_POSE_STALE_SEC = 5.0   # perception older than this at start => refuse
MAX_PORT_REPROJ_PX = float(os.environ.get("RL_INSERT_MAX_REPROJ_PX", "25.0"))

# Controlled no-training validation knobs.  The normal submission path can
# disable prepositioning once an upstream macro owns the handoff.
PREPOSITION_HANDOFF = os.environ.get("RL_INSERT_PREPOSITION", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
HANDOFF_GAP_M = float(os.environ.get("RL_INSERT_HANDOFF_GAP_M", "0.026"))
HANDOFF_LATERAL_SIGMA_M = float(
    os.environ.get("RL_INSERT_HANDOFF_LATERAL_SIGMA_M", "0.002")
)
HANDOFF_AXIAL_SIGMA_M = float(os.environ.get("RL_INSERT_HANDOFF_AXIAL_SIGMA_M", "0.003"))
HANDOFF_ROT_SIGMA_RAD = float(os.environ.get("RL_INSERT_HANDOFF_ROT_SIGMA_RAD", "0.025"))
HANDOFF_SEED = int(os.environ.get("RL_INSERT_HANDOFF_SEED", "0"))

# raw: original Contract-A behavior; zero: isolate pose/action contract;
# baseline: subtract the six-axis wrench observed before the handoff.
WRENCH_MODE = os.environ.get("RL_INSERT_WRENCH_MODE", "raw").strip().lower()

STEP_DT = 0.10          # s between RL steps
MAX_RL_STEPS = int(os.environ.get("RL_INSERT_STEPS", "400"))
INSERT_DEPTH = 0.045    # m, mouth -> seated
HANDOFF_MAX_DIST = 0.12  # m; farther than this from the mouth => macro failed
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
SAFETY_MAX_LATERAL_M = float(os.environ.get("RL_INSERT_SAFETY_MAX_LATERAL_M", "0.012"))
SAFETY_MAX_RETREAT_M = float(os.environ.get("RL_INSERT_SAFETY_MAX_RETREAT_M", "0.015"))

# Impedance mirroring training (CartesianImpedanceAction, kp~100):
STIFFNESS = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

# Grasp transforms from the Phase-1 task description ("Reference Grasp Poses"):
# pose of gripper/tcp EXPRESSED IN the plug link frame. Quaternions given as
# xyzw in the docs; stored here as wxyz.
GRASP_TCP_IN_PLUG = {
    # plug_type: (translation xyz [m], quaternion wxyz)
    "sfp": (np.array([0.000, 0.027, -0.000]),
            np.array([-0.822, -0.568, 0.023, -0.015])),
    "sc":  (np.array([-0.000, -0.005, 0.001]),
            np.array([0.170, 0.685, -0.158, 0.691])),
}

# Pose of the MuJoCo SFP tip body expressed in gripper/tcp.  This is the
# transform consumed by student_env_a.py: q_tip/x_tip come from sfp_tip_link,
# while q_gripper/x_gripper come from gripper_tcp.  It is obtained by composing
# the exported tool<->plug weld with lc_plug->sfp_module->sfp_tip and removing
# the gripper_tcp site offset.  Do not replace it with a task "grasp pose": that
# pose names a different plug frame and previously produced a ~29 mm / ~89 deg
# observation error at an otherwise aligned handoff.
# Home joint configuration used for obs[0:6] = joint_pos - home.
# Override via env: RL_INSERT_HOME_QPOS="q1,q2,q3,q4,q5,q6".
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

# VERIFY: quaternion component order fed to the net. MuJoCo = wxyz.
QUAT_ORDER_WXYZ = True

# VERIFY: which columns of the port rotation matrix are (lat_x, lat_y, insert_axis).
# Training: scene._cart_frame_R = [_lat_x, _lat_y, _insert_axis].
PORT_LAT_X_COL, PORT_LAT_Y_COL, PORT_INSERT_COL = 0, 1, 2
PORT_INSERT_SIGN = +1.0  # +1 if the port frame axis points INTO the port

# VERIFY (the unresolved handoff question): X reference for delta_port.
# Training used the port MOUTH (scene._port_pos).
DELTA_REF_IS_MOUTH = True

# VERIFY: wrench frame/scale. obs[51:57] = wrench * 0.1, training sensor frame.
WRENCH_SCALE = 0.1

# Success/abort thresholds for the reconstructed loop tail (see note above
# _run): seat tolerance on depth, lateral gate at seat, and a sustained-force
# abort protecting the Tier-2 wrench budget (scoring threshold is 20 N).
SEAT_DEPTH_TOL = 0.002   # m
SEAT_LATERAL_MAX = 0.005  # m
FORCE_ABORT_N = 25.0
FORCE_ABORT_STEPS = 8    # consecutive steps above FORCE_ABORT_N => abort
# ---------------------------------------------------------------------------


def _q_to_R(qw, qx, qy, qz):
    """Rotation matrix from quaternion (w, x, y, z)."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-12
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_axis_angle(R):
    """3-vector axis*angle from rotation matrix."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= (2.0 * np.sin(angle) + 1e-12)
    return axis * angle


def _axis_angle_to_R(v):
    angle = np.linalg.norm(v)
    if angle < 1e-10:
        return np.eye(3)
    k = v / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _R_to_q(R):
    """Quaternion (w, x, y, z) from rotation matrix."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        q = np.array([w, x, y, z])
    else:
        # 180-degree rotation: pick the dominant diagonal
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        v = np.zeros(3)
        v[i] = s / 4.0
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = np.array([(R[k, j] - R[j, k]) / s, *v])
    return q / (np.linalg.norm(q) + 1e-12)


class RLInsert(PerceptionInsert):
    def _load_final_insert_policy(self):
        """PerceptionInsert hook: RLInsert owns the only final-insert network."""
        self._final_insert_policy = None
        self._final_insert_policy_kind = None

    def __init__(self, parent_node):
        super().__init__(parent_node)
        log = self.get_logger()
        if WRENCH_MODE not in ("raw", "zero", "baseline"):
            raise ValueError("RL_INSERT_WRENCH_MODE must be raw, zero, or baseline")
        if PORT_SOURCE not in ("perception", "external"):
            raise ValueError("RL_INSERT_PORT_SOURCE must be perception or external")
        self._wrench_baseline = np.zeros(6, dtype=np.float64)
        self._handoff_rng = np.random.default_rng(HANDOFF_SEED)

        # Load torch + model HERE so failures reject goals visibly instead of
        # dying silently inside the insert_cable thread.
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

        # Port pose in base_link.  A static override is bench-only; otherwise
        # use the bundled YOLO/multiview estimator unless an external publisher
        # was explicitly requested.
        self._port_pose = None       # (pos(3), quat wxyz(4))
        self._port_pose_t = 0.0      # monotonic receive time
        if STATIC_PORT_POSE:
            v = np.array([float(x) for x in STATIC_PORT_POSE.split(",")])
            assert v.size == 7, "RL_INSERT_PORT_POSE must be x,y,z,qw,qx,qy,qz"
            self._port_pose = (v[:3], v[3:] / np.linalg.norm(v[3:]))
            self._port_pose_t = float("inf")
            log.warn(f"[rl] using STATIC port pose {v} (bench mode)")
        elif PORT_SOURCE == "external":
            self._port_sub = parent_node.create_subscription(
                PoseStamped, PORT_POSE_TOPIC, self._port_pose_cb, 10)
            log.info(f"[rl] subscribed to perception port pose on {PORT_POSE_TOPIC}")
        else:
            log.info("[rl] using bundled SFP YOLO + multiview port perception")

        self._home_qpos = None
        if HOME_QPOS_ENV:
            self._home_qpos = np.array([float(v) for v in HOME_QPOS_ENV.split(",")])
            assert self._home_qpos.shape == (6,)
            log.info(f"[rl] home qpos from env: {self._home_qpos}")
        else:
            self._home_qpos = AIC_HOME_QPOS.copy()
            log.info(f"[rl] using training home qpos: {self._home_qpos}")

    # ------------------------------------------------------------- callbacks
    def _port_pose_cb(self, msg: PoseStamped):
        if msg.header.frame_id not in ("", "base_link"):
            self.get_logger().warn(
                f"[rl] port pose frame_id={msg.header.frame_id!r}; expected base_link",
                throttle_duration_sec=5.0)
        p, o = msg.pose.position, msg.pose.orientation
        self._port_pose = (np.array([p.x, p.y, p.z]),
                           np.array([o.w, o.x, o.y, o.z]))
        self._port_pose_t = time.monotonic()

    # ------------------------------------------------------------------ util
    def _forward(self, obs69):
        with self._torch.no_grad():
            out = self._model(self._torch.from_numpy(obs69.astype(np.float32)))
        return np.clip(out.detach().cpu().numpy().reshape(-1), -1.0, 1.0)

    def _tcp(self):
        t = self._parent_node._tf_buffer.lookup_transform(
            "base_link", "gripper/tcp", Time())
        tr, ro = t.transform.translation, t.transform.rotation
        return (np.array([tr.x, tr.y, tr.z]),
                np.array([ro.w, ro.x, ro.y, ro.z]))  # wxyz

    def _tip_from_tcp(self, tcp_pos, tcp_quat, plug_type):
        """Plug tip pose from the robot's own TCP + fixed grasp transform.

        Task description gives T_plug<-tcp (pose of gripper/tcp expressed in
        the plug link frame). Then:
            T_base<-plug = T_base<-tcp * inverse(T_plug<-tcp)
        """
        R_tcp = _q_to_R(*tcp_quat)
        if plug_type == "sfp":
            return sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)

        # Legacy SC path.  The bundled Contract-A artifact is SFP-only, but
        # retain the old conversion until SC gets its own measured contract.
        t_pt, q_pt = GRASP_TCP_IN_PLUG[plug_type]
        R_pt = _q_to_R(*q_pt)
        R_tip = R_tcp @ R_pt.T
        tip_pos = tcp_pos - R_tip @ t_pt
        return tip_pos, R_tip

    def _tcp_target_for_tip(self, tip_pos, R_tip):
        """Return a base-frame TCP Pose for a requested SFP tip pose."""
        tcp_pos, q_tcp = tcp_pose_for_sfp_tip(tip_pos, R_tip)
        return Pose(
            position=Point(x=float(tcp_pos[0]), y=float(tcp_pos[1]), z=float(tcp_pos[2])),
            orientation=Quaternion(
                w=float(q_tcp[0]), x=float(q_tcp[1]),
                y=float(q_tcp[2]), z=float(q_tcp[3])),
        )

    def _preposition_handoff(self, move_robot, send_feedback, port_pos, port_quat):
        """Move the attached SFP to a randomized last-inch handoff pose."""
        Rp = port_frame(port_quat)
        lateral = np.clip(
            self._handoff_rng.normal(0.0, HANDOFF_LATERAL_SIGMA_M, size=2),
            -0.004, 0.004,
        )
        gap = float(np.clip(
            HANDOFF_GAP_M + self._handoff_rng.normal(0.0, HANDOFF_AXIAL_SIGMA_M),
            0.018, 0.040,
        ))
        rotvec_port = np.clip(
            self._handoff_rng.normal(0.0, HANDOFF_ROT_SIGMA_RAD, size=3),
            -0.06, 0.06,
        )
        R_tip = Rp @ _axis_angle_to_R(rotvec_port)

        self.get_logger().info(
            "[rl] randomized handoff target | "
            f"gap_mm={gap*1000:.1f} lateral_mm={(lateral*1000).round(2).tolist()} "
            f"rot_deg={np.degrees(rotvec_port).round(2).tolist()}"
        )
        send_feedback("perception-guided RL handoff")
        # Stage along the retract axis so the final approach is straight and the
        # cable/controller have time to settle before the policy takes over.
        for stage_gap, hold_s in ((0.12, 3.0), (0.065, 2.5), (gap, 2.5)):
            tip_target = (
                np.asarray(port_pos, dtype=np.float64)
                + Rp[:, 0] * lateral[0]
                + Rp[:, 1] * lateral[1]
                - Rp[:, 2] * stage_gap
            )
            self.set_pose_target(
                move_robot,
                self._tcp_target_for_tip(tip_target, R_tip),
                stiffness=STIFFNESS,
                damping=DAMPING,
            )
            self.sleep_for(hold_s)

    def _joint_vec(self, joint_state, attr):
        vals = dict(zip(joint_state.name, getattr(joint_state, attr)))
        try:
            return np.array([vals[j] for j in JOINT_ORDER])
        except KeyError:
            arr = np.array(list(getattr(joint_state, attr)))
            return arr[:6] if arr.size >= 6 else np.zeros(6)

    # ------------------------------------------------------- obs construction
    def _build_obs(self, obs_msg, tcp_pos, tcp_quat, tip_pos, R_tip,
                   port_pos, port_quat, prev_tcp, dt, last_action):
        """69-dim observation, mirroring RL/teacher/student_env_a.py layout."""
        jp = self._joint_vec(obs_msg.joint_states, "position")
        jv = self._joint_vec(obs_msg.joint_states, "velocity")
        v_lin = np.zeros(3, dtype=np.float64)
        v_ang = np.zeros(3, dtype=np.float64)
        if prev_tcp is not None and dt > 1e-4:
            v_lin = (tcp_pos - prev_tcp[0]) / dt
            R_prev = _q_to_R(*prev_tcp[1])
            R_cur = _q_to_R(*tcp_quat)
            v_ang = _R_to_axis_angle(R_cur @ R_prev.T) / dt

        w = obs_msg.wrist_wrench.wrench
        wrench = np.array([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ], dtype=np.float64)
        return build_observation69(
            joint_pos=jp,
            joint_vel=jv,
            tcp_pos=tcp_pos,
            tcp_quat=tcp_quat,
            tcp_linear_velocity_world=v_lin,
            tcp_angular_velocity_world=v_ang,
            port_pos=port_pos,
            port_quat=port_quat,
            tip_pos=tip_pos,
            tip_rotation=R_tip,
            wrench=wrench,
            last_action=last_action,
            home_qpos=self._home_qpos,
            wrench_mode=WRENCH_MODE,
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
        if plug_type not in GRASP_TCP_IN_PLUG:
            log.error(f"[rl] no grasp transform for plug_type={plug_type!r}")
            return False

        if plug_type != "sfp":
            log.error("[rl] Contract-A student is SFP-only")
            return False

        self._wait_for_stable_clock()
        self._wait_for_transform("base_link", "gripper/tcp", timeout_sec=8.0)
        self.sleep_for(1.0)

        # --- inputs ready? (observations + perception port pose)
        obs = None
        t0 = time.monotonic()
        while obs is None and time.monotonic() - t0 < 10.0:
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.2)
        if obs is None:
            log.error("[rl] no Observation in 10 s -- zenoh wiring broken")
            return False


        w0 = obs.wrist_wrench.wrench
        self._wrench_baseline = np.array([
            w0.force.x, w0.force.y, w0.force.z,
            w0.torque.x, w0.torque.y, w0.torque.z,
        ], dtype=np.float64)
        log.info(
            f"[rl] wrench mode={WRENCH_MODE} baseline="
            f"{np.round(self._wrench_baseline, 4).tolist()}"
        )
        log.info(
            f"[rl] deploy action adapter: sign={ACTION_SIGN.tolist()} "
            f"gain={ACTION_GAIN.tolist()}"
        )

        if not STATIC_PORT_POSE and PORT_SOURCE == "perception":
            perceived = self.perceive_port_position(task, obs)
            if perceived is None or self._last_port_quat_wxyz is None:
                log.error("[rl] bundled perception failed to produce an SFP port pose")
                return False
            reproj_px = getattr(self, "_last_port_reproj_px", None)
            if (reproj_px is None or not np.isfinite(reproj_px)
                    or reproj_px > MAX_PORT_REPROJ_PX):
                log.error(
                    "[rl] rejecting SFP perception pose: "
                    f"reprojection={reproj_px!r}px limit={MAX_PORT_REPROJ_PX:.1f}px"
                )
                return False
            port_pos = np.asarray(perceived[0], dtype=np.float64)
            port_quat = np.asarray(self._last_port_quat_wxyz, dtype=np.float64)
            port_quat /= max(float(np.linalg.norm(port_quat)), 1e-12)
            self._port_pose = (port_pos, port_quat)
            self._port_pose_t = time.monotonic()
            log.info(
                f"[rl] perceived port pose p={np.round(port_pos, 5).tolist()} "
                f"q_wxyz={np.round(port_quat, 5).tolist()} "
                f"reproj={reproj_px:.2f}px"
            )

        if PORT_SOURCE == "external" and not STATIC_PORT_POSE:
            t0 = time.monotonic()
            while self._port_pose is None and time.monotonic() - t0 < 15.0:
                self.sleep_for(0.2)
        if self._port_pose is None:
            log.error(f"[rl] no perception port pose on {PORT_POSE_TOPIC} in 15 s")
            return False
        if (time.monotonic() - self._port_pose_t) > PORT_POSE_STALE_SEC:
            log.warn("[rl] perception port pose is stale; proceeding with last value")
        port_pos, port_quat = self._port_pose
        log.info(f"[rl] port pose: p={np.round(port_pos, 4)} q={np.round(port_quat, 4)}")

        if PREPOSITION_HANDOFF:
            self._preposition_handoff(
                move_robot, send_feedback, port_pos, port_quat)

        # --- handoff sanity: are we inside the training envelope (~72 mm)?
        tcp_pos, tcp_quat = self._tcp()
        tip_pos, R_tip = self._tip_from_tcp(tcp_pos, tcp_quat, plug_type)
        dist = float(np.linalg.norm(tip_pos - port_pos))
        Rp = port_frame(port_quat)
        handoff_delta = Rp.T @ (tip_pos - port_pos)
        handoff_rot = _R_to_axis_angle(Rp.T @ R_tip)
        log.info(
            f"[rl] handoff check: |tip-mouth|={dist*1000:.1f}mm "
            f"delta_port_mm={(handoff_delta*1000).round(2).tolist()} "
            f"rot_err_deg={np.degrees(handoff_rot).round(2).tolist()}"
        )
        if dist > HANDOFF_MAX_DIST:
            log.error(f"[rl] tip is {dist*1000:.0f} mm from the port mouth -- "
                      "outside the last-inch envelope. The upstream process must "
                      "hand off within ~72 mm. Aborting so the failure is visible.")
            return False

        # ------------------------------ RL loop ------------------------------
        send_feedback("rl last-inch")
        insert_axis = Rp[:, 2]
        last_action = np.zeros(6, dtype=np.float32)
        prev_tcp = None
        prev_t = None
        cmd_pos, cmd_R = tcp_pos.copy(), _q_to_R(*tcp_quat)

        # NOTE: everything below the set_pose_target call is RECONSTRUCTED --
        # the emailed original was truncated after "depth". Diff against the
        # author's copy before a scored run.
        force_hot_steps = 0
        depth = float("nan")
        handoff_depth = float(handoff_delta[2])

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
            tip_pos, R_tip = self._tip_from_tcp(tcp_pos, tcp_quat, plug_type)

            # refresh perception if it keeps publishing (rigid board: optional)
            if self._port_pose is not None:
                port_pos, port_quat = self._port_pose

            now = time.monotonic()
            dt = (now - prev_t) if prev_t else STEP_DT
            o = self._build_obs(obs, tcp_pos, tcp_quat, tip_pos, R_tip,
                                port_pos, port_quat, prev_tcp, dt, last_action)
            raw_act = self._forward(o)
            act = np.clip(raw_act * ACTION_SIGN * ACTION_GAIN, -1.0, 1.0)
            if step == 0 or step % 50 == 0:
                # Full obs dump: diff this against a MuJoCo rollout at the same
                # pose to root-cause any train/deploy convention mismatch.
                log.info("[rl] obs69=" + np.array2string(
                    o, precision=5, max_line_width=1000, separator=","))
            # This field is the network's previous output.  ACTION_SIGN adapts
            # the deploy actuator convention and is deliberately not folded
            # back into the learned observation.
            last_action = raw_act
            prev_tcp = (tcp_pos, tcp_quat)
            prev_t = now

            # port-frame deltas -> base frame, applied to the COMMANDED pose
            dp, drot_world = deploy_action_delta(act, port_quat)
            dR = _axis_angle_to_R(drot_world)
            cmd_pos = cmd_pos + dp
            cmd_R = dR @ cmd_R
            q = _R_to_q(cmd_R)
            self.set_pose_target(
                move_robot,
                Pose(position=Point(x=cmd_pos[0], y=cmd_pos[1], z=cmd_pos[2]),
                     orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])),
                stiffness=STIFFNESS, damping=DAMPING)

            depth_vec = Rp.T @ (tip_pos - port_pos)
            depth = float(depth_vec[2])
            lateral = float(np.linalg.norm(depth_vec[:2]))
            if step % 10 == 0:
                dp_port = Rp.T @ dp
                log.info(
                    f"[rl] step {step}: depth={depth*1000:.1f}/"
                    f"{INSERT_DEPTH*1000:.0f}mm lateral={lateral*1000:.1f}mm "
                    f"raw_act={np.round(raw_act, 3).tolist()} "
                    f"deploy_act={np.round(act, 3).tolist()} "
                    f"dp_port_mm={(dp_port*1000).round(3).tolist()}"
                )

            if (lateral > SAFETY_MAX_LATERAL_M
                    or depth < handoff_depth - SAFETY_MAX_RETREAT_M):
                log.error(
                    "[rl] contract safety abort | "
                    f"depth_mm={depth*1000:.1f} initial_mm={handoff_depth*1000:.1f} "
                    f"lateral_mm={lateral*1000:.1f}"
                )
                send_feedback("rl contract safety abort")
                return False

            if depth >= INSERT_DEPTH - SEAT_DEPTH_TOL and lateral < SEAT_LATERAL_MAX:
                log.info(f"[rl] seated: depth={depth*1000:.1f} mm "
                         f"lateral={lateral*1000:.1f} mm after {step + 1} steps")
                send_feedback("rl insert seated")
                return True

            # Tier-2 wrench budget guard: sustained force => back off and abort
            # rather than grind (the scored threshold is 20 N).
            w = obs.wrist_wrench.wrench
            f_mag = float(np.linalg.norm([w.force.x, w.force.y, w.force.z]))
            force_hot_steps = force_hot_steps + 1 if f_mag > FORCE_ABORT_N else 0
            if force_hot_steps >= FORCE_ABORT_STEPS:
                log.error(f"[rl] sustained wrist force {f_mag:.1f} N over "
                          f"{FORCE_ABORT_STEPS} steps at depth={depth*1000:.1f} mm "
                          "-- jammed; retreating along the insert axis and aborting.")
                retreat = cmd_pos - insert_axis * 0.010
                q = _R_to_q(cmd_R)
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
