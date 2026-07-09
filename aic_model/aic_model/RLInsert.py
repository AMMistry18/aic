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

from aic_model.policy import Policy
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.time import Time
from tf2_ros import TransformException

# ----------------------------- configuration -------------------------------
MODEL_PATH = os.environ.get("RL_INSERT_MODEL", "/models/final_insert_sfp_contractA_v1.ts")
PORT_POSE_TOPIC = os.environ.get("RL_INSERT_PORT_TOPIC", "/aic_perception/port_pose")
STATIC_PORT_POSE = os.environ.get("RL_INSERT_PORT_POSE", "")  # "x,y,z,qw,qx,qy,qz"
PORT_POSE_STALE_SEC = 5.0   # perception older than this at start => refuse

# Action scaling: net outputs tanh in [-1,1]^6; deploy multiplies by these.
POS_SCALE = np.array([0.0015, 0.0015, 0.0035])   # m   (1.5, 1.5, 3.5 mm)
ROT_SCALE = np.array([0.08, 0.08, 0.12])         # rad

STEP_DT = 0.10          # s between RL steps
MAX_RL_STEPS = 400      # step budget for the last inch
INSERT_DEPTH = 0.045    # m, mouth -> seated
HANDOFF_MAX_DIST = 0.12  # m; farther than this from the mouth => macro failed

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

# VERIFY: home joint configuration used for obs[0:6] = joint_pos - home.
# Override via env: RL_INSERT_HOME_QPOS="q1,q2,q3,q4,q5,q6"
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


class RLInsert(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        log = self.get_logger()

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

        # Perception input: port pose in base_link.
        self._port_pose = None       # (pos(3), quat wxyz(4))
        self._port_pose_t = 0.0      # monotonic receive time
        if STATIC_PORT_POSE:
            v = np.array([float(x) for x in STATIC_PORT_POSE.split(",")])
            assert v.size == 7, "RL_INSERT_PORT_POSE must be x,y,z,qw,qx,qy,qz"
            self._port_pose = (v[:3], v[3:] / np.linalg.norm(v[3:]))
            self._port_pose_t = float("inf")
            log.warn(f"[rl] using STATIC port pose {v} (bench mode)")
        else:
            self._port_sub = parent_node.create_subscription(
                PoseStamped, PORT_POSE_TOPIC, self._port_pose_cb, 10)
            log.info(f"[rl] subscribed to perception port pose on {PORT_POSE_TOPIC}")

        self._home_qpos = None
        if HOME_QPOS_ENV:
            self._home_qpos = np.array([float(v) for v in HOME_QPOS_ENV.split(",")])
            assert self._home_qpos.shape == (6,)
            log.info(f"[rl] home qpos from env: {self._home_qpos}")
        else:
            log.warn("[rl] RL_INSERT_HOME_QPOS not set -- will use first observed "
                     "joint state as 'home'. VERIFY against training before scoring.")

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
        t_pt, q_pt = GRASP_TCP_IN_PLUG[plug_type]
        R_pt = _q_to_R(*q_pt)
        R_tcp = _q_to_R(*tcp_quat)
        R_tip = R_tcp @ R_pt.T
        tip_pos = tcp_pos - R_tip @ t_pt
        return tip_pos, R_tip

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
        o = np.zeros(69, dtype=np.float32)
        R_port = _q_to_R(*port_quat)
        # training basis: columns [lat_x, lat_y, insert_axis]
        Rp = np.column_stack([
            R_port[:, PORT_LAT_X_COL],
            R_port[:, PORT_LAT_Y_COL],
            PORT_INSERT_SIGN * R_port[:, PORT_INSERT_COL],
        ])

        jp = self._joint_vec(obs_msg.joint_states, "position")
        jv = self._joint_vec(obs_msg.joint_states, "velocity")
        o[0:6] = jp - self._home_qpos
        o[6:12] = jv
        o[12:15] = tcp_pos
        o[15:19] = tcp_quat if QUAT_ORDER_WXYZ else np.roll(tcp_quat, -1)

        # tcp velocity in port frame (finite difference)
        if prev_tcp is not None and dt > 1e-4:
            v_lin = (tcp_pos - prev_tcp[0]) / dt
            R_prev = _q_to_R(*prev_tcp[1])
            R_cur = _q_to_R(*tcp_quat)
            v_ang = _R_to_axis_angle(R_cur @ R_prev.T) / dt
            o[19:22] = Rp.T @ v_lin
            o[22:25] = Rp.T @ v_ang

        o[25:28] = port_pos
        o[28:32] = port_quat if QUAT_ORDER_WXYZ else np.roll(port_quat, -1)

        # delta_port = Rp.T @ (tip - X); X = port MOUTH (training convention)
        X = port_pos if DELTA_REF_IS_MOUTH else port_pos + Rp[:, 2] * INSERT_DEPTH
        delta = Rp.T @ (tip_pos - X)
        o[32:35] = delta

        # rot error of tip vs port, expressed in port frame
        rot_err = _R_to_axis_angle(Rp.T @ R_tip)
        o[35:38] = rot_err

        hint = np.clip(np.concatenate([
            [-6.0 * delta[0], -6.0 * delta[1], -8.0 * delta[2]],
            -3.0 * rot_err,
        ]), -1.0, 1.0)
        o[38:44] = hint
        o[44:50] = hint          # scripted_hint mirrors hint at deploy
        o[50] = 1.0

        w = obs_msg.wrist_wrench.wrench
        o[51:57] = WRENCH_SCALE * np.array([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ])
        o[57:63] = last_action

        # tip x and z axes expressed in the port frame
        o[63:66] = Rp.T @ R_tip[:, 0]
        o[66:69] = Rp.T @ R_tip[:, 2]
        return o

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
        plug_type = (task.plug_type or "sfp").lower()
        log.info(f"[rl] task: cable={task.cable_name} plug={task.plug_name} "
                 f"({plug_type}) port={task.port_name} module={task.target_module_name}")
        if plug_type not in GRASP_TCP_IN_PLUG:
            log.error(f"[rl] no grasp transform for plug_type={plug_type!r}")
            return False

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

        if self._home_qpos is None:
            self._home_qpos = self._joint_vec(obs.joint_states, "position")
            log.warn(f"[rl] using first joint state as home: {self._home_qpos}")

        # --- handoff sanity: are we inside the training envelope (~72 mm)?
        tcp_pos, tcp_quat = self._tcp()
        tip_pos, R_tip = self._tip_from_tcp(tcp_pos, tcp_quat, plug_type)
        dist = float(np.linalg.norm(tip_pos - port_pos))
        log.info(f"[rl] handoff check: |tip-mouth| = {dist*1000:.1f} mm")
        if dist > HANDOFF_MAX_DIST:
            log.error(f"[rl] tip is {dist*1000:.0f} mm from the port mouth -- "
                      "outside the last-inch envelope. The upstream process must "
                      "hand off within ~72 mm. Aborting so the failure is visible.")
            return False

        # ------------------------------ RL loop ------------------------------
        send_feedback("rl last-inch")
        R_port = _q_to_R(*port_quat)
        Rp = np.column_stack([
            R_port[:, PORT_LAT_X_COL], R_port[:, PORT_LAT_Y_COL],
            PORT_INSERT_SIGN * R_port[:, PORT_INSERT_COL]])
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
            act = self._forward(o)
            if step == 0 or step % 50 == 0:
                # Full obs dump: diff this against a MuJoCo rollout at the same
                # pose to root-cause any train/deploy convention mismatch.
                log.info("[rl] obs69=" + np.array2string(
                    o, precision=5, max_line_width=1000, separator=","))
            last_action = act
            prev_tcp = (tcp_pos, tcp_quat)
            prev_t = now

            # port-frame deltas -> base frame, applied to the COMMANDED pose
            dp = Rp @ (act[:3] * POS_SCALE)
            dR = _axis_angle_to_R(Rp @ (act[3:] * ROT_SCALE))
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
                log.info(f"[rl] step {step}: depth={depth*1000:.1f}/"
                         f"{INSERT_DEPTH*1000:.0f} mm lateral={lateral*1000:.1f} mm")

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
