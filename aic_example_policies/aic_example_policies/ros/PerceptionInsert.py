"""
PerceptionInsert.py

Full perception + insertion policy. Uses HSV blob detection for SC ports
and YOLO keypoint detection for NIC ports, triangulates the target port's
3D position from the three wrist cameras, then runs CheatCode's descent
loop using the perceived Transform instead of ground-truth TF.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from .perception_core import PerceptionCore

CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]
DEBUG_DIR = "/tmp/perception_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


def ros_image_to_cv2(img_msg):
    arr = np.frombuffer(img_msg.data, dtype=np.uint8)
    if img_msg.encoding == "mono8":
        return cv2.cvtColor(arr.reshape(img_msg.height, img_msg.width), cv2.COLOR_GRAY2BGR)
    arr = arr.reshape(img_msg.height, img_msg.width, 3)
    return arr.copy() if img_msg.encoding == "bgr8" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def tf_to_4x4(tf_msg):
    if hasattr(tf_msg, "transform"):
        tf_msg = tf_msg.transform
    t, q = tf_msg.translation, tf_msg.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class PerceptionInsert(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        self._debug_counter = 0
        weights = (Path(__file__).parent / "weights" / "best.pt").resolve()
        self.get_logger().info(f"Loading NIC weights from {weights}")
        self._pc = PerceptionCore(nic_weights=str(weights))

    # ── Observation helpers ────────────────────────────────────────────────

    def _get_cam_data(self, obs, cam_name):
        img_map = {"left_camera": obs.left_image, "center_camera": obs.center_image, "right_camera": obs.right_image}
        info_map = {"left_camera": obs.left_camera_info, "center_camera": obs.center_camera_info, "right_camera": obs.right_camera_info}
        img_msg, info_msg = img_map.get(cam_name), info_map.get(cam_name)
        if img_msg is None or info_msg is None:
            return None
        K = np.array(info_msg.k).reshape(3, 3)
        if K[0, 0] == 0:
            return None
        try:
            bgr = ros_image_to_cv2(img_msg)
        except Exception:
            return None
        return bgr, K

    def _lookup_cam_from_base(self, cam_name):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                f"{cam_name}/optical", "base_link", Time(), Duration(seconds=0.2)
            )
        except TransformException as e:
            self.get_logger().warn(f"{cam_name}: TF lookup failed: {e}")
            return None
        return tf_to_4x4(tf)

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

    def _closest_to_center(self, dets, img_w, img_h, kind):
        if not dets:
            return None
        cx, cy = img_w / 2.0, img_h / 2.0
        def pt(d):
            if kind == "sc":
                return d["centroid"]
            x1, y1, x2, y2 = d["bbox"]
            return ((x1+x2)/2.0, (y1+y2)/2.0)
        return min(dets, key=lambda d: (pt(d)[0]-cx)**2 + (pt(d)[1]-cy)**2)

    # ── Perception ─────────────────────────────────────────────────────────

    def perceive_port_position(self, task: Task, obs) -> np.ndarray | None:
        views = self._build_views(obs)
        if len(views) < 2:
            self.get_logger().error(f"Only {len(views)} cam views usable")
            return None

        if task.port_type == "sc":
            pts, projs = [], []
            for cam, (bgr, K, T) in views.items():
                blobs = self._pc.detect_sc(bgr)
                picked = self._closest_to_center(blobs, bgr.shape[1], bgr.shape[0], "sc")
                if picked is None:
                    self.get_logger().warn(f"{cam}: no SC blob")
                    continue
                pts.append(picked["centroid"])
                projs.append(self._pc.build_projection_matrix(K, T))
                self.get_logger().info(f"{cam}: SC centroid={picked['centroid']} area={picked['area']}")
            if len(pts) < 2:
                return None
            X = self._pc.triangulate(pts, projs)

        elif task.port_type == "sfp":
            kp_slice = slice(0, 4) if task.port_name == "sfp_port_0" else slice(4, 8)
            per_cam = {}
            for cam, (bgr, K, T) in views.items():
                nics = self._pc.detect_nic(bgr)
                picked = self._closest_to_center(nics, bgr.shape[1], bgr.shape[0], "nic")
                if picked is None:
                    self.get_logger().warn(f"{cam}: no NIC")
                    continue
                per_cam[cam] = (picked["kps"][kp_slice], self._pc.build_projection_matrix(K, T))
                self.get_logger().info(f"{cam}: NIC conf={picked['conf']:.2f}")
            if len(per_cam) < 2:
                return None
            cams = list(per_cam.keys())
            kp_3d = []
            for i in range(4):
                pts_2d = [tuple(per_cam[c][0][i]) for c in cams]
                Ps = [per_cam[c][1] for c in cams]
                kp_3d.append(self._pc.triangulate(pts_2d, Ps))
            X = np.array(kp_3d).mean(axis=0)
        else:
            self.get_logger().error(f"Unknown port_type {task.port_type}")
            return None

        self._save_debug_viz(views, X, task)
        return X

    def _save_debug_viz(self, views, X, task):
        self._debug_counter += 1
        tid = self._debug_counter
        for cam, (bgr, K, T) in views.items():
            viz = bgr.copy()
            blobs = self._pc.detect_sc(bgr) if task.port_type == "sc" else []
            for b in blobs:
                x, y, w, h = b["bbox"]
                cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 255), 2)
            P = K @ T[:3, :4]
            proj = P @ np.append(X, 1.0)
            if proj[2] > 0:
                u, v = proj[0]/proj[2], proj[1]/proj[2]
                cv2.circle(viz, (int(u), int(v)), 12, (0, 0, 255), 3)
                cv2.putText(viz, "REPROJ", (int(u)+15, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(viz, f"T{tid} {cam} {task.target_module_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(viz, f"3D ({X[0]:.3f},{X[1]:.3f},{X[2]:.3f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imwrite(f"{DEBUG_DIR}/trial{tid:02d}_{cam}.png", viz)

    # ── Build Transform from perception ────────────────────────────────────

    def build_port_transform(self, X: np.ndarray) -> Transform:
        """
        Position from triangulation, orientation from current gripper TCP.
        The gripper spawns pre-aligned to the port (per task doc), so its
        orientation is a valid starting estimate for the port orientation.
        """
        gripper_tf = self._parent_node._tf_buffer.lookup_transform(
            "base_link", "gripper/tcp", Time())
        q = gripper_tf.transform.rotation
        return Transform(
            translation=Vector3(x=float(X[0]), y=float(X[1]), z=float(X[2])),
            rotation=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w),
        )

    # ── CheatCode calc_gripper_pose (unchanged) ────────────────────────────

    def calc_gripper_pose(self, port_transform, slerp_fraction=1.0, position_fraction=1.0,
                          z_offset=0.1, reset_xy_integrator=False):
        q_port = (port_transform.rotation.w, port_transform.rotation.x,
                  port_transform.rotation.y, port_transform.rotation.z)
        q_diff = (1.0, 0.0, 0.0, 0.0)
        gripper_tf = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
        q_gripper = (gripper_tf.transform.rotation.w, gripper_tf.transform.rotation.x,
                     gripper_tf.transform.rotation.y, gripper_tf.transform.rotation.z)
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (gripper_tf.transform.translation.x, gripper_tf.transform.translation.y, gripper_tf.transform.translation.z)
        port_xy = (port_transform.translation.x, port_transform.translation.y)
        # Hardcoded plug offset along gripper z (SC=4cm, SFP=4.2cm below TCP)
        plug_z_offset = 0.04 if self._task.port_type == "sc" else 0.042
        plug_tip_gripper_offset = (0.0, 0.0, plug_z_offset)
        # Approximate plug tip position = gripper_xyz with z offset
        plug_xyz = (gripper_xyz[0], gripper_xyz[1], gripper_xyz[2] - plug_z_offset)
        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]
        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup, self._max_integrator_windup)
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup, self._max_integrator_windup)

        i_gain = 0.15
        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend = (
            position_fraction*target_x + (1.0-position_fraction)*gripper_xyz[0],
            position_fraction*target_y + (1.0-position_fraction)*gripper_xyz[1],
            position_fraction*target_z + (1.0-position_fraction)*gripper_xyz[2],
        )
        return Pose(
            position=Point(x=blend[0], y=blend[1], z=blend[2]),
            orientation=Quaternion(w=q_gripper_slerp[0], x=q_gripper_slerp[1],
                                   y=q_gripper_slerp[2], z=q_gripper_slerp[3]),
        )

    # ── Main ───────────────────────────────────────────────────────────────

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self.get_logger().info(f"DEBUG task fields: cable_name={task.cable_name} plug_name={task.plug_name} port_name={task.port_name}")
        self.get_logger().info(f"PerceptionInsert start | {task.port_type} {task.target_module_name}")
        self._task = task
        self.sleep_for(2.0)

        obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation")
            return False

        # XY scan pattern: center, then 4 cardinal offsets, then 4 diagonals
        scan_offsets = [
            (0.0, 0.0),       # initial pose
            (0.05, 0.0),      # +X
            (-0.05, 0.0),     # -X
            (0.0, 0.05),      # +Y
            (0.0, -0.05),     # -Y
            (0.07, 0.07),     # +X +Y diagonal
            (-0.07, 0.07),    # -X +Y diagonal
            (0.07, -0.07),    # +X -Y
            (-0.07, -0.07),   # -X -Y
        ]
        
        X = None
        for i, (dx, dy) in enumerate(scan_offsets):
            if i > 0:
                # Move to scan offset in base_link XY, keep same Z and orientation
                g = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time()).transform
                scan_pose = Pose(
                    position=Point(
                        x=g.translation.x + dx,
                        y=g.translation.y + dy,
                        z=g.translation.z  # keep height
                    ),
                    orientation=Quaternion(x=g.rotation.x, y=g.rotation.y, z=g.rotation.z, w=g.rotation.w)
                )
                self.set_pose_target(move_robot=move_robot, pose=scan_pose)
                self.sleep_for(1.5)
                obs = get_observation()
        
            X = self.perceive_port_position(task, obs)
            if X is not None:
                self.get_logger().info(f"Port found at scan offset ({dx},{dy})")
                break
            else:
                self.get_logger().warn(f"Scan {i+1}/{len(scan_offsets)}: no detection at offset ({dx},{dy})")
        
        if X is None:
            self.get_logger().error("Perception failed after XY scan")
            return False

        port_transform = self.build_port_transform(X)
        self.get_logger().info(f"Perceived port at {X.tolist()}")
        send_feedback(f"Perceived port at ({X[0]:.3f},{X[1]:.3f},{X[2]:.3f})")

        # Interpolate from current to above port
        z_offset = 0.2
        for t in range(0, 100):
            f = t / 100.0
            try:
                self.set_pose_target(move_robot=move_robot, pose=self.calc_gripper_pose(
                    port_transform, slerp_fraction=f, position_fraction=f,
                    z_offset=z_offset, reset_xy_integrator=True))
            except TransformException as ex:
                self.get_logger().warn(f"TF fail interp: {ex}")
            self.sleep_for(0.05)

        # Descent
        while z_offset >= -0.015:
            z_offset -= 0.0005
            try:
                self.set_pose_target(move_robot=move_robot,
                                     pose=self.calc_gripper_pose(port_transform, z_offset=z_offset))
            except TransformException as ex:
                self.get_logger().warn(f"TF fail descent: {ex}")
            self.sleep_for(0.05)

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(5.0)
        self.get_logger().info("PerceptionInsert done")
        return True