"""
PerceptionInsert.py

Perception-guided insertion. Debug build: saves camera screenshots at
perception time, start of descent, and end of descent so we can diagnose
the 3cm miss. Also fixes the Z formula and extends descent depth.

Z formula fix:
  Original: target_z = port_z + z_offset - plug_z
    → plug tip = gripper_z - plug_z = port_z + z_offset - 2*plug_z  (WRONG)
  Fixed:    target_z = port_z + z_offset + plug_z
    → plug tip = gripper_z - plug_z = port_z + z_offset  (CORRECT)

Descent extended to z_offset = -0.025 (25mm below port entrance).
"""
import csv
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

    def _gripper_xyz_from_tf(self):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z])
        except TransformException:
            return None

    def _fts_z(self, obs):
        w = getattr(obs, "wrist_wrench", None)
        if w is None:
            return 0.0
        return w.wrench.force.z

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

    def perceive_port_position(self, task, obs):
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

        return X, views

    # ── Debug viz ──────────────────────────────────────────────────────────

    def _save_viz(self, views, X, task, label, gripper_xyz=None):
        """
        RED   = reprojected triangulated port position
        GREEN = reprojected current gripper TCP
        CYAN  = reprojected estimated plug tip (gripper - plug_z in Z)
        Overlaid text shows XY error and tip-above-port in mm.
        """
        self._debug_counter += 1
        tid = self._debug_counter
        plug_z = 0.04 if task.port_type == "sc" else 0.042

        for cam, (bgr, K, T) in views.items():
            viz = bgr.copy()
            P = K @ T[:3, :4]

            # Red: port
            proj = P @ np.append(X, 1.0)
            if proj[2] > 0:
                u, v = int(proj[0]/proj[2]), int(proj[1]/proj[2])
                cv2.circle(viz, (u, v), 14, (0, 0, 255), 3)
                cv2.putText(viz, "PORT", (u+16, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if gripper_xyz is not None:
                # Green: TCP
                gproj = P @ np.append(gripper_xyz, 1.0)
                if gproj[2] > 0:
                    gu, gv = int(gproj[0]/gproj[2]), int(gproj[1]/gproj[2])
                    cv2.circle(viz, (gu, gv), 10, (0, 255, 0), 3)
                    cv2.putText(viz, "TCP", (gu+12, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Cyan: plug tip
                plug_tip = gripper_xyz.copy()
                plug_tip[2] -= plug_z
                tproj = P @ np.append(plug_tip, 1.0)
                if tproj[2] > 0:
                    tu, tv = int(tproj[0]/tproj[2]), int(tproj[1]/tproj[2])
                    cv2.circle(viz, (tu, tv), 8, (255, 255, 0), 3)
                    cv2.putText(viz, "TIP", (tu+10, tv), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                xy_err_mm = np.linalg.norm(gripper_xyz[:2] - X[:2]) * 1000
                tip_above_mm = (plug_tip[2] - X[2]) * 1000
                cv2.putText(viz, f"XY_err={xy_err_mm:.1f}mm  tip_above_port={tip_above_mm:.1f}mm",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.putText(viz, f"{label} T{tid} {cam}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(viz, f"port=({X[0]:.3f},{X[1]:.3f},{X[2]:.3f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            fname = f"{DEBUG_DIR}/t{tid:02d}_{label}_{cam}.png"
            cv2.imwrite(fname, viz)
            self.get_logger().info(f"Saved {fname}")

    # ── Build Transform ────────────────────────────────────────────────────

    def build_port_transform(self, X):
        gripper_tf = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
        q = gripper_tf.transform.rotation
        return Transform(
            translation=Vector3(x=float(X[0]), y=float(X[1]), z=float(X[2])),
            rotation=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w),
        )

    # ── calc_gripper_pose — Z formula fixed ────────────────────────────────

    def calc_gripper_pose(self, port_transform, slerp_fraction=1.0, position_fraction=1.0,
                          z_offset=0.1, reset_xy_integrator=False):
        q_diff = (1.0, 0.0, 0.0, 0.0)
        gripper_tf = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
        q_gripper = (gripper_tf.transform.rotation.w, gripper_tf.transform.rotation.x,
                     gripper_tf.transform.rotation.y, gripper_tf.transform.rotation.z)
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (gripper_tf.transform.translation.x,
                       gripper_tf.transform.translation.y,
                       gripper_tf.transform.translation.z)
        port_xy = (port_transform.translation.x, port_transform.translation.y)
        plug_z = 0.04 if self._task.port_type == "sc" else 0.042

        plug_tip_xyz = (gripper_xyz[0], gripper_xyz[1], gripper_xyz[2] - plug_z)
        tip_x_error = port_xy[0] - plug_tip_xyz[0]
        tip_y_error = port_xy[1] - plug_tip_xyz[1]

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

        # FIXED: plug tip target = port_z + z_offset
        # gripper TCP = plug tip + plug_z = port_z + z_offset + plug_z
        target_z = port_transform.translation.z + z_offset + plug_z

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
        self.get_logger().info(f"PerceptionInsert start | {task.port_type} {task.target_module_name}")
        self._task = task
        self.sleep_for(2.0)

        obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation")
            return False

        fts_baseline = self._fts_z(obs)
        self.get_logger().info(f"FTS baseline: {fts_baseline:.2f}N")

        # Perception with scan fallback
        scan_offsets = [
            (0.0, 0.0),
            (0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05),
            (0.07, 0.07), (-0.07, 0.07), (0.07, -0.07), (-0.07, -0.07),
        ]

        X = None
        views = None
        for i, (dx, dy) in enumerate(scan_offsets):
            if i > 0:
                g = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time()).transform
                scan_pose = Pose(
                    position=Point(x=g.translation.x+dx, y=g.translation.y+dy, z=g.translation.z),
                    orientation=Quaternion(x=g.rotation.x, y=g.rotation.y, z=g.rotation.z, w=g.rotation.w)
                )
                self.set_pose_target(move_robot=move_robot, pose=scan_pose)
                self.sleep_for(1.5)
                obs = get_observation()

            result = self.perceive_port_position(task, obs)
            if result is not None:
                X, views = result
                self.get_logger().info(f"Port found at scan offset ({dx},{dy}): {X.tolist()}")
                break
            self.get_logger().warn(f"Scan {i+1}/{len(scan_offsets)}: no detection at ({dx},{dy})")

        if X is None:
            self.get_logger().error("Perception failed")
            return False

        plug_z = 0.04 if task.port_type == "sc" else 0.042

        # Screenshot 1: at perception time
        g0 = self._gripper_xyz_from_tf()
        self._save_viz(views, X, task, "01_perception", gripper_xyz=g0)
        send_feedback(f"Port at ({X[0]:.3f},{X[1]:.3f},{X[2]:.3f})")

        # Log the Z formula result so we can verify it's correct
        self.get_logger().info(
            f"Z formula check: port_z={X[2]:.4f} plug_z={plug_z:.3f} "
            f"gripper_target@z_offset=0 = {X[2]+0+plug_z:.4f} "
            f"(plug tip will land exactly at port_z)"
        )

        port_transform = self.build_port_transform(X)

        # Interpolate to above port
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

        # Screenshot 2: at start of descent
        obs = get_observation()
        views2 = self._build_views(obs)
        g1 = self._gripper_xyz_from_tf()
        self._save_viz(views2, X, task, "02_descent_start", gripper_xyz=g1)
        if g1 is not None:
            self.get_logger().info(
                f"Descent start: gripper_z={g1[2]:.4f} plug_tip_z={g1[2]-plug_z:.4f} "
                f"port_z={X[2]:.4f} tip_above={( g1[2]-plug_z-X[2])*1000:.1f}mm "
                f"XY_err={np.linalg.norm(g1[:2]-X[:2])*1000:.1f}mm"
            )

        # CSV log: z_offset, gripper_z, plug_tip_z, port_z, fts, fts_delta
        csv_path = f"{DEBUG_DIR}/t{self._debug_counter:02d}_descent.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["z_offset", "gripper_z", "plug_tip_z", "port_z", "fts_z", "fts_delta"])

        # Descent — extended to -0.025 (25mm below port entrance)
        fts_stop = False
        step = 0
        while z_offset >= -0.025:
            z_offset -= 0.0005
            step += 1
            try:
                self.set_pose_target(move_robot=move_robot,
                                     pose=self.calc_gripper_pose(port_transform, z_offset=z_offset))
            except TransformException as ex:
                self.get_logger().warn(f"TF fail descent: {ex}")
            self.sleep_for(0.05)

            # Log every 10 steps (~5mm)
            if step % 10 == 0:
                obs = get_observation()
                fts = self._fts_z(obs)
                g = self._gripper_xyz_from_tf()
                gz = g[2] if g is not None else float("nan")
                tip_z = gz - plug_z
                delta = fts - fts_baseline
                csv_writer.writerow([f"{z_offset:.4f}", f"{gz:.4f}", f"{tip_z:.4f}",
                                     f"{X[2]:.4f}", f"{fts:.3f}", f"{delta:.3f}"])
                # Stop if force delta exceeds 18N (20N is the penalty line, give margin)
                if abs(delta) > 18.0:
                    self.get_logger().warn(
                        f"FTS {abs(delta):.1f}N > 18N limit at z_offset={z_offset:.4f}, stopping")
                    fts_stop = True
                    break

        csv_file.close()
        self.get_logger().info(f"Descent CSV: {csv_path}")

        # Screenshot 3: at end of descent
        obs = get_observation()
        views3 = self._build_views(obs)
        g2 = self._gripper_xyz_from_tf()
        self._save_viz(views3, X, task, "03_descent_end", gripper_xyz=g2)
        if g2 is not None:
            self.get_logger().info(
                f"Descent end: gripper_z={g2[2]:.4f} plug_tip_z={g2[2]-plug_z:.4f} "
                f"port_z={X[2]:.4f} tip_above={( g2[2]-plug_z-X[2])*1000:.1f}mm "
                f"XY_err={np.linalg.norm(g2[:2]-X[:2])*1000:.1f}mm fts_stop={fts_stop}"
            )

        self.sleep_for(3.0)
        self.get_logger().info("PerceptionInsert done")
        return True