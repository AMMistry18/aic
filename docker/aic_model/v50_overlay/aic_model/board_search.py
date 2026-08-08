"""Dark taskboard detection and bounded camera framing for :mod:`CableInsertionPolicy`.

The detector is deliberately independent of ROS so it can be tuned from saved
camera frames with ``python board_search.py IMAGE``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# OpenCV HSV values are in [0, 255] for saturation and value.
BOARD_V_MIN = 15
BOARD_V_MAX = 95
BOARD_S_MAX = 80
MIN_BLOB_AREA_FRAC = 0.03
MIN_SOLIDITY = 0.55
MORPH_KERNEL_PX = 9
BORDER_MARGIN_PX = 2

PROBE_M = 0.02
CENTER_TOL_FRAC = 0.08
MAX_MOVE_M = 0.05
RAISE_STEP_M = 0.025
MAX_MOVES = 3
SETTLE_SEC = 0.7
MOTION_SETTLE_TIMEOUT_SEC = 3.0
MOTION_SETTLE_TOL_M = 0.003
FRESH_IMAGE_TIMEOUT_SEC = 2.0
POLL_SEC = 0.10
MAX_JACOBIAN_CONDITION = 100.0
MIN_TCP_Z_M = 0.05
MAX_TCP_Z_M = 1.50


class BoardSearch:
    """Move the arm until a whole dark taskboard is framed (at most 3 moves)."""

    def __init__(self, policy):
        self.p = policy

    @staticmethod
    def _detect(bgr):
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("bgr must be an HxWx3 numpy image")
        if bgr.size == 0:
            raise ValueError("bgr must not be empty")

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, BOARD_V_MIN], dtype=np.uint8),
            np.array([179, BOARD_S_MAX, BOARD_V_MAX], dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(bgr.shape[0] * bgr.shape[1])
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = float(w * h)
            area_frac = area / image_area
            solidity = area / rect_area if rect_area else 0.0
            if area_frac < MIN_BLOB_AREA_FRAC or solidity < MIN_SOLIDITY:
                continue
            # Area is primary (the board should dominate); solidity breaks close
            # calls in favor of the compact plate rather than a ragged shadow.
            candidates.append((area * solidity, area, contour, (x, y, w, h)))

        if not candidates:
            return (False, None, None, 0.0, None, False, mask, None)

        _, area, contour, bbox = max(candidates, key=lambda item: item[:2])
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            return (False, None, None, 0.0, None, False, mask, None)
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        x, y, w, h = bbox
        image_h, image_w = bgr.shape[:2]
        touches_border = (
            x <= BORDER_MARGIN_PX
            or y <= BORDER_MARGIN_PX
            or x + w >= image_w - BORDER_MARGIN_PX
            or y + h >= image_h - BORDER_MARGIN_PX
        )
        return (
            True,
            cx,
            cy,
            area / image_area,
            bbox,
            touches_border,
            mask,
            contour,
        )

    def detect_board(self, bgr):
        """Return board detection data for one BGR image.

        The tuple is ``(found, cx, cy, area_frac, bbox, touches_border, mask)``.
        ``bbox`` is ``(x, y, width, height)`` and centroid coordinates are pixels.
        """
        return self._detect(bgr)[:7]

    @staticmethod
    def _is_success(detection, image_shape):
        found, cx, cy, _, _, touches_border, _ = detection
        if not found or touches_border:
            return False
        height, width = image_shape[:2]
        return (
            abs(cx - width / 2.0) <= CENTER_TOL_FRAC * width
            and abs(cy - height / 2.0) <= CENTER_TOL_FRAC * height
        )

    def _detect_all(self, obs):
        detections = []
        for cam_name in ("left_camera", "center_camera", "right_camera"):
            cam_data = self.p._get_cam_data(obs, cam_name)
            if cam_data is None:
                continue
            bgr, _ = cam_data
            detection = self.detect_board(bgr)
            if detection[0]:
                detections.append((detection[3], cam_name, bgr, detection))
        return max(detections, default=None, key=lambda item: item[0])

    @staticmethod
    def _image_stamp_ns(obs, cam_name):
        image = getattr(obs, f"{cam_name.replace('_camera', '')}_image", None)
        header = getattr(image, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return None
        sec = getattr(stamp, "sec", None)
        nanosec = getattr(stamp, "nanosec", None)
        if sec is None or nanosec is None:
            return None
        return int(sec) * 1_000_000_000 + int(nanosec)

    def _observe_camera(self, get_observation, cam_name, newer_than=None):
        """Return a post-motion image, not the cached frame from before a probe."""
        deadline = time.monotonic() + FRESH_IMAGE_TIMEOUT_SEC
        while True:
            obs = get_observation()
            if obs is not None:
                cam_data = self.p._get_cam_data(obs, cam_name)
                if cam_data is not None:
                    bgr, _ = cam_data
                    stamp = self._image_stamp_ns(obs, cam_name)
                    if newer_than is None or stamp is None or stamp > newer_than:
                        return bgr, self.detect_board(bgr), stamp
            if time.monotonic() >= deadline:
                return None
            self.p.sleep_for(POLL_SEC)

    def _move_by(self, move_robot, delta):
        from geometry_msgs.msg import Point, Pose, Quaternion

        delta = np.asarray(delta, dtype=np.float64)
        magnitude = float(np.linalg.norm(delta))
        if magnitude > MAX_MOVE_M:
            delta *= MAX_MOVE_M / magnitude
        tcp_pos, tcp_quat = self.p._tcp()
        target = np.asarray(tcp_pos, dtype=np.float64) + delta
        target[2] = np.clip(target[2], MIN_TCP_Z_M, MAX_TCP_Z_M)
        pose = Pose(
            position=Point(x=float(target[0]), y=float(target[1]), z=float(target[2])),
            orientation=Quaternion(
                w=float(tcp_quat[0]), x=float(tcp_quat[1]),
                y=float(tcp_quat[2]), z=float(tcp_quat[3]),
            ),
        )
        self.p.set_pose_target(move_robot, pose, frame_id="base_link")
        deadline = time.monotonic() + MOTION_SETTLE_TIMEOUT_SEC
        actual = np.asarray(tcp_pos, dtype=np.float64)
        while True:
            try:
                actual, _ = self.p._tcp()
                actual = np.asarray(actual, dtype=np.float64)
            except Exception:
                actual = np.asarray(tcp_pos, dtype=np.float64)
            if float(np.linalg.norm(target - actual)) <= MOTION_SETTLE_TOL_M:
                self.p.sleep_for(SETTLE_SEC)
                return actual - np.asarray(tcp_pos, dtype=np.float64)
            if time.monotonic() >= deadline:
                self.p.get_logger().warn(
                    f"[board_search] TCP did not settle within {MOTION_SETTLE_TIMEOUT_SEC:.1f}s; "
                    f"remaining_error={np.linalg.norm(target - actual):.4f}m"
                )
                return actual - np.asarray(tcp_pos, dtype=np.float64)
            self.p.sleep_for(POLL_SEC)

    def _camera_axes_in_base(self, cam_name):
        """Return the selected camera's image-plane and back-away axes in base_link.

        ``_lookup_cam_from_base`` returns camera-from-base, so its transpose maps
        a unit camera-frame direction to base_link.  The wrist cameras move rigidly
        with the TCP while this routine keeps orientation fixed.
        """
        T_cam_from_base = self.p._lookup_cam_from_base(cam_name)
        if T_cam_from_base is None:
            return None
        R_cam_from_base = np.asarray(T_cam_from_base, dtype=np.float64)[:3, :3]
        if R_cam_from_base.shape != (3, 3):
            return None
        R_base_from_cam = R_cam_from_base.T
        image_u = R_base_from_cam[:, 0]
        image_v = R_base_from_cam[:, 1]
        # Camera +Z points toward the observed scene. Move opposite it to
        # increase standoff when the board overflows the image.
        back_away = -R_base_from_cam[:, 2]
        if min(np.linalg.norm(image_u), np.linalg.norm(image_v), np.linalg.norm(back_away)) < 0.9:
            return None
        return image_u, image_v, back_away

    def run(self, get_observation, move_robot):
        """Frame the whole board with two XY probes and one correction."""
        log = self.p.get_logger()
        obs = get_observation()
        if obs is None:
            log.error("[board_search] no observation available")
            return False
        chosen = self._detect_all(obs)
        if chosen is None:
            log.error("[board_search] no valid dark board blob in any camera")
            return False

        _, cam_name, bgr0, det0 = chosen
        stamp0 = self._image_stamp_ns(obs, cam_name)
        log.info(
            f"[board_search] selected {cam_name}: area={det0[3]:.3f} "
            f"centroid=({det0[1]:.1f},{det0[2]:.1f}) border={det0[5]}"
        )
        if self._is_success(det0, bgr0.shape):
            log.info("[board_search] board already fully framed; moves=0")
            return True

        axes = self._camera_axes_in_base(cam_name)
        if axes is None:
            log.error(f"[board_search] no usable camera TF for {cam_name}")
            return False
        image_u, image_v, back_away = axes
        log.info(
            f"[board_search] {cam_name} probe axes in base: "
            f"u={np.round(image_u, 3).tolist()} v={np.round(image_v, 3).tolist()}"
        )

        # Moves 1 and 2 measure independent columns of d(pixel)/d(camera image
        # plane).  Fixed base X/Y can be parallel to the viewing axis, as the v36
        # trial showed, and then cannot produce an invertible image Jacobian.
        actual_u_vector = self._move_by(move_robot, PROBE_M * image_u)
        actual_du = float(np.dot(actual_u_vector, image_u))
        sample1 = self._observe_camera(get_observation, cam_name, newer_than=stamp0)
        if sample1 is None or not sample1[1][0]:
            log.error(f"[board_search] board lost after image-U probe in {cam_name}")
            return False
        bgr1, det1, stamp1 = sample1
        if self._is_success(det1, bgr1.shape):
            log.info("[board_search] board fully framed; moves=1")
            return True

        actual_v_vector = self._move_by(move_robot, PROBE_M * image_v)
        actual_dv = float(np.dot(actual_v_vector, image_v))
        sample2 = self._observe_camera(get_observation, cam_name, newer_than=stamp1)
        if sample2 is None or not sample2[1][0]:
            log.error(f"[board_search] board lost after image-V probe in {cam_name}")
            return False
        bgr2, det2, stamp2 = sample2
        if self._is_success(det2, bgr2.shape):
            log.info("[board_search] board fully framed; moves=2")
            return True

        if abs(actual_du) < 1e-6 or abs(actual_dv) < 1e-6:
            log.error("[board_search] probe motion was clipped to zero")
            return False
        pix0 = np.array(det0[1:3], dtype=np.float64)
        pix1 = np.array(det1[1:3], dtype=np.float64)
        pix2 = np.array(det2[1:3], dtype=np.float64)
        jacobian = np.column_stack(((pix1 - pix0) / actual_du, (pix2 - pix1) / actual_dv))
        condition = float(np.linalg.cond(jacobian))
        if not np.isfinite(condition) or condition > MAX_JACOBIAN_CONDITION:
            log.error(
                f"[board_search] image Jacobian is singular/ill-conditioned: "
                f"cond={condition:.1f} J={np.round(jacobian, 2).tolist()}"
            )
            return False

        height, width = bgr2.shape[:2]
        pixel_error = np.array([width / 2.0, height / 2.0]) - pix2
        image_plane_correction = np.linalg.solve(jacobian, pixel_error)
        backoff = 0.0
        if det2[5]:
            # Scale the standoff adjustment mildly with observed coverage. The
            # final vector clamp remains the hard safety bound.
            backoff = RAISE_STEP_M * float(np.clip(det2[3] / 0.25, 0.5, 2.0))
        correction = (
            image_plane_correction[0] * image_u
            + image_plane_correction[1] * image_v
            + backoff * back_away
        )
        applied = self._move_by(move_robot, correction)
        log.info(
            f"[board_search] correction={np.round(applied, 4).tolist()} "
            f"J={np.round(jacobian, 2).tolist()} cond={condition:.1f}"
        )

        final_sample = self._observe_camera(get_observation, cam_name, newer_than=stamp2)
        if final_sample is None or not final_sample[1][0]:
            log.error(f"[board_search] board not detected after move {MAX_MOVES}")
            return False
        final_bgr, final_det, _ = final_sample
        centered = self._is_success(final_det, final_bgr.shape)
        fully_in_frame = not final_det[5]
        log.info(
            f"[board_search] final moves={MAX_MOVES} centered={centered} "
            f"fully_in_frame={fully_in_frame} area={final_det[3]:.3f} "
            f"centroid=({final_det[1]:.1f},{final_det[2]:.1f})"
        )
        # After exhausting the budget, a fully visible board is still useful
        # pre-positioning even when the strict center tolerance was just missed.
        return bool(centered or fully_in_frame)


def _main():
    parser = argparse.ArgumentParser(description="Detect the dark taskboard in a saved frame")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, help="debug image path (default: IMAGE.board.png)")
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"could not read image: {args.image}")

    result = BoardSearch._detect(bgr)
    found, cx, cy, area_frac, bbox, touches_border, mask, contour = result
    print({
        "found": found, "centroid": (cx, cy), "area_frac": area_frac,
        "bbox": bbox, "touches_border": touches_border,
    })
    overlay = bgr.copy()
    if found:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        cv2.circle(overlay, (round(cx), round(cy)), 6, (0, 0, 255), -1)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    debug = np.hstack((overlay, mask_bgr))
    output = args.output or args.image.with_name(f"{args.image.stem}.board.png")
    if not cv2.imwrite(str(output), debug):
        raise OSError(f"could not write debug overlay: {output}")
    print(f"debug_overlay={output}")


if __name__ == "__main__":
    _main()
