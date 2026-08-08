"""Visual recovery mixed into the canonical InsertionPolicy.

This module is deliberately independent of the main controller flow. It gets
first refusal at the adaptive recovery site before a full retract-and-retry.
"""

from __future__ import annotations

import itertools
import os
import time

import numpy as np

from .visual_gap import (
    detect_dark_port_opening,
    encode_port_opening_diagnostic_jpeg,
)


VISUAL_GAP_WEDGE_ENABLE = os.environ.get(
    "RL_INSERT_VISUAL_GAP_WEDGE_ENABLE", "0"
).strip().lower() in ("1", "true", "yes")
VISUAL_GAP_MAX_STEPS = int(os.environ.get("RL_INSERT_VISUAL_GAP_MAX_STEPS", "8"))
VISUAL_GAP_MAX_STEP_M = float(
    os.environ.get("RL_INSERT_VISUAL_GAP_MAX_STEP_M", "0.0010"))
VISUAL_GAP_MAX_TOTAL_M = float(
    os.environ.get("RL_INSERT_VISUAL_GAP_MAX_TOTAL_M", "0.0060"))
VISUAL_GAP_MAX_PORT_OFFSET_M = float(
    os.environ.get("RL_INSERT_VISUAL_GAP_MAX_PORT_OFFSET_M", "0.0080"))
VISUAL_GAP_MAX_VIEW_DISAGREE_M = float(
    os.environ.get("RL_INSERT_VISUAL_GAP_MAX_VIEW_DISAGREE_M", "0.0030"))
VISUAL_GAP_PIXEL_TOL_FRAC = float(
    os.environ.get("RL_INSERT_VISUAL_GAP_PIXEL_TOL_FRAC", "0.12"))
VISUAL_GAP_MIN_PIXEL_TOL = float(
    os.environ.get("RL_INSERT_VISUAL_GAP_MIN_PIXEL_TOL", "2.0"))
VISUAL_GAP_DEBUG = os.environ.get(
    "RL_INSERT_VISUAL_GAP_DEBUG", "0"
).strip().lower() in ("1", "true", "yes")
VISUAL_GAP_DEBUG_IMAGE_CHUNK = int(
    os.environ.get("RL_INSERT_VISUAL_GAP_DEBUG_IMAGE_CHUNK", "1600"))


class VisualGapRecoveryMixin:
    """Camera-based replacement for v41's low-force circling recovery."""

    @staticmethod
    def _visual_gap_wedge_enabled() -> bool:
        return VISUAL_GAP_WEDGE_ENABLE

    @staticmethod
    def _visual_gap_project_point(P, point):
        point = np.asarray(point, dtype=np.float64)
        projected = P @ np.array(
            [point[0], point[1], point[2], 1.0], dtype=np.float64)
        if not np.all(np.isfinite(projected)) or projected[2] <= 1e-6:
            return None
        return np.array(
            [projected[0] / projected[2], projected[1] / projected[2]],
            dtype=np.float64,
        )

    def _visual_gap_ray_to_port_plane(
        self, uv, K, T_cam_from_base, *, port_pos, normal
    ):
        T_base_from_cam = self._pc.invert_transform(T_cam_from_base)
        origin = T_base_from_cam[:3, 3]
        ray_cam = np.linalg.solve(
            np.asarray(K, dtype=np.float64),
            np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64),
        )
        ray_base = T_base_from_cam[:3, :3] @ ray_cam
        ray_norm = float(np.linalg.norm(ray_base))
        if ray_norm <= 1e-9:
            return None
        ray_base /= ray_norm
        denominator = float(np.dot(normal, ray_base))
        if abs(denominator) <= 1e-6:
            return None
        distance = float(
            np.dot(normal, np.asarray(port_pos, dtype=np.float64) - origin)
            / denominator
        )
        if not np.isfinite(distance) or distance <= 0.0:
            return None
        return origin + distance * ray_base

    @staticmethod
    def _visual_gap_consensus(view_hits, *, port_pos, Rp):
        if len(view_hits) < 2:
            return None
        points = [hit["plane_point"] for hit in view_hits]
        best_pair = None
        for i, j in itertools.combinations(range(len(points)), 2):
            delta = Rp.T @ (points[i] - points[j])
            disagreement = float(np.linalg.norm(delta[:2]))
            if best_pair is None or disagreement < best_pair[0]:
                best_pair = (disagreement, i, j)
        if best_pair is None or best_pair[0] > VISUAL_GAP_MAX_VIEW_DISAGREE_M:
            return None
        _, i, j = best_pair
        kept = [points[i], points[j]]
        pair_mean = np.mean(kept, axis=0)
        for k, point in enumerate(points):
            if k in (i, j):
                continue
            delta = Rp.T @ (point - pair_mean)
            if float(np.linalg.norm(delta[:2])) <= VISUAL_GAP_MAX_VIEW_DISAGREE_M:
                kept.append(point)
        hole_point = np.mean(kept, axis=0)
        offset = Rp.T @ (hole_point - np.asarray(port_pos, dtype=np.float64))
        if float(np.linalg.norm(offset[:2])) > VISUAL_GAP_MAX_PORT_OFFSET_M:
            return None
        return hole_point - Rp[:, 2] * float(offset[2])

    @staticmethod
    def _visual_gap_debug_summary(diagnostics):
        candidates = diagnostics.get("candidates", [])
        candidate_text = []
        for candidate in candidates[:8]:
            center = candidate.get("center_uv")
            center_text = (
                np.round(np.asarray(center), 1).tolist()
                if center is not None else None
            )
            candidate_text.append(
                f"label={candidate.get('label')} area={candidate.get('area_px')} "
                f"ratio={candidate.get('area_ratio', float('nan')):.2f} "
                f"center={center_text} "
                f"dist={candidate.get('center_distance', float('nan')):.2f} "
                f"score={candidate.get('score', float('nan')):.2f} "
                f"reject={candidate.get('rejection')}"
            )
        return "; ".join(candidate_text) if candidate_text else "none"

    @staticmethod
    def _visual_gap_emit_debug_image(log, *, camera, step, encoded_jpeg):
        chunk_size = max(512, VISUAL_GAP_DEBUG_IMAGE_CHUNK)
        parts = [
            encoded_jpeg[i:i + chunk_size]
            for i in range(0, len(encoded_jpeg), chunk_size)
        ]
        image_id = f"{time.time_ns()}-{camera}-s{step}"
        for index, part in enumerate(parts, start=1):
            log.info(
                f"[visual-gap-debug-image] id={image_id} camera={camera} "
                f"step={step} part={index}/{len(parts)} data={part}")

    def _run_visual_gap_wedge_recovery(
        self,
        get_observation,
        move_robot,
        *,
        raw_port_pos,
        Rp,
        R_seat,
        local_port_kps,
        stiffness,
        damping,
        step_dt,
    ):
        """Center the plug's mouth-plane intercept in at least two cameras.

        Return the detected physical opening point after convergence.  Any
        ordinary failure returns None so the untouched v41 adaptive sweep runs.
        Exceptions are handled at the v41 call site for the same fail-open path.
        """
        from aic_perception.gripper_masks import GripperMaskBank

        log = self.get_logger()
        mask_bank = GripperMaskBank()
        raw_port_pos = np.asarray(raw_port_pos, dtype=np.float64).reshape(3)
        Rp = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
        insert_axis = Rp[:, 2]
        local_port_kps = np.asarray(local_port_kps, dtype=np.float64).reshape(4, 3)
        mouth_world = raw_port_pos + (Rp @ local_port_kps.T).T
        commanded_total = np.zeros(2, dtype=np.float64)

        for step in range(VISUAL_GAP_MAX_STEPS + 1):
            self._enforce_action_deadline(move_robot)
            obs = get_observation()
            if obs is None:
                log.warn("[visual-gap] no observation; falling back to adaptive sweep")
                return None
            views = self._build_views(obs)
            if len(views) < 2:
                log.warn(f"[visual-gap] only {len(views)} usable camera views; "
                         "falling back to adaptive sweep")
                return None

            tcp_pos, tcp_quat = self._tcp()
            tip_pos, _ = self._tip_from_tcp(tcp_pos, tcp_quat)
            raw_depth = float(np.dot(tip_pos - raw_port_pos, insert_axis))
            # Compare the opening with the current descent line at the mouth
            # plane. Comparing the physically deeper wedged tip directly would
            # create camera-dependent parallax and false corrections.
            plug_mouth_intercept = tip_pos - insert_axis * raw_depth
            hits = []
            for camera, (bgr, K, T_cam_from_base) in views.items():
                P = self._pc.build_projection_matrix(K, T_cam_from_base)
                quad_uv = [
                    self._visual_gap_project_point(P, point) for point in mouth_world
                ]
                if any(uv is None for uv in quad_uv):
                    continue
                quad_uv = np.asarray(quad_uv, dtype=np.float64)
                ignored = mask_bank.ignored_pixels(camera, bgr.shape)
                diagnostics = {} if VISUAL_GAP_DEBUG else None
                detection = detect_dark_port_opening(
                    bgr, quad_uv, ignored, diagnostics=diagnostics)
                if VISUAL_GAP_DEBUG:
                    log.info(
                        f"[visual-gap-debug] camera={camera} step={step} "
                        f"reason={diagnostics.get('reason')} "
                        f"quad_uv={np.round(quad_uv, 1).tolist()} "
                        f"search_uv={np.asarray(diagnostics.get('search_polygon', []), dtype=int).tolist()} "
                        f"expected_area_px={diagnostics.get('expected_area_px', float('nan')):.1f} "
                        f"valid_px={diagnostics.get('valid_pixels', 0)} "
                        f"ignored_px={diagnostics.get('ignored_pixels', 0)} "
                        f"p10={diagnostics.get('p10', float('nan')):.1f} "
                        f"p50={diagnostics.get('p50', float('nan')):.1f} "
                        f"p90={diagnostics.get('p90', float('nan')):.1f} "
                        f"contrast={diagnostics.get('contrast', float('nan')):.1f} "
                        f"otsu={diagnostics.get('otsu_threshold', float('nan')):.1f} "
                        f"threshold={diagnostics.get('threshold', float('nan')):.1f} "
                        f"dark_px={diagnostics.get('dark_pixels', 0)} "
                        f"selected_uv={np.round(np.asarray(diagnostics.get('center_uv', [])), 1).tolist()} "
                        f"bbox={diagnostics.get('bbox_xywh')} "
                        f"score={diagnostics.get('score', float('nan')):.2f}")
                    log.info(
                        f"[visual-gap-debug] camera={camera} step={step} "
                        "components="
                        f"{self._visual_gap_debug_summary(diagnostics)}")
                    if step == 0:
                        try:
                            encoded_jpeg = encode_port_opening_diagnostic_jpeg(
                                bgr,
                                quad_uv,
                                detection,
                                diagnostics,
                                ignored_pixels=ignored,
                            )
                            self._visual_gap_emit_debug_image(
                                log,
                                camera=camera,
                                step=step,
                                encoded_jpeg=encoded_jpeg,
                            )
                        except Exception as ex:
                            log.warn(
                                f"[visual-gap-debug] camera={camera} image encode "
                                f"failed: {ex}")
                if detection is None:
                    log.warn(f"[visual-gap] {camera}: opening not found")
                    continue
                plug_uv = self._visual_gap_project_point(P, plug_mouth_intercept)
                if plug_uv is None:
                    continue
                plane_point = self._visual_gap_ray_to_port_plane(
                    detection.center_uv,
                    K,
                    T_cam_from_base,
                    port_pos=raw_port_pos,
                    normal=insert_axis,
                )
                if plane_point is None:
                    continue
                if VISUAL_GAP_DEBUG:
                    plane_offset = Rp.T @ (plane_point - raw_port_pos)
                    log.info(
                        f"[visual-gap-debug] camera={camera} step={step} "
                        f"plane_offset_mm="
                        f"{np.round(plane_offset * 1000.0, 2).tolist()} "
                        f"pixel_error={np.linalg.norm(detection.center_uv-plug_uv):.2f}")
                hits.append({
                    "camera": camera,
                    "plane_point": plane_point,
                    "pixel_error": float(
                        np.linalg.norm(detection.center_uv - plug_uv)),
                    "pixel_tolerance": max(
                        VISUAL_GAP_MIN_PIXEL_TOL,
                        VISUAL_GAP_PIXEL_TOL_FRAC
                        * np.sqrt(detection.expected_area_px),
                    ),
                })

            if len(hits) < 2:
                log.warn(f"[visual-gap] opening visible in {len(hits)} camera(s); "
                         "falling back to adaptive sweep")
                return None
            if VISUAL_GAP_DEBUG:
                pair_text = []
                for first, second in itertools.combinations(hits, 2):
                    delta = Rp.T @ (
                        first["plane_point"] - second["plane_point"])
                    pair_text.append(
                        f'{first["camera"]}<->{second["camera"]}='
                        f'{np.linalg.norm(delta[:2])*1000.0:.2f}mm '
                        f'delta_xy_mm={np.round(delta[:2]*1000.0, 2).tolist()}')
                log.info(
                    f"[visual-gap-debug] step={step} pairwise="
                    + "; ".join(pair_text))
            hole_point = self._visual_gap_consensus(
                hits, port_pos=raw_port_pos, Rp=Rp)
            if hole_point is None:
                log.warn("[visual-gap] camera rays disagree; falling back to adaptive sweep")
                return None

            centered = [
                hit for hit in hits
                if hit["pixel_error"] <= hit["pixel_tolerance"]
            ]
            errors = ", ".join(
                f'{hit["camera"]}={hit["pixel_error"]:.1f}/'
                f'{hit["pixel_tolerance"]:.1f}px' for hit in hits
            )
            if len(centered) >= 2:
                offset = Rp.T @ (hole_point - raw_port_pos)
                log.info(f"[visual-gap] WEDGE RECOVERED: centered in "
                         f"{len(centered)}/{len(hits)} cameras after {step} steps; "
                         f"{errors}; hole_offset_mm="
                         f"{np.round(offset[:2] * 1000.0, 2).tolist()}")
                return hole_point
            if step >= VISUAL_GAP_MAX_STEPS:
                log.warn(f"[visual-gap] no convergence in {VISUAL_GAP_MAX_STEPS} "
                         f"steps ({errors}); falling back to adaptive sweep")
                return None

            correction = (Rp.T @ (hole_point - plug_mouth_intercept))[:2]
            correction_norm = float(np.linalg.norm(correction))
            if not np.isfinite(correction_norm) or correction_norm <= 1e-6:
                log.warn("[visual-gap] invalid correction; falling back to adaptive sweep")
                return None
            step_xy = correction.copy()
            if correction_norm > VISUAL_GAP_MAX_STEP_M:
                step_xy *= VISUAL_GAP_MAX_STEP_M / correction_norm
            if float(np.linalg.norm(commanded_total + step_xy)) > VISUAL_GAP_MAX_TOTAL_M:
                log.warn(f"[visual-gap] {VISUAL_GAP_MAX_TOTAL_M*1000.0:.1f}mm "
                         "excursion cap reached; falling back to adaptive sweep")
                return None
            commanded_total += step_xy
            lat_world = Rp[:, 0] * step_xy[0] + Rp[:, 1] * step_xy[1]
            log.info(f"[visual-gap] wedge step {step}: {errors}; nudge_mm="
                     f"{np.round(step_xy*1000.0, 2).tolist()} total_mm="
                     f"{np.round(commanded_total*1000.0, 2).tolist()}")
            self.set_pose_target(
                move_robot,
                self._tcp_target_for_tip(tip_pos + lat_world, R_seat),
                stiffness=stiffness,
                damping=damping,
            )
            self.sleep_for(step_dt)
        return None
