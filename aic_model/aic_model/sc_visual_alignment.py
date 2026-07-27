"""Camera-directed lateral alignment for the physical SC duplex opening.

The SC pose network is useful for selecting the correct adapter and estimating
its plane, but its four label corners have a repeatable common-centre bias.  This
module measures the physical opening in the same wrist-camera images instead:

* blue pixels confirm that the ROI belongs to an SC adapter;
* the two dark duplex bores provide the actual mouth centre;
* camera rays intersect the already-estimated port plane; and
* agreeing views produce a bounded lateral target.

Everything here is ROS-free and fail-soft.  Missing/occluded imagery returns
``None``; it is never itself a reason to abandon an insertion attempt.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from .visual_gap import detect_dark_port_opening


SC_BLUE_LOWER = np.array([90, 80, 60], dtype=np.uint8)
SC_BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)


@dataclass(frozen=True)
class ScDuplexOpeningDetection:
    """Physical duplex-mouth centre detected in one image."""

    center_uv: np.ndarray
    bore_centers_uv: tuple[np.ndarray | None, np.ndarray | None]
    detected_bores: int
    blue_fraction: float
    score: float


@dataclass(frozen=True)
class ScOpeningEstimate:
    """Opening point fused on the port plane."""

    point_world: np.ndarray
    cameras: tuple[str, ...]
    disagreement_m: float
    single_view: bool


def _finite_quad(value: np.ndarray) -> np.ndarray:
    quad = np.asarray(value, dtype=np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        raise ValueError("each expected SC bore quad must be a finite 4x2 array")
    return quad


def detect_sc_duplex_opening(
    image: np.ndarray,
    expected_bore_quads_uv: np.ndarray,
    ignored_pixels: np.ndarray | None = None,
    *,
    diagnostics: dict | None = None,
    search_scale: float = 1.65,
    min_blue_fraction: float = 0.01,
) -> ScDuplexOpeningDetection | None:
    """Detect the centre of the two physical SC bores.

    The housing is asymmetric, so its blue-blob centroid is deliberately never
    used as the insertion target.  Blue is only an association gate.  Each dark
    bore is localized independently near its CAD projection; when only one bore
    remains visible, the known projected bore-to-duplex offset recovers the
    centre with lower confidence.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image must be BGR or BGRA")
    bgr = array[:, :, :3]
    bore_quads = np.asarray(expected_bore_quads_uv, dtype=np.float64)
    if bore_quads.shape != (2, 4, 2):
        raise ValueError("expected_bore_quads_uv must have shape 2x4x2")
    bore_quads = np.stack([_finite_quad(quad) for quad in bore_quads])
    if not 1.0 < search_scale <= 3.0:
        raise ValueError("search_scale must be in (1, 3]")

    height, width = bgr.shape[:2]
    if ignored_pixels is not None:
        ignored = np.asarray(ignored_pixels, dtype=bool)
        if ignored.shape != (height, width):
            raise ValueError("ignored_pixels shape must match the image")
    else:
        ignored = None

    expected_center = bore_quads.mean(axis=(0, 1))
    all_points = bore_quads.reshape(-1, 2)
    search_points = expected_center + search_scale * (all_points - expected_center)
    search_hull = cv2.convexHull(np.rint(search_points).astype(np.int32))
    search_hull[:, 0, 0] = np.clip(search_hull[:, 0, 0], 0, width - 1)
    search_hull[:, 0, 1] = np.clip(search_hull[:, 0, 1], 0, height - 1)
    valid = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(valid, search_hull, 255)
    if ignored is not None:
        valid[ignored] = 0
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 25:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(reason="too_few_valid_pixels", valid_pixels=valid_count)
        return None

    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, SC_BLUE_LOWER, SC_BLUE_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    blue_fraction = float(np.count_nonzero((blue != 0) & (valid != 0)) / valid_count)
    if blue_fraction < min_blue_fraction:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(
                reason="blue_association_failed",
                blue_fraction=blue_fraction,
                valid_pixels=valid_count,
            )
        return None

    bore_detections = []
    bore_diagnostics = []
    for quad in bore_quads:
        diag = {}
        detection = detect_dark_port_opening(
            bgr,
            quad,
            ignored,
            diagnostics=diag,
            search_scale=search_scale,
            min_area_ratio=0.12,
            max_area_ratio=1.45,
            max_center_distance=0.85,
        )
        bore_detections.append(detection)
        bore_diagnostics.append(diag)

    visible = [index for index, detection in enumerate(bore_detections)
               if detection is not None]
    if not visible:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(
                reason="no_dark_bore",
                blue_fraction=blue_fraction,
                bores=bore_diagnostics,
            )
        return None

    centers: list[np.ndarray | None] = [
        None if detection is None else np.asarray(detection.center_uv, dtype=np.float64)
        for detection in bore_detections
    ]
    if len(visible) == 2:
        observed_delta = centers[1] - centers[0]
        expected_delta = bore_quads[1].mean(axis=0) - bore_quads[0].mean(axis=0)
        expected_separation = float(np.linalg.norm(expected_delta))
        observed_separation = float(np.linalg.norm(observed_delta))
        separation_ratio = observed_separation / max(expected_separation, 1e-6)
        if not 0.55 <= separation_ratio <= 1.55:
            if diagnostics is not None:
                diagnostics.clear()
                diagnostics.update(
                    reason="bore_pair_geometry",
                    blue_fraction=blue_fraction,
                    separation_ratio=separation_ratio,
                    bores=bore_diagnostics,
                )
            return None
        center_uv = 0.5 * (centers[0] + centers[1])
        score = float(np.mean([detection.score for detection in bore_detections]))
        reason = "accepted_pair"
    else:
        index = visible[0]
        # Infer the duplex centre from the observed bore and its projected CAD
        # offset.  This remains directed when one bore is hidden by the plug.
        projected_bore_center = bore_quads[index].mean(axis=0)
        center_uv = centers[index] + (expected_center - projected_bore_center)
        score = float(bore_detections[index].score + 0.5)
        separation_ratio = float("nan")
        reason = "accepted_single_bore"

    result = ScDuplexOpeningDetection(
        center_uv=np.asarray(center_uv, dtype=np.float64),
        bore_centers_uv=(centers[0], centers[1]),
        detected_bores=len(visible),
        blue_fraction=blue_fraction,
        score=score,
    )
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            reason=reason,
            center_uv=result.center_uv.copy(),
            blue_fraction=blue_fraction,
            detected_bores=len(visible),
            separation_ratio=separation_ratio,
            score=score,
            bores=bore_diagnostics,
        )
    return result


def project_point_px(projection: np.ndarray, point_world: np.ndarray) -> np.ndarray | None:
    """Project a world point, returning ``None`` behind/at the camera."""

    point = np.asarray(point_world, dtype=np.float64).reshape(3)
    projected = np.asarray(projection, dtype=np.float64).reshape(3, 4) @ np.array(
        [point[0], point[1], point[2], 1.0], dtype=np.float64
    )
    if not np.all(np.isfinite(projected)) or projected[2] <= 1e-6:
        return None
    return projected[:2] / projected[2]


def ray_to_plane(
    uv: np.ndarray,
    K: np.ndarray,
    T_cam_from_world: np.ndarray,
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """Intersect an image ray with a world-frame plane."""

    try:
        T_world_from_cam = np.linalg.inv(
            np.asarray(T_cam_from_world, dtype=np.float64).reshape(4, 4)
        )
        ray_cam = np.linalg.solve(
            np.asarray(K, dtype=np.float64).reshape(3, 3),
            np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64),
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    origin = T_world_from_cam[:3, 3]
    ray_world = T_world_from_cam[:3, :3] @ ray_cam
    ray_norm = float(np.linalg.norm(ray_world))
    if ray_norm <= 1e-9:
        return None
    ray_world /= ray_norm
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    denominator = float(np.dot(normal, ray_world))
    if abs(denominator) <= 1e-6:
        return None
    distance = float(
        np.dot(normal, np.asarray(plane_point, dtype=np.float64).reshape(3) - origin)
        / denominator
    )
    if not np.isfinite(distance) or distance <= 0.0:
        return None
    return origin + distance * ray_world


def fuse_sc_opening_hits(
    view_hits: list[dict],
    *,
    origin_port_pos: np.ndarray,
    Rp: np.ndarray,
    max_view_disagreement_m: float,
    max_total_offset_m: float,
    allow_single_view: bool = True,
) -> ScOpeningEstimate | None:
    """Fuse camera ray hits using the closest agreeing pair.

    One high-confidence view remains usable when the other wrists are occluded.
    A disagreement between multiple views does not silently choose a winner; it
    returns ``None`` so the controller retains its last valid target.
    """

    if not view_hits:
        return None
    origin = np.asarray(origin_port_pos, dtype=np.float64).reshape(3)
    frame = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
    points = [np.asarray(hit["plane_point"], dtype=np.float64).reshape(3)
              for hit in view_hits]
    cameras = [str(hit.get("camera", "unknown")) for hit in view_hits]

    if len(points) == 1:
        if not allow_single_view:
            return None
        point = points[0]
        disagreement = float("nan")
        used_cameras = (cameras[0],)
        single_view = True
    else:
        best_pair = None
        for i, j in itertools.combinations(range(len(points)), 2):
            delta = frame.T @ (points[i] - points[j])
            disagreement = float(np.linalg.norm(delta[:2]))
            if best_pair is None or disagreement < best_pair[0]:
                best_pair = (disagreement, i, j)
        if best_pair is None or best_pair[0] > max_view_disagreement_m:
            return None
        disagreement, i, j = best_pair
        kept_indices = [i, j]
        pair_mean = 0.5 * (points[i] + points[j])
        for index, point_candidate in enumerate(points):
            if index in kept_indices:
                continue
            delta = frame.T @ (point_candidate - pair_mean)
            if float(np.linalg.norm(delta[:2])) <= max_view_disagreement_m:
                kept_indices.append(index)
        point = np.mean([points[index] for index in kept_indices], axis=0)
        used_cameras = tuple(cameras[index] for index in kept_indices)
        single_view = False

    offset = frame.T @ (point - origin)
    if not np.all(np.isfinite(offset)) or float(np.linalg.norm(offset[:2])) > max_total_offset_m:
        return None
    point_on_origin_plane = point - frame[:, 2] * float(offset[2])
    return ScOpeningEstimate(
        point_world=point_on_origin_plane,
        cameras=used_cameras,
        disagreement_m=disagreement,
        single_view=single_view,
    )


def bounded_visual_port_update(
    current_port_pos: np.ndarray,
    origin_port_pos: np.ndarray,
    observed_port_pos: np.ndarray,
    Rp: np.ndarray,
    *,
    max_step_m: float,
    max_total_m: float,
    step_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a clipped lateral target and applied port-frame correction."""

    current = np.asarray(current_port_pos, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_port_pos, dtype=np.float64).reshape(3)
    observed = np.asarray(observed_port_pos, dtype=np.float64).reshape(3)
    frame = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
    correction = (frame.T @ (observed - current))[:2]
    norm = float(np.linalg.norm(correction))
    if not np.isfinite(norm):
        return None
    step_limit = max(0.0, float(max_step_m) * float(step_scale))
    step = correction.copy()
    if norm > step_limit > 0.0:
        step *= step_limit / norm
    elif step_limit <= 0.0 and norm > 0.0:
        return None
    candidate = current + frame[:, 0] * step[0] + frame[:, 1] * step[1]
    total = (frame.T @ (candidate - origin))[:2]
    total_norm = float(np.linalg.norm(total))
    if total_norm > max_total_m:
        # Saturate at the immutable excursion boundary instead of crossing it.
        if total_norm <= 1e-12 or max_total_m <= 0.0:
            return None
        total *= float(max_total_m) / total_norm
        candidate = origin + frame[:, 0] * total[0] + frame[:, 1] * total[1]
        step = (frame.T @ (candidate - current))[:2]
    return candidate, step


__all__ = [
    "ScDuplexOpeningDetection",
    "ScOpeningEstimate",
    "bounded_visual_port_update",
    "detect_sc_duplex_opening",
    "fuse_sc_opening_hits",
    "project_point_px",
    "ray_to_plane",
]
