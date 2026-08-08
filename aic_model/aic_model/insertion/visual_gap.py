"""Standalone dark-port-opening detector for pre-descent visual alignment.

The detector intentionally knows nothing about ROS, TF, or robot motion.  It
receives one BGR camera frame, the projected quadrilateral of the expected SFP
opening, and an optional fixed gripper ignore mask.  This keeps the image
segmentation independently testable on saved wrist-camera frames.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PortOpeningDetection:
    """Accepted dark-opening component in image coordinates."""

    center_uv: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    area_px: int
    expected_area_px: float
    threshold: float
    contrast: float
    score: float
    component_mask: np.ndarray
    search_polygon: np.ndarray


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("image must be grayscale, BGR, or BGRA")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def detect_dark_port_opening(
    image: np.ndarray,
    expected_quad_uv: np.ndarray,
    ignored_pixels: np.ndarray | None = None,
    *,
    diagnostics: dict | None = None,
    search_scale: float = 1.55,
    min_contrast: float = 24.0,
    min_area_ratio: float = 0.16,
    max_area_ratio: float = 1.35,
    max_center_distance: float = 0.75,
) -> PortOpeningDetection | None:
    """Find the dark SFP opening near its projected quadrilateral.

    Thresholds are expressed relative to the projected opening wherever
    possible, so the same detector works on the saved 442x393 frames and the
    full-resolution live camera stream.  ``ignored_pixels`` is True where the
    fixed gripper mask says pixels must not participate.
    """

    gray = _as_gray_u8(image)
    height, width = gray.shape
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "reason": "initializing",
            "image_shape": (height, width),
            "search_scale": float(search_scale),
        })
    quad = np.asarray(expected_quad_uv, dtype=np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        raise ValueError("expected_quad_uv must be a finite 4x2 array")
    if not 1.0 < search_scale <= 3.0:
        raise ValueError("search_scale must be in (1, 3]")

    expected_hull = cv2.convexHull(quad.astype(np.float32))
    expected_area = float(cv2.contourArea(expected_hull))
    if diagnostics is not None:
        diagnostics["expected_area_px"] = expected_area
    if expected_area < 20.0:
        if diagnostics is not None:
            diagnostics["reason"] = "expected_area_too_small"
        return None

    center = quad.mean(axis=0)
    search_quad = center + search_scale * (quad - center)
    search_hull = cv2.convexHull(np.rint(search_quad).astype(np.int32))
    search_hull[:, 0, 0] = np.clip(search_hull[:, 0, 0], 0, width - 1)
    search_hull[:, 0, 1] = np.clip(search_hull[:, 0, 1], 0, height - 1)

    valid = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(valid, search_hull, 255)
    if ignored_pixels is not None:
        ignored = np.asarray(ignored_pixels, dtype=bool)
        if ignored.shape != gray.shape:
            raise ValueError("ignored_pixels shape must match the image")
        valid[ignored] = 0

    if diagnostics is not None:
        diagnostics.update({
            "search_polygon": search_hull[:, 0, :].copy(),
            "valid_mask": valid.copy(),
            "valid_pixels": int(np.count_nonzero(valid)),
            "ignored_pixels": int(np.count_nonzero(ignored_pixels))
            if ignored_pixels is not None else 0,
        })

    values = gray[valid != 0]
    if values.size < max(25, int(0.35 * expected_area)):
        if diagnostics is not None:
            diagnostics["reason"] = "too_few_valid_pixels"
        return None
    p10, p50, p90 = np.percentile(values, [10.0, 50.0, 90.0])
    contrast = float(p90 - p10)
    if diagnostics is not None:
        diagnostics.update({
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90),
            "contrast": contrast,
        })
    if contrast < min_contrast:
        if diagnostics is not None:
            diagnostics["reason"] = "low_contrast"
        return None

    otsu_threshold, _ = cv2.threshold(
        values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    # Keep the threshold on the dark side of the local distribution.  This
    # rejects medium-gray cage walls while retaining the nearly black opening.
    threshold = float(min(otsu_threshold, p50 - 0.10 * contrast))
    if threshold <= float(p10) + 2.0:
        threshold = float(p10) + 0.25 * contrast

    dark = np.zeros_like(valid)
    dark[(valid != 0) & (gray <= threshold)] = 255
    if diagnostics is not None:
        diagnostics.update({
            "otsu_threshold": float(otsu_threshold),
            "threshold": threshold,
        })
    min_edge = max(
        1.0,
        min(float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)),
    )
    kernel_size = 3 if min_edge >= 9.0 else 1
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    if diagnostics is not None:
        diagnostics.update({
            "kernel_size": kernel_size,
            "dark_mask": dark.copy(),
            "dark_pixels": int(np.count_nonzero(dark)),
            "candidates": [],
        })

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    norm_scale = max(np.sqrt(expected_area), 1.0)
    best = None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / expected_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            if diagnostics is not None:
                diagnostics["candidates"].append({
                    "label": label,
                    "area_px": area,
                    "area_ratio": float(area_ratio),
                    "rejection": "area_ratio",
                })
            continue
        component_center = np.asarray(centroids[label], dtype=np.float64)
        center_distance = float(np.linalg.norm(component_center - center) / norm_scale)
        if center_distance > max_center_distance:
            if diagnostics is not None:
                diagnostics["candidates"].append({
                    "label": label,
                    "area_px": area,
                    "area_ratio": float(area_ratio),
                    "center_uv": component_center.copy(),
                    "center_distance": center_distance,
                    "rejection": "center_distance",
                })
            continue
        component = labels == label
        mean_darkness = float(gray[component].mean()) / 255.0
        # A real opening normally fills roughly half the projected mouth quad.
        area_cost = abs(float(np.log(max(area_ratio, 1e-6) / 0.55)))
        score = 2.5 * center_distance + 0.35 * area_cost + 0.30 * mean_darkness
        if diagnostics is not None:
            diagnostics["candidates"].append({
                "label": label,
                "area_px": area,
                "area_ratio": float(area_ratio),
                "center_uv": component_center.copy(),
                "center_distance": center_distance,
                "score": float(score),
                "rejection": None,
            })
        if best is None or score < best[0]:
            best = (score, label, area, component_center)

    if best is None:
        if diagnostics is not None:
            diagnostics["reason"] = "no_accepted_component"
        return None
    score, label, area, component_center = best
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])
    result = PortOpeningDetection(
        center_uv=component_center,
        bbox_xywh=(x, y, w, h),
        area_px=area,
        expected_area_px=expected_area,
        threshold=threshold,
        contrast=contrast,
        score=float(score),
        component_mask=(labels == label),
        search_polygon=search_hull[:, 0, :].copy(),
    )
    if diagnostics is not None:
        diagnostics.update({
            "reason": "accepted",
            "selected_label": int(label),
            "center_uv": component_center.copy(),
            "bbox_xywh": (x, y, w, h),
            "area_px": int(area),
            "score": float(score),
        })
    return result


def draw_port_opening_debug(
    image: np.ndarray,
    expected_quad_uv: np.ndarray,
    detection: PortOpeningDetection | None,
    *,
    ignored_pixels: np.ndarray | None = None,
) -> np.ndarray:
    """Return a BGR overlay for offline detector review."""

    array = np.asarray(image)
    if array.ndim == 2:
        overlay = cv2.cvtColor(_as_gray_u8(array), cv2.COLOR_GRAY2BGR)
    elif array.shape[2] == 4:
        overlay = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    else:
        overlay = np.ascontiguousarray(array[:, :, :3].copy())
    quad = np.rint(np.asarray(expected_quad_uv)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [quad], True, (255, 180, 0), 1, cv2.LINE_AA)
    if ignored_pixels is not None:
        ignored = np.asarray(ignored_pixels, dtype=bool)
        if ignored.shape == overlay.shape[:2]:
            red = np.zeros_like(overlay)
            red[:] = (0, 0, 255)
            overlay[ignored] = cv2.addWeighted(overlay[ignored], 0.70, red[ignored], 0.30, 0)
    if detection is None:
        center = tuple(np.rint(np.asarray(expected_quad_uv).mean(axis=0)).astype(int))
        cv2.drawMarker(overlay, center, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
        return overlay
    component = detection.component_mask
    green = np.zeros_like(overlay)
    green[:] = (0, 255, 0)
    overlay[component] = cv2.addWeighted(overlay[component], 0.45, green[component], 0.55, 0)
    search = detection.search_polygon.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [search], True, (0, 255, 255), 1, cv2.LINE_AA)
    uv = tuple(np.rint(detection.center_uv).astype(int))
    cv2.drawMarker(overlay, uv, (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
    x, y, w, h = detection.bbox_xywh
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(
        overlay,
        f"gap score={detection.score:.2f} contrast={detection.contrast:.0f}",
        (max(0, x - 4), max(12, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return overlay


def draw_port_opening_diagnostic(
    image: np.ndarray,
    expected_quad_uv: np.ndarray,
    detection: PortOpeningDetection | None,
    diagnostics: dict,
    *,
    ignored_pixels: np.ndarray | None = None,
) -> np.ndarray:
    """Build a four-panel crop showing the exact segmentation decision."""

    overlay = draw_port_opening_debug(
        image, expected_quad_uv, detection, ignored_pixels=ignored_pixels)
    gray = _as_gray_u8(image)
    height, width = gray.shape
    polygon = diagnostics.get("search_polygon")
    if polygon is None:
        polygon = np.asarray(expected_quad_uv, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x0 = max(0, int(np.floor(np.min(polygon[:, 0]))) - 8)
    y0 = max(0, int(np.floor(np.min(polygon[:, 1]))) - 8)
    x1 = min(width, int(np.ceil(np.max(polygon[:, 0]))) + 9)
    y1 = min(height, int(np.ceil(np.max(polygon[:, 1]))) + 9)
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, width, height

    valid = np.asarray(
        diagnostics.get("valid_mask", np.zeros_like(gray)), dtype=np.uint8)
    dark = np.asarray(
        diagnostics.get("dark_mask", np.zeros_like(gray)), dtype=np.uint8)
    component = (
        detection.component_mask.astype(np.uint8) * 255
        if detection is not None else np.zeros_like(gray)
    )
    panels = [
        overlay[y0:y1, x0:x1].copy(),
        cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(dark[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(component[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR),
    ]
    labels = ("camera crop", "grayscale", "threshold mask", "selected component")
    panel_height = 180
    rendered = []
    for label, panel in zip(labels, panels):
        scale = panel_height / max(panel.shape[0], 1)
        panel = cv2.resize(
            panel,
            (max(1, int(round(panel.shape[1] * scale))), panel_height),
            interpolation=cv2.INTER_NEAREST if "mask" in label or "component" in label
            else cv2.INTER_AREA,
        )
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 20), (0, 0, 0), -1)
        cv2.putText(
            panel, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(panel)
    mosaic = cv2.hconcat(rendered)
    footer = np.zeros((42, mosaic.shape[1], 3), dtype=np.uint8)
    reason = str(diagnostics.get("reason", "unknown"))
    stats = (
        f"reason={reason} p10/p50/p90="
        f"{diagnostics.get('p10', float('nan')):.1f}/"
        f"{diagnostics.get('p50', float('nan')):.1f}/"
        f"{diagnostics.get('p90', float('nan')):.1f} "
        f"contrast={diagnostics.get('contrast', float('nan')):.1f} "
        f"otsu={diagnostics.get('otsu_threshold', float('nan')):.1f} "
        f"threshold={diagnostics.get('threshold', float('nan')):.1f}"
    )
    cv2.putText(
        footer, stats, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        footer,
        f"crop_xyxy={[x0, y0, x1, y1]} valid={int(np.count_nonzero(valid))} "
        f"dark={int(np.count_nonzero(dark))} candidates="
        f"{len(diagnostics.get('candidates', []))}",
        (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([mosaic, footer])


def encode_port_opening_diagnostic_jpeg(
    image: np.ndarray,
    expected_quad_uv: np.ndarray,
    detection: PortOpeningDetection | None,
    diagnostics: dict,
    *,
    ignored_pixels: np.ndarray | None = None,
    quality: int = 85,
) -> str:
    """Return a compact base64 JPEG suitable for chunked diagnostic logging."""

    mosaic = draw_port_opening_diagnostic(
        image,
        expected_quad_uv,
        detection,
        diagnostics,
        ignored_pixels=ignored_pixels,
    )
    ok, encoded = cv2.imencode(
        ".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("failed to encode visual-gap diagnostic JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
