"""ROS-free purple-insignia detection for deterministic Stage 1 acquisition.

This intentionally uses the wider, hardware-proven HSV band from
``PerceptionInsert._sc_purple_logo_centroid_px``.  It is an acquisition cue
only: the existing Stage-2 landmark detector remains the sole authority for
declaring that the insignia is complete enough for PnP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


PURPLE_HSV_LOWER = np.array([125, 45, 45], dtype=np.uint8)
PURPLE_HSV_UPPER = np.array([165, 255, 255], dtype=np.uint8)
PURPLE_MIN_AREA_PX = 100


@dataclass(frozen=True)
class PurpleReport:
    """Purple segmentation summary for one wrist-camera image."""

    seen: bool
    full: bool
    edges: frozenset[str] = field(default_factory=frozenset)
    area_frac: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    centroid: tuple[float, float] | None = None
    # Normalized image error: positive means right/down of image centre.
    center_error: tuple[float, float] = (0.0, 0.0)


def analyze_purple(
    image: np.ndarray,
    *,
    margin_px: int = 10,
    min_area_px: int = PURPLE_MIN_AREA_PX,
) -> PurpleReport:
    """Segment the largest purple insignia candidate in ``image``."""

    import cv2

    if image is None or getattr(image, "size", 0) == 0:
        return PurpleReport(seen=False, full=False)
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] >= 3:
        bgr = image[:, :, :3]
    else:
        raise ValueError("image must be gray or BGR/BGRA")

    height, width = bgr.shape[:2]
    if height < 2 or width < 2:
        return PurpleReport(seen=False, full=False)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PURPLE_HSV_LOWER, PURPLE_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_area = 0
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area >= int(min_area_px) and area > best_area:
            best_contour = contour
            best_area = area
    if best_contour is None:
        return PurpleReport(seen=False, full=False)

    x, y, box_width, box_height = cv2.boundingRect(best_contour)
    x0, y0 = int(x), int(y)
    x1, y1 = int(x + box_width - 1), int(y + box_height - 1)
    margin = max(0, int(margin_px))
    edges: set[str] = set()
    if x0 <= margin:
        edges.add("left")
    if y0 <= margin:
        edges.add("top")
    if x1 >= width - 1 - margin:
        edges.add("right")
    if y1 >= height - 1 - margin:
        edges.add("bottom")

    moments = cv2.moments(best_contour)
    if abs(float(moments["m00"])) < 1e-6:
        return PurpleReport(seen=False, full=False)
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    center_error = (
        (cx - 0.5 * float(width - 1)) / max(0.5 * float(width - 1), 1.0),
        (cy - 0.5 * float(height - 1)) / max(0.5 * float(height - 1), 1.0),
    )
    return PurpleReport(
        seen=True,
        full=not edges,
        edges=frozenset(edges),
        area_frac=float(best_area) / float(height * width),
        bbox=(x0, y0, x1, y1),
        centroid=(cx, cy),
        center_error=center_error,
    )

