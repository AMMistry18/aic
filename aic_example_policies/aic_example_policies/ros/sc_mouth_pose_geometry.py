"""Geometry and dataset conventions for the physical SC-mouth pose model.

This is deliberately separate from the legacy ``best_sc_pose.pt`` convention.
That model predicts a virtual 8.8 x 6.0 mm rectangle, not an observable SC
feature.  The new model is trained on the physical *front mouth* outline plus
its centre, all expressed in the published entrance-frame plane.

Do not substitute these dimensions into ``LOCAL_SC_PORT_KPS`` while keeping
the legacy checkpoint.  The model weight and its local keypoint geometry form
one contract and must switch together after held-out evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


SC_MOUTH_CLASS_NAMES = ["sc_mouth"]
SC_MOUTH_KEYPOINT_COUNT = 5
SC_MOUTH_KPT_SHAPE = (SC_MOUTH_KEYPOINT_COUNT, 3)

# Measured from the SC CAD front face.  The binding throat is 22.407 x 7.85 mm
# through the asymmetric lip, but this target names the externally observable
# physical front-mouth outline: 22.407 x 8.10 mm.  The entrance frame is
# centred mechanically on that face; use its z=0 plane for all labels.
SC_FRONT_MOUTH_WIDTH_M = 0.022407
SC_FRONT_MOUTH_HEIGHT_M = 0.00810

# Keep the legacy corner winding so a later atomic runtime migration has one
# less convention to translate.  KP4 is the explicit physical-mouth centre.
#
#   KP0 (+W,+H) ---- KP1 (-W,+H)
#       |                 |
#   KP3 (+W,-H) ---- KP2 (-W,-H)
#                    KP4 centre
LOCAL_SC_FRONT_MOUTH_KPS_M = np.array(
    [
        [+SC_FRONT_MOUTH_WIDTH_M / 2.0, +SC_FRONT_MOUTH_HEIGHT_M / 2.0, 0.0],
        [-SC_FRONT_MOUTH_WIDTH_M / 2.0, +SC_FRONT_MOUTH_HEIGHT_M / 2.0, 0.0],
        [-SC_FRONT_MOUTH_WIDTH_M / 2.0, -SC_FRONT_MOUTH_HEIGHT_M / 2.0, 0.0],
        [+SC_FRONT_MOUTH_WIDTH_M / 2.0, -SC_FRONT_MOUTH_HEIGHT_M / 2.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# Horizontal image flip exchanges the left/right corners and preserves centre.
SC_MOUTH_FLIP_IDX = [1, 0, 3, 2, 4]


def split_for_trial(trial_index: int) -> str:
    """Return a stable 80/10/10 split for a one-based global trial index.

    Splitting by whole trial prevents adjacent wrist viewpoints and synchronized
    camera images from leaking between train and held-out evaluation splits.
    """

    if trial_index < 1:
        raise ValueError(f"trial_index must be one-based, got {trial_index}")
    remainder = trial_index % 10
    if remainder == 0:
        return "test"
    if remainder == 9:
        return "val"
    return "train"


def format_yolo_pose_label(
    bbox_xyxy: Iterable[float],
    pixels: np.ndarray,
    flags: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    class_id: int = 0,
) -> str:
    """Format one normalized five-keypoint Ultralytics pose label."""

    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2).copy()
    visibility = np.asarray(flags, dtype=np.int32).reshape(-1)
    if points.shape != (SC_MOUTH_KEYPOINT_COUNT, 2):
        raise ValueError(f"expected {SC_MOUTH_KEYPOINT_COUNT} mouth keypoints")
    if visibility.shape != (SC_MOUTH_KEYPOINT_COUNT,):
        raise ValueError(f"expected {SC_MOUTH_KEYPOINT_COUNT} visibility flags")
    if np.any((visibility < 0) | (visibility > 2)):
        raise ValueError("visibility flags must be in [0, 2]")
    if image_width <= 1 or image_height <= 1:
        raise ValueError("image dimensions must exceed one pixel")

    width = float(image_width)
    height = float(image_height)
    points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)
    points[visibility == 0] = 0.0
    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    if not (x_max > x_min and y_max > y_min):
        raise ValueError(f"invalid bbox: {bbox_xyxy}")

    tokens = [
        str(int(class_id)),
        f"{((x_min + x_max) * 0.5) / width:.7f}",
        f"{((y_min + y_max) * 0.5) / height:.7f}",
        f"{(x_max - x_min) / width:.7f}",
        f"{(y_max - y_min) / height:.7f}",
    ]
    for point, flag in zip(points, visibility):
        tokens.extend(
            [
                f"{point[0] / width:.7f}",
                f"{point[1] / height:.7f}",
                str(int(flag)),
            ]
        )
    return " ".join(tokens)


def write_dataset_yaml(dataset_root: str | Path) -> Path:
    """Write the canonical three-way SC physical-mouth dataset descriptor."""

    root = Path(dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / "aic_sc_mouth_pose.yaml"
    output.write_text(
        "\n".join(
            [
                "# Simulator-TF labels for the physical SC front mouth (not the legacy virtual rectangle).",
                f"# Front mouth: {SC_FRONT_MOUTH_WIDTH_M * 1000:.3f} x {SC_FRONT_MOUTH_HEIGHT_M * 1000:.3f} mm; 4 corners + centre.",
                "# Split by complete randomized trial: 80% train, 10% val, 10% test.",
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                f"names: {SC_MOUTH_CLASS_NAMES}",
                f"kpt_shape: [{SC_MOUTH_KPT_SHAPE[0]}, {SC_MOUTH_KPT_SHAPE[1]}]",
                f"flip_idx: {SC_MOUTH_FLIP_IDX}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


__all__ = [
    "LOCAL_SC_FRONT_MOUTH_KPS_M",
    "SC_FRONT_MOUTH_HEIGHT_M",
    "SC_FRONT_MOUTH_WIDTH_M",
    "SC_MOUTH_CLASS_NAMES",
    "SC_MOUTH_FLIP_IDX",
    "SC_MOUTH_KEYPOINT_COUNT",
    "SC_MOUTH_KPT_SHAPE",
    "format_yolo_pose_label",
    "split_for_trial",
    "write_dataset_yaml",
]
