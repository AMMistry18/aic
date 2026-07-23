"""Geometry shared by SFP plug-pose data generation and runtime inference.

The object frame is Gazebo's ``sfp_tip_link`` frame.  Its local +Z axis is the
inward insertion axis used by the deployment controller; the visible SFP
housing extends mostly along local -Z from the tip origin.

The eight keypoints form two rectangles on the rigid metal housing.  Keeping
the points non-coplanar makes the full six-degree-of-freedom pose observable
and avoids the planar PnP ambiguity of an entrance-face-only label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


SFP_PLUG_CLASS_NAMES = ["sfp_plug"]
SFP_PLUG_KPT_SHAPE = (8, 3)

# Keypoint order, looking from the insertion tip toward the cable:
#
#     near plane                         rear plane
#       0 ----- 1                         4 ----- 5
#       |       |                         |       |
#       3 ----- 2                         7 ----- 6
#
# Coordinates are metres in sfp_tip_link.  The body dimensions come from
# aic_assets/models/SFP Module/model.sdf.  The points sit slightly inside the
# collision/visual bounds so their visual locations are stable at grazing
# camera angles.
SFP_PLUG_LOCAL_KEYPOINTS_M = np.array(
    [
        [+0.0064, +0.0038, -0.0020],
        [-0.0064, +0.0038, -0.0020],
        [-0.0064, -0.0038, -0.0020],
        [+0.0064, -0.0038, -0.0020],
        [+0.0064, +0.0038, -0.0200],
        [-0.0064, +0.0038, -0.0200],
        [-0.0064, -0.0038, -0.0200],
        [+0.0064, -0.0038, -0.0200],
    ],
    dtype=np.float64,
)

# Horizontal image flipping is disabled for training because the plug can
# roll in camera space.  This permutation is still recorded in the dataset
# YAML for tools that require it and is correct for the canonical front view.
SFP_PLUG_FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6]


def validate_transform(transform: np.ndarray, *, name: str = "transform") -> np.ndarray:
    """Validate and return a finite rigid 4x4 transform."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f"{name} rotation must be right-handed")
    return matrix


def project_keypoints(
    local_keypoints_m: np.ndarray,
    camera_from_object: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project object-frame keypoints and return pixels plus positive-depth mask."""

    points = np.asarray(local_keypoints_m, dtype=np.float64).reshape(-1, 3)
    transform = validate_transform(camera_from_object, name="camera_from_object")
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")

    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    camera_points = (transform @ homogeneous.T).T[:, :3]
    in_front = camera_points[:, 2] > 0.01
    safe_z = np.where(in_front, camera_points[:, 2], 1.0)
    projected = (K @ camera_points.T).T
    pixels = projected[:, :2] / safe_z[:, None]
    return pixels, in_front


def visibility_flags(
    pixels: np.ndarray,
    in_front: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Return Ultralytics visibility flags (0 absent, 1 hidden, 2 visible)."""

    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    front = np.asarray(in_front, dtype=bool).reshape(-1)
    if len(points) != len(front):
        raise ValueError("pixels and in_front length mismatch")
    inside = (
        front
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < int(image_width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < int(image_height))
    )
    flags = np.zeros(len(points), dtype=np.int32)
    flags[front] = 1
    flags[inside] = 2
    return flags


def padded_bbox(
    pixels: np.ndarray,
    in_front: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    padding: float = 0.25,
    min_side_px: float = 16.0,
) -> tuple[float, float, float, float] | None:
    """Return a clipped padded pixel bbox around all positive-depth keypoints."""

    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    front = np.asarray(in_front, dtype=bool).reshape(-1)
    if len(points) != len(front) or not np.any(front):
        return None
    visible = points[front]
    x_min, y_min = np.min(visible, axis=0)
    x_max, y_max = np.max(visible, axis=0)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    if width <= 0.0 or height <= 0.0:
        return None
    x_min = max(0.0, float(x_min - padding * width))
    x_max = min(float(image_width - 1), float(x_max + padding * width))
    y_min = max(0.0, float(y_min - padding * height))
    y_max = min(float(image_height - 1), float(y_max + padding * height))
    if (x_max - x_min) < min_side_px or (y_max - y_min) < min_side_px:
        return None
    return x_min, y_min, x_max, y_max


def format_yolo_pose_label(
    bbox_xyxy: Iterable[float],
    pixels: np.ndarray,
    flags: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    class_id: int = 0,
) -> str:
    """Format one normalized Ultralytics YOLO-pose label row."""

    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2).copy()
    vis = np.asarray(flags, dtype=np.int32).reshape(-1)
    if points.shape != (SFP_PLUG_KPT_SHAPE[0], 2) or vis.shape != (SFP_PLUG_KPT_SHAPE[0],):
        raise ValueError(f"expected {SFP_PLUG_KPT_SHAPE[0]} keypoints")
    if np.any((vis < 0) | (vis > 2)):
        raise ValueError("visibility flags must be in [0, 2]")
    width = float(image_width)
    height = float(image_height)
    points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)
    points[vis == 0] = 0.0

    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    cx = ((x_min + x_max) * 0.5) / width
    cy = ((y_min + y_max) * 0.5) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    tokens = [
        str(int(class_id)),
        f"{cx:.7f}",
        f"{cy:.7f}",
        f"{box_width:.7f}",
        f"{box_height:.7f}",
    ]
    for point, flag in zip(points, vis):
        tokens.extend(
            [
                f"{point[0] / width:.7f}",
                f"{point[1] / height:.7f}",
                str(int(flag)),
            ]
        )
    return " ".join(tokens)


def write_dataset_yaml(dataset_root: str | Path) -> Path:
    """Write the canonical SFP plug-pose dataset descriptor."""

    root = Path(dataset_root).expanduser().resolve()
    output = root / "aic_sfp_plug_pose.yaml"
    output.write_text(
        "\n".join(
            [
                "# Simulator-GT SFP plug pose; split by complete randomized trial.",
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                f"names: {SFP_PLUG_CLASS_NAMES}",
                f"kpt_shape: [{SFP_PLUG_KPT_SHAPE[0]}, {SFP_PLUG_KPT_SHAPE[1]}]",
                f"flip_idx: {SFP_PLUG_FLIP_IDX}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


__all__ = [
    "SFP_PLUG_CLASS_NAMES",
    "SFP_PLUG_FLIP_IDX",
    "SFP_PLUG_KPT_SHAPE",
    "SFP_PLUG_LOCAL_KEYPOINTS_M",
    "format_yolo_pose_label",
    "padded_bbox",
    "project_keypoints",
    "validate_transform",
    "visibility_flags",
    "write_dataset_yaml",
]
