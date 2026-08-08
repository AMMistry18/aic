"""Geometry shared by SC plug-pose collection and runtime inference.

The object frame is Gazebo's ``sc_tip_link`` frame from the canonical
``SC Plug`` asset.  The model places that frame at the ferrule tips and rotates
it so local +Z is the insertion axis.  Consequently, the visible plug body
extends along local -Z.

The eight keypoints form a non-coplanar cuboid inside the rigid duplex blue
housing.  They are derived from the collision geometry in
``aic_assets/models/SC Plug/model.sdf`` rather than from the unrelated SC port
opening dimensions used by the legacy ``best_sc_pose.pt`` model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .sfp_plug_pose_geometry import (
    padded_bbox,
    project_keypoints,
    validate_transform,
    visibility_flags,
)


SC_PLUG_CLASS_NAMES = ["sc_plug"]
SC_PLUG_KPT_SHAPE = (8, 3)

# The SC Plug SDF defines sc_tip_link at plug-frame x=+11.65 mm with
# rpy=(-pi/2, 0, -pi/2).  In sc_tip_link coordinates, x spans the two duplex
# housings, y spans their height, and z points forward along insertion.  The
# selected near/rear planes map back to plug-frame x=+9.0/-3.0 mm, safely
# inside the collision-backed blue housing bounds (the mesh ends at about
# +9.4 mm).  The transverse extents
# map to plug-frame y=+/-10.0 mm and z=+/-3.2 mm, also inside those bounds.
#
# Keypoint order looking from the insertion tip toward the cable:
#
#     near plane                         rear plane
#       0 ----- 1                         4 ----- 5
#       |       |                         |       |
#       3 ----- 2                         7 ----- 6
SC_PLUG_LOCAL_KEYPOINTS_M = np.array(
    [
        [-0.0100, -0.0032, -0.00265],
        [+0.0100, -0.0032, -0.00265],
        [+0.0100, +0.0032, -0.00265],
        [-0.0100, +0.0032, -0.00265],
        [-0.0100, -0.0032, -0.01465],
        [+0.0100, -0.0032, -0.01465],
        [+0.0100, +0.0032, -0.01465],
        [-0.0100, +0.0032, -0.01465],
    ],
    dtype=np.float64,
)

SC_PLUG_FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6]


def format_yolo_pose_label(
    bbox_xyxy: Iterable[float],
    pixels: np.ndarray,
    flags: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    class_id: int = 0,
) -> str:
    """Format one normalized eight-keypoint Ultralytics pose label."""

    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2).copy()
    vis = np.asarray(flags, dtype=np.int32).reshape(-1)
    expected = SC_PLUG_KPT_SHAPE[0]
    if points.shape != (expected, 2) or vis.shape != (expected,):
        raise ValueError(f"expected {expected} keypoints")
    if np.any((vis < 0) | (vis > 2)):
        raise ValueError("visibility flags must be in [0, 2]")

    width = float(image_width)
    height = float(image_height)
    points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)
    points[vis == 0] = 0.0

    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    tokens = [
        str(int(class_id)),
        f"{((x_min + x_max) * 0.5) / width:.7f}",
        f"{((y_min + y_max) * 0.5) / height:.7f}",
        f"{(x_max - x_min) / width:.7f}",
        f"{(y_max - y_min) / height:.7f}",
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
    """Write the canonical SC plug-pose dataset descriptor."""

    root = Path(dataset_root).expanduser().resolve()
    output = root / "aic_sc_plug_pose.yaml"
    output.write_text(
        "\n".join(
            [
                "# Simulator-GT SC plug pose; split by complete randomized trial.",
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                f"names: {SC_PLUG_CLASS_NAMES}",
                f"kpt_shape: [{SC_PLUG_KPT_SHAPE[0]}, {SC_PLUG_KPT_SHAPE[1]}]",
                f"flip_idx: {SC_PLUG_FLIP_IDX}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


__all__ = [
    "SC_PLUG_CLASS_NAMES",
    "SC_PLUG_FLIP_IDX",
    "SC_PLUG_KPT_SHAPE",
    "SC_PLUG_LOCAL_KEYPOINTS_M",
    "format_yolo_pose_label",
    "padded_bbox",
    "project_keypoints",
    "validate_transform",
    "visibility_flags",
    "write_dataset_yaml",
]
