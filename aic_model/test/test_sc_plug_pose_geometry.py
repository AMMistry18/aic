from pathlib import Path

import numpy as np
import yaml

from aic_model.sc_plug_pose_geometry import (
    SC_PLUG_LOCAL_KEYPOINTS_M,
    format_yolo_pose_label,
    padded_bbox,
    project_keypoints,
    visibility_flags,
    write_dataset_yaml,
)


K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])


def _rotation_xyz(rx, ry, rz):
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    return np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ]
    )


def test_sc_keypoints_are_noncoplanar_and_inside_housing_collision_bounds():
    assert SC_PLUG_LOCAL_KEYPOINTS_M.shape == (8, 3)
    centered = SC_PLUG_LOCAL_KEYPOINTS_M - SC_PLUG_LOCAL_KEYPOINTS_M.mean(axis=0)
    assert np.linalg.matrix_rank(centered) == 3

    # SDF pose of sc_tip_link relative to sc_plug_link.
    plug_from_tip = _rotation_xyz(-1.5708, 0.0, -1.5708)
    points_plug = (
        plug_from_tip @ SC_PLUG_LOCAL_KEYPOINTS_M.T
    ).T + np.array([0.01165, 0.0, 0.0])
    np.testing.assert_allclose(
        sorted(np.unique(np.round(points_plug[:, 0], 5))), [-0.003, 0.009], atol=1e-5
    )
    assert np.max(np.abs(points_plug[:, 1])) <= 0.01001
    assert np.max(np.abs(points_plug[:, 2])) <= 0.00321
    assert points_plug[:, 0].max() <= 0.0094


def test_sc_yolo_label_has_exact_eight_keypoint_schema():
    camera_from_tip = np.eye(4)
    camera_from_tip[2, 3] = 0.35
    pixels, in_front = project_keypoints(
        SC_PLUG_LOCAL_KEYPOINTS_M, camera_from_tip, K
    )
    flags = visibility_flags(pixels, in_front, 640, 480)
    bbox = padded_bbox(pixels, in_front, 640, 480)
    assert bbox is not None
    tokens = format_yolo_pose_label(bbox, pixels, flags, 640, 480).split()
    assert len(tokens) == 5 + 8 * 3
    assert tokens[0] == "0"
    assert [int(tokens[7 + 3 * index]) for index in range(8)] == [2] * 8


def test_sc_dataset_yaml_is_separate_from_legacy_port_dataset(tmp_path: Path):
    output = write_dataset_yaml(tmp_path)
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert output.name == "aic_sc_plug_pose.yaml"
    assert data["names"] == ["sc_plug"]
    assert data["kpt_shape"] == [8, 3]
