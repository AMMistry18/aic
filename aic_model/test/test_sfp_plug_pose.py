from types import SimpleNamespace

import numpy as np

from aic_model.sfp_plug_pose import (
    PlugKeypointDetection,
    PlugPoseView,
    SfpPlugPoseEstimator,
    fit_rigid_transform,
    fuse_multiview_keypoints,
    stamp_to_seconds,
)
from aic_model.sfp_plug_pose_geometry import (
    SFP_PLUG_LOCAL_KEYPOINTS_M,
    format_yolo_pose_label,
    padded_bbox,
    project_keypoints,
    visibility_flags,
)


K = np.array(
    [
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
)


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


def _synthetic_views_and_detections(stamp=10.0):
    rotation = _rotation_xyz(0.12, -0.08, 0.18)
    position = np.array([0.025, -0.015, 0.72])
    world_from_plug = np.eye(4)
    world_from_plug[:3, :3] = rotation
    world_from_plug[:3, 3] = position
    views = []
    detections = []
    for name, camera_position in (
        ("left", [-0.12, 0.00, 0.0]),
        ("center", [0.00, 0.03, 0.0]),
        ("right", [0.12, 0.00, 0.0]),
    ):
        world_from_camera = np.eye(4)
        world_from_camera[:3, 3] = camera_position
        camera_from_plug = np.linalg.inv(world_from_camera) @ world_from_plug
        pixels, in_front = project_keypoints(
            SFP_PLUG_LOCAL_KEYPOINTS_M, camera_from_plug, K
        )
        assert np.all(in_front)
        views.append(
            PlugPoseView(
                camera_name=name,
                image_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
                K=K,
                T_world_from_camera=world_from_camera,
                stamp_s=stamp,
                frame_id=f"{name}:{stamp}",
            )
        )
        detections.append(
            PlugKeypointDetection(
                camera_name=name,
                keypoints_px=pixels,
                keypoint_confidences=np.full(8, 0.95),
                box_confidence=0.97,
            )
        )
    return views, detections, position, rotation


def test_stamp_to_seconds_accepts_ros_shapes_and_rejects_missing():
    assert stamp_to_seconds(SimpleNamespace(sec=2, nanosec=500_000_000)) == 2.5
    assert stamp_to_seconds(SimpleNamespace(nanoseconds=3_250_000_000)) == 3.25
    try:
        stamp_to_seconds(None)
    except ValueError as exc:
        assert "required" in str(exc)
    else:
        raise AssertionError("missing image timestamp must be rejected")


def test_yolo_label_has_eight_normalized_keypoints():
    camera_from_plug = np.eye(4)
    camera_from_plug[2, 3] = 0.3
    pixels, in_front = project_keypoints(
        SFP_PLUG_LOCAL_KEYPOINTS_M, camera_from_plug, K
    )
    flags = visibility_flags(pixels, in_front, 640, 480)
    bbox = padded_bbox(pixels, in_front, 640, 480)
    assert bbox is not None

    parts = format_yolo_pose_label(bbox, pixels, flags, 640, 480).split()

    assert len(parts) == 5 + 8 * 3
    assert parts[0] == "0"
    assert all(0.0 <= float(parts[index]) <= 1.0 for index in range(1, len(parts)) if index not in range(7, len(parts), 3))
    assert [int(parts[7 + 3 * index]) for index in range(8)] == [2] * 8


def test_weighted_rigid_fit_recovers_full_pose():
    rotation = _rotation_xyz(0.2, -0.15, 0.35)
    position = np.array([0.3, -0.2, 0.7])
    world = (rotation @ SFP_PLUG_LOCAL_KEYPOINTS_M.T).T + position

    recovered_rotation, recovered_position, rmse = fit_rigid_transform(
        SFP_PLUG_LOCAL_KEYPOINTS_M, world, np.linspace(0.5, 1.0, 8)
    )

    np.testing.assert_allclose(recovered_position, position, atol=1e-12)
    np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-12)
    assert rmse < 1e-12


def test_multiview_fusion_recovers_sfp_tip_frame():
    views, detections, expected_position, expected_rotation = (
        _synthetic_views_and_detections()
    )

    position, rotation, rmse, reprojection, count, confidence = (
        fuse_multiview_keypoints(views, detections)
    )

    np.testing.assert_allclose(position, expected_position, atol=1e-9)
    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-8)
    assert rmse < 1e-9
    assert reprojection < 1e-7
    assert count == 8
    assert np.isclose(confidence, 0.95)


class _ArrayBox:
    def __init__(self, value):
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _Boxes:
    def __init__(self, confidences):
        self.conf = _ArrayBox(confidences)

    def __len__(self):
        return len(self.conf.numpy())


class _Keypoints:
    def __init__(self, xy, confidence):
        self.xy = _ArrayBox(xy)
        self.conf = _ArrayBox(confidence)


class _FakeYolo:
    def __init__(self, detections):
        self._detections = detections

    def __call__(self, _images, **_kwargs):
        return [
            SimpleNamespace(
                boxes=_Boxes([detection.box_confidence]),
                keypoints=_Keypoints(
                    detection.keypoints_px[None, ...],
                    detection.keypoint_confidences[None, ...],
                ),
            )
            for detection in self._detections
        ]


def test_estimator_returns_fresh_relative_pose_and_fails_closed_when_stale():
    views, detections, expected_position, expected_rotation = (
        _synthetic_views_and_detections(stamp=10.0)
    )
    estimator = SfpPlugPoseEstimator(
        model=_FakeYolo(detections),
        min_pose_confidence=0.1,
    )

    relative = estimator.estimate_relative_to_port(
        views,
        port_position_world=np.zeros(3),
        port_rotation_world=np.eye(3),
        now_s=10.05,
        max_age_s=0.2,
        min_stamp_s=9.9,
    )

    assert relative is not None
    np.testing.assert_allclose(relative.translation_port, expected_position, atol=1e-9)
    np.testing.assert_allclose(
        relative.rotation_port_from_plug, expected_rotation, atol=1e-8
    )
    np.testing.assert_allclose(relative.axis_port, expected_rotation[:, 2], atol=1e-8)
    assert relative.view_count == 3
    assert np.isclose(relative.age_s, 0.05)
    assert estimator.estimate_relative_to_port(
        views,
        np.zeros(3),
        np.eye(3),
        now_s=10.5,
        max_age_s=0.2,
    ) is None
    assert estimator.estimate_relative_to_port(
        views,
        np.zeros(3),
        np.eye(3),
        now_s=10.05,
        min_stamp_s=10.0,
    ) is None


def test_estimator_requires_two_unique_cameras():
    views, detections, _, _ = _synthetic_views_and_detections()
    estimator = SfpPlugPoseEstimator(model=_FakeYolo(detections[:1]))

    assert estimator.estimate_multiview(views[:1]) is None


def test_estimator_fails_closed_on_model_runtime_error():
    class BrokenModel:
        def __call__(self, _images, **_kwargs):
            raise RuntimeError("inference backend unavailable")

    views, _, _, _ = _synthetic_views_and_detections()
    estimator = SfpPlugPoseEstimator(model=BrokenModel())

    assert estimator.estimate_multiview(views) is None
