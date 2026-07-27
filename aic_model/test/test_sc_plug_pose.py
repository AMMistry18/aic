"""Tests for the fail-closed SC plug-pose estimator.

These mirror ``test_sfp_plug_pose.py`` but pin the properties that matter for
SC specifically: that the estimator fits the *SC* keypoint geometry, that it
returns the ``sc_tip_link`` origin with no offset applied, and that every
fail-closed path still refuses to produce a pose.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# The pixi env carries an older installed copy of aic_model in site-packages
# that predates the plug-pose modules, which is why test_sfp_plug_pose.py and
# test_sc_plug_pose_geometry.py currently fail to collect.  Resolve the source
# tree first, the same way eval_sfp_plug_pose_model.py does at the repo root,
# so this file tests the code actually being edited.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
for _name in [n for n in sys.modules if n == "aic_model" or n.startswith("aic_model.")]:
    _module = sys.modules[_name]
    _file = getattr(_module, "__file__", None) or ""
    if not _file.startswith(str(_SOURCE_ROOT)):
        del sys.modules[_name]

from aic_model.sc_plug_pose import (  # noqa: E402
    ScPlugPoseEstimator,
    default_sc_plug_pose_weights,
    load_sc_plug_pose_estimator,
)
from aic_model.sc_plug_pose_geometry import (  # noqa: E402
    SC_PLUG_LOCAL_KEYPOINTS_M,
    project_keypoints,
)
from aic_model.sfp_plug_pose import (  # noqa: E402
    PlugKeypointDetection,
    PlugPoseView,
    SfpPlugPoseEstimator,
)
from aic_model.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M  # noqa: E402


K = np.array([[1236.6, 0.0, 576.0], [0.0, 1236.6, 512.0], [0.0, 0.0, 1.0]])


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


def _truth_pose():
    world_from_plug = np.eye(4)
    world_from_plug[:3, :3] = _rotation_xyz(0.09, -0.06, 0.15)
    world_from_plug[:3, 3] = np.array([0.012, -0.008, 0.278])
    return world_from_plug


def _views_and_detections(stamp=10.0, noise_px=0.0, seed=7, cameras=None):
    """Synthetic three-camera observation of the SC plug at a realistic range."""

    rng = np.random.default_rng(seed)
    world_from_plug = _truth_pose()
    if cameras is None:
        cameras = (
            ("left_camera", [-0.100, 0.0, 0.0]),
            ("center_camera", [0.0, 0.010, 0.0]),
            ("right_camera", [0.100, 0.0, 0.0]),
        )
    views, detections = [], []
    for name, camera_position in cameras:
        world_from_camera = np.eye(4)
        world_from_camera[:3, 3] = camera_position
        camera_from_plug = np.linalg.inv(world_from_camera) @ world_from_plug
        pixels, _ = project_keypoints(SC_PLUG_LOCAL_KEYPOINTS_M, camera_from_plug, K)
        if noise_px:
            pixels = pixels + rng.normal(0.0, noise_px, size=pixels.shape)
        views.append(
            PlugPoseView(
                camera_name=name,
                image_bgr=np.zeros((1024, 1152, 3), dtype=np.uint8),
                K=K,
                T_world_from_camera=world_from_camera,
                stamp_s=stamp,
                frame_id=f"{name}/optical",
            )
        )
        detections.append(
            PlugKeypointDetection(
                camera_name=name,
                keypoints_px=pixels,
                keypoint_confidences=np.ones(len(pixels)) * 0.9,
                box_confidence=0.9,
            )
        )
    return views, detections, world_from_plug


def _estimator(**kwargs):
    # model=<sentinel> bypasses the checkpoint requirement; detections are
    # injected directly so these tests never touch ultralytics.
    return ScPlugPoseEstimator(model=SimpleNamespace(), **kwargs)


def test_estimator_uses_sc_keypoints_not_sfp():
    estimator = _estimator()
    assert np.array_equal(estimator.local_keypoints_m, SC_PLUG_LOCAL_KEYPOINTS_M)
    assert not np.array_equal(estimator.local_keypoints_m, SFP_PLUG_LOCAL_KEYPOINTS_M)


def test_sfp_estimator_default_is_unchanged():
    """Parameterising the shared class must not move the SFP default."""

    estimator = SfpPlugPoseEstimator(model=SimpleNamespace())
    assert np.array_equal(estimator.local_keypoints_m, SFP_PLUG_LOCAL_KEYPOINTS_M)


def test_recovers_exact_pose_from_clean_keypoints():
    views, detections, truth = _views_and_detections()
    estimator = _estimator()
    estimate = estimator.estimate_multiview(views, detections=detections)
    assert estimate is not None
    # The fitted translation is the sc_tip_link origin itself: no TCP->tip
    # constant is applied anywhere in this path.
    assert np.allclose(estimate.position_world, truth[:3, 3], atol=1e-9)
    assert np.allclose(estimate.rotation_world_from_plug, truth[:3, :3], atol=1e-9)
    assert estimate.triangulated_keypoint_count == 8


def test_estimate_tip_pose_returns_controller_shaped_tuple():
    views, detections, truth = _views_and_detections()
    estimator = _estimator()
    estimator._predict = lambda selected: detections  # noqa: SLF001
    result = estimator.estimate_tip_pose(views)
    assert result is not None
    tip_position, tip_rotation = result
    assert np.allclose(tip_position, truth[:3, 3], atol=1e-9)
    assert np.allclose(tip_rotation, truth[:3, :3], atol=1e-9)
    assert tip_rotation.shape == (3, 3)


def test_moderate_pixel_noise_stays_inside_working_target():
    """2 px keypoint noise must still land near the 0.4 mm working target."""

    errors_mm = []
    for seed in range(40):
        views, detections, truth = _views_and_detections(noise_px=2.0, seed=seed)
        estimate = _estimator().estimate_multiview(views, detections=detections)
        assert estimate is not None
        errors_mm.append(
            float(np.linalg.norm(estimate.position_world - truth[:3, 3]) * 1000.0)
        )
    assert float(np.median(errors_mm)) < 0.6


def test_fails_closed_with_one_camera():
    views, detections, _ = _views_and_detections()
    estimate = _estimator().estimate_multiview(views[:1], detections=detections[:1])
    assert estimate is None


def test_fails_closed_on_duplicate_camera_names():
    views, detections, _ = _views_and_detections()
    duplicated = [views[0], views[0], views[2]]
    assert _estimator().estimate_multiview(duplicated, detections=detections) is None


def test_fails_closed_on_stale_frames():
    views, detections, _ = _views_and_detections(stamp=10.0)
    estimator = _estimator()
    assert (
        estimator.estimate_multiview(
            views, detections=detections, now_s=10.05, max_age_s=0.35
        )
        is not None
    )
    assert (
        estimator.estimate_multiview(
            views, detections=detections, now_s=99.0, max_age_s=0.35
        )
        is None
    )


def test_fails_closed_on_unsynchronized_frames():
    views, detections, _ = _views_and_detections()
    skewed = [
        PlugPoseView(
            camera_name=view.camera_name,
            image_bgr=view.image_bgr,
            K=view.K,
            T_world_from_camera=view.T_world_from_camera,
            stamp_s=view.stamp_s + (5.0 if index == 0 else 0.0),
            frame_id=view.frame_id,
        )
        for index, view in enumerate(views)
    ]
    assert _estimator().estimate_multiview(skewed, detections=detections) is None


def test_fails_closed_when_geometry_does_not_match_the_plug():
    """Keypoints from a differently shaped body must be rejected, not fitted."""

    views, detections, _ = _views_and_detections()
    wrong = [
        PlugKeypointDetection(
            camera_name=detection.camera_name,
            keypoints_px=detection.keypoints_px + np.linspace(0, 90, 16).reshape(8, 2),
            keypoint_confidences=detection.keypoint_confidences,
            box_confidence=detection.box_confidence,
        )
        for detection in detections
    ]
    assert _estimator().estimate_multiview(views, detections=wrong) is None


def test_fails_closed_when_too_few_keypoints_are_confident():
    views, detections, _ = _views_and_detections()
    sparse = [
        PlugKeypointDetection(
            camera_name=detection.camera_name,
            keypoints_px=detection.keypoints_px,
            keypoint_confidences=np.array([0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            box_confidence=detection.box_confidence,
        )
        for detection in detections
    ]
    assert _estimator().estimate_multiview(views, detections=sparse) is None


def test_requires_now_s_when_max_age_is_given():
    views, detections, _ = _views_and_detections()
    # The ValueError is swallowed by the fail-closed wrapper, so the contract
    # is that no pose comes back rather than a silently unchecked one.
    assert (
        _estimator().estimate_multiview(views, detections=detections, max_age_s=0.2)
        is None
    )


def test_rejects_attempt_to_override_keypoint_geometry():
    with pytest.raises(TypeError):
        ScPlugPoseEstimator(
            model=SimpleNamespace(), local_keypoints_m=SFP_PLUG_LOCAL_KEYPOINTS_M
        )


def test_missing_weights_yields_no_estimator_rather_than_a_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_SC_PLUG_POSE_WEIGHTS", str(tmp_path / "absent.pt"))
    assert default_sc_plug_pose_weights() is None
    assert load_sc_plug_pose_estimator() is None


def test_explicit_missing_weights_path_raises():
    with pytest.raises(FileNotFoundError):
        ScPlugPoseEstimator("/nonexistent/best_sc_plug_pose.pt")


# ---------------------------------------------------------------------------
# Crop-refine second pass.  The controller enables this in the field (it is
# what reaches the 0.27 mm median), and until now it was only ever validated
# offline on TACC -- pin its mechanics here.
# ---------------------------------------------------------------------------
class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeBoxes:
    def __init__(self, confs, xyxy=None):
        self.conf = _FakeTensor(confs)
        if xyxy is not None:
            self.xyxy = _FakeTensor(xyxy)
        self._count = len(np.atleast_1d(confs))

    def __len__(self):
        return self._count


class _FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = _FakeTensor(xy)
        self.conf = _FakeTensor(conf)


class _FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _FakeYolo:
    """Callable standing in for the Ultralytics model inside ``_predict``.

    Programmed with one response list per expected call; records every batch
    of images it is handed so tests can assert the crop geometry.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, images, **_kwargs):
        self.calls.append([np.asarray(image) for image in images])
        return self._responses.pop(0)


def _one_view():
    views, _, _ = _views_and_detections()
    return [views[1]]


def _result_with(keypoints_xy, box_xyxy=None, conf=0.9):
    kp = np.asarray(keypoints_xy, dtype=np.float64)[None, :, :]
    kp_conf = np.full((1, kp.shape[1]), 0.9)
    boxes = _FakeBoxes([conf], xyxy=None if box_xyxy is None else [box_xyxy])
    return _FakeResult(boxes=boxes, keypoints=_FakeKeypoints(kp, kp_conf))


def test_crop_refine_remaps_crop_keypoints_by_pure_translation(monkeypatch):
    monkeypatch.delenv("AIC_PLUG_POSE_CROP_PAD", raising=False)
    truth = np.array([[520.0 + 3.0 * i, 420.0 + 2.0 * i] for i in range(8)])
    box = [500.0, 400.0, 540.0, 440.0]
    # pad 6 on a 40 px box -> half=120 -> crop origin (400, 300), inside frame.
    origin = np.array([400.0, 300.0])
    model = _FakeYolo([
        [_result_with(truth + 5.0, box_xyxy=box)],       # full frame, biased
        [_result_with(truth - origin, box_xyxy=[100.0, 100.0, 140.0, 140.0])],
    ])
    estimator = ScPlugPoseEstimator(model=model, crop_refine=True)

    detections = estimator.detect_views(_one_view())

    assert len(model.calls) == 2, "crop refine must run a second pass"
    assert model.calls[1][0].shape == (240, 240, 3), "padded crop, native res"
    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].keypoints_px, truth, atol=1e-9)


def test_crop_refine_keeps_full_frame_result_when_crop_misses(monkeypatch):
    monkeypatch.delenv("AIC_PLUG_POSE_CROP_PAD", raising=False)
    full = np.array([[520.0 + i, 420.0 + i] for i in range(8)])
    model = _FakeYolo([
        [_result_with(full, box_xyxy=[500.0, 400.0, 540.0, 440.0])],
        [_FakeResult(boxes=None, keypoints=None)],       # crop re-detect fails
    ])
    estimator = ScPlugPoseEstimator(model=model, crop_refine=True)

    detections = estimator.detect_views(_one_view())

    assert len(model.calls) == 2
    assert len(detections) == 1, "a failed crop must not drop the view"
    np.testing.assert_allclose(detections[0].keypoints_px, full, atol=1e-9)


def test_crop_refine_defaults_off_and_runs_a_single_pass(monkeypatch):
    monkeypatch.delenv("AIC_PLUG_POSE_CROP_REFINE", raising=False)
    full = np.array([[520.0 + i, 420.0 - i] for i in range(8)])
    model = _FakeYolo([[_result_with(full, box_xyxy=[500.0, 400.0, 540.0, 440.0])]])
    estimator = ScPlugPoseEstimator(model=model)

    assert estimator.crop_refine is False, "SFP-path safety: off unless asked"
    detections = estimator.detect_views(_one_view())

    assert len(model.calls) == 1, "no second inference when disabled"
    np.testing.assert_allclose(detections[0].keypoints_px, full, atol=1e-9)


def test_crop_refine_tolerates_results_without_boxes_xyxy(monkeypatch):
    # Guards the guarded ``getattr(result.boxes, "xyxy", None)``: a result that
    # only carries confidences must skip refinement, not crash mid-run.
    monkeypatch.delenv("AIC_PLUG_POSE_CROP_PAD", raising=False)
    full = np.array([[520.0 + i, 420.0 + 2.0 * i] for i in range(8)])
    model = _FakeYolo([[_result_with(full, box_xyxy=None)]])
    estimator = ScPlugPoseEstimator(model=model, crop_refine=True)

    detections = estimator.detect_views(_one_view())

    assert len(model.calls) == 1, "nothing to crop without a box"
    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].keypoints_px, full, atol=1e-9)
