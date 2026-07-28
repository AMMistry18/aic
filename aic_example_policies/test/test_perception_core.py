"""Headless mechanics tests for SC-port crop-refine inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SOURCE_ROOT))

from aic_example_policies.ros.perception_core import PerceptionCore


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)

    def __len__(self):
        return len(self.conf.value)


class _Keypoints:
    def __init__(self, xy):
        self.xy = _Tensor(xy)


class _Result:
    def __init__(self, xyxy=None, conf=None, kps=None):
        self.boxes = None if xyxy is None else _Boxes(xyxy, conf)
        self.keypoints = None if kps is None else _Keypoints(kps)


class _FakeYolo:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append((images, kwargs))
        return self.responses.pop(0)


class _FailOnRefineYolo(_FakeYolo):
    def __call__(self, images, **kwargs):
        if self.calls:
            raise RuntimeError("synthetic crop batch failure")
        return super().__call__(images, **kwargs)


def _quad(x, y):
    return np.array(
        [
            [x, y],
            [x + 4, y],
            [x + 4, y + 3],
            [x, y + 3],
            [x + 2, y + 1.5],
        ],
        dtype=float,
    )


def _core(model):
    core = PerceptionCore(sc_weights="unused.pt")
    core._sc_yolo = model
    return core


def test_sc_crop_refine_uses_native_crop_and_translates_keypoints():
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    full_box = [40, 30, 60, 50]
    full_kps = _quad(45, 35)
    # pad=2 creates native crop [30:70, 20:60].
    crop_kps = _quad(15, 15)
    model = _FakeYolo([
        [_Result([full_box], [0.9], [full_kps])],
        [_Result([[10, 10, 30, 30]], [0.8], [crop_kps])],
    ])

    detections = _core(model).detect_sc_pose(image, crop_refine=True, crop_pad_scale=2.0)

    assert len(model.calls) == 2
    assert model.calls[1][0][0].shape == (40, 40, 3)
    assert detections[0]["bbox"] == (40, 30, 20, 20)
    np.testing.assert_allclose(detections[0]["kps"], full_kps)
    np.testing.assert_allclose(detections[0]["mouth_center"], full_kps[4])
    assert detections[0]["centroid"] == tuple(full_kps[4])
    assert detections[0]["area"] == 400


def test_sc_crop_refine_keeps_each_port_identity_over_higher_conf_distractor():
    image = np.zeros((140, 200, 3), dtype=np.uint8)
    first_box = [20, 20, 40, 40]
    second_box = [120, 60, 140, 80]
    first_kps = _quad(25, 25)
    second_kps = _quad(125, 65)
    # In each crop the 0.99 candidate is a different, non-overlapping port.
    # The lower-confidence candidate is the only one that overlaps the origin
    # box and must be selected.
    model = _FakeYolo([
        [_Result([first_box, second_box], [0.9, 0.85], [first_kps, second_kps])],
        [
            _Result(
                [[0, 0, 8, 8], [10, 10, 30, 30]], [0.99, 0.4],
                [_quad(1, 1), _quad(15, 15)],
            ),
            _Result(
                [[0, 0, 8, 8], [10, 10, 30, 30]], [0.99, 0.5],
                [_quad(1, 1), _quad(15, 15)],
            ),
        ],
    ])

    detections = _core(model).detect_sc_pose(image, crop_refine=True, crop_pad_scale=2.0)

    assert len(model.calls) == 2
    assert len(model.calls[1][0]) == 2, "all coarse ports get their own crop"
    by_centre_x = {round(det["centroid"][0]): det for det in detections}
    assert set(by_centre_x) == {27, 127}
    assert by_centre_x[27]["conf"] == 0.4
    assert by_centre_x[127]["conf"] == 0.5
    np.testing.assert_allclose(by_centre_x[27]["kps"], first_kps)
    np.testing.assert_allclose(by_centre_x[127]["kps"], second_kps)


def test_sc_crop_refine_falls_back_when_no_crop_candidate_associates():
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    full_box = [40, 30, 60, 50]
    full_kps = _quad(45, 35)
    model = _FakeYolo([
        [_Result([full_box], [0.9], [full_kps])],
        [_Result([[0, 0, 8, 8]], [0.99], [_quad(1, 1)])],
    ])

    detections = _core(model).detect_sc_pose(image, crop_refine=True, crop_pad_scale=2.0)

    assert len(detections) == 1
    assert detections[0]["conf"] == 0.9
    np.testing.assert_allclose(detections[0]["kps"], full_kps)


def test_sc_crop_refine_falls_back_when_crop_inference_raises():
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    full_kps = _quad(45, 35)
    model = _FailOnRefineYolo([
        [_Result([[40, 30, 60, 50]], [0.9], [full_kps])],
    ])

    detections = _core(model).detect_sc_pose(
        image, crop_refine=True, crop_pad_scale=2.0
    )

    assert detections[0]["conf"] == 0.9
    np.testing.assert_allclose(detections[0]["kps"], full_kps)


def test_sc_crop_refine_defaults_to_measured_pad_24(monkeypatch):
    monkeypatch.delenv("AIC_SC_POSE_CROP_REFINE", raising=False)
    monkeypatch.delenv("AIC_SC_POSE_CROP_PAD", raising=False)
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    full_kps = _quad(45, 35)
    model = _FakeYolo([
        [_Result([[40, 30, 60, 50]], [0.9], [full_kps])],
        [_Result([[40, 30, 60, 50]], [0.8], [full_kps])],
    ])

    detections = _core(model).detect_sc_pose(image)

    assert len(model.calls) == 2
    assert len(detections) == 1


def test_sc_crop_refine_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("AIC_SC_POSE_CROP_REFINE", "0")
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    model = _FakeYolo([[_Result([[40, 30, 60, 50]], [0.9], [_quad(45, 35)])]])

    detections = _core(model).detect_sc_pose(image)

    assert len(model.calls) == 1
    assert len(detections) == 1
