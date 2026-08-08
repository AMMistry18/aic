"""Metric tests for the headless SC-port crop-refine evaluator."""

from __future__ import annotations

import numpy as np

from tools.perception.sc_port.evaluate_model import (
    cyclic_keypoint_errors,
    globally_match_detections,
)


def _label(center_x):
    # 100x100 image, broad bbox so both assignment choices pass the distance gate.
    points = [
        (center_x - 2, 48),
        (center_x + 2, 48),
        (center_x + 2, 52),
        (center_x - 2, 52),
    ]
    tokens = ["0", f"{center_x / 100:.6f}", "0.5", "0.4", "0.4"]
    for x, y in points:
        tokens.extend([f"{x / 100:.6f}", f"{y / 100:.6f}", "2"])
    return tokens


def _detection(center_x):
    return {
        "centroid": (center_x, 50.0),
        "kps": np.array(
            [
                [center_x - 2, 48],
                [center_x + 2, 48],
                [center_x + 2, 52],
                [center_x - 2, 52],
            ],
            dtype=np.float64,
        ),
    }


def test_cyclic_keypoint_error_accepts_runtime_corner_relabeling():
    ground_truth = _detection(50)["kps"]
    predicted = np.roll(ground_truth, 2, axis=0)

    roll, errors = cyclic_keypoint_errors(predicted, ground_truth, np.ones(4, dtype=bool))

    assert roll == 2
    np.testing.assert_allclose(errors, 0.0)


def test_detection_matching_uses_global_assignment():
    # Greedy closest-pair matching chooses GT 30 -> det 29 first, leaving the
    # worse GT 20 -> det 35 pairing.  The global optimum is 20 -> 29, 30 -> 35.
    pairs = globally_match_detections(
        [_label(20), _label(30)],
        [_detection(29), _detection(35)],
        100,
        100,
    )

    assert [(gt, det) for gt, det, _distance in pairs] == [(0, 0), (1, 1)]
