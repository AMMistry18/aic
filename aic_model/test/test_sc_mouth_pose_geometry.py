from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE_ROOT = REPO_ROOT / "aic_example_policies"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aic_example_policies.ros.sc_mouth_pose_geometry import (  # noqa: E402
    LOCAL_SC_FRONT_MOUTH_KPS_M,
    SC_FRONT_MOUTH_HEIGHT_M,
    SC_FRONT_MOUTH_WIDTH_M,
    SC_MOUTH_FLIP_IDX,
    format_yolo_pose_label,
    split_for_trial,
    write_dataset_yaml,
)
from tools.perception.sc_mouth.train import EXPECTED_LABEL_TOKENS  # noqa: E402


def test_physical_front_mouth_is_distinct_from_legacy_virtual_target():
    assert LOCAL_SC_FRONT_MOUTH_KPS_M.shape == (5, 3)
    np.testing.assert_allclose(LOCAL_SC_FRONT_MOUTH_KPS_M[4], [0.0, 0.0, 0.0])
    spans = LOCAL_SC_FRONT_MOUTH_KPS_M[:4].max(axis=0) - LOCAL_SC_FRONT_MOUTH_KPS_M[:4].min(axis=0)
    np.testing.assert_allclose(spans, [SC_FRONT_MOUTH_WIDTH_M, SC_FRONT_MOUTH_HEIGHT_M, 0.0])
    assert SC_FRONT_MOUTH_WIDTH_M == 0.022407
    assert SC_FRONT_MOUTH_HEIGHT_M == 0.00810
    assert SC_FRONT_MOUTH_WIDTH_M > 2.5 * 0.0088
    assert SC_MOUTH_FLIP_IDX == [1, 0, 3, 2, 4]


def test_physical_mouth_labels_have_exact_five_keypoint_schema():
    pixels = np.array(
        [[410.0, 300.0], [230.0, 300.0], [230.0, 365.0], [410.0, 365.0], [320.0, 332.5]]
    )
    label = format_yolo_pose_label((210.0, 280.0, 430.0, 385.0), pixels, np.full(5, 2), 640, 480)
    tokens = label.split()
    assert len(tokens) == EXPECTED_LABEL_TOKENS == 20
    assert tokens[0] == "0"
    assert [int(tokens[7 + 3 * index]) for index in range(5)] == [2] * 5


def test_trial_split_is_stable_across_resumed_batches():
    all_splits = [split_for_trial(index) for index in range(1, 91)]
    assert all_splits.count("train") == 72
    assert all_splits.count("val") == 9
    assert all_splits.count("test") == 9
    assert all_splits[30:60] == [split_for_trial(index) for index in range(31, 61)]


def test_dataset_yaml_declares_new_geometry_and_held_out_test_split(tmp_path: Path):
    output = write_dataset_yaml(tmp_path)
    content = output.read_text(encoding="utf-8")
    assert output.name == "aic_sc_mouth_pose.yaml"
    assert "physical SC front mouth" in content
    assert "kpt_shape: [5, 3]" in content
    assert "test: images/test" in content
