from __future__ import annotations

import cv2
import numpy as np
import pytest

from aic_perception.board_visibility import analyze_board
from aic_perception.gripper_masks import GripperMaskBank


@pytest.mark.parametrize(
    ("camera", "ignored_count"),
    (
        ("left_camera", 4611),
        ("center_camera", 4531),
        ("right_camera", 4787),
    ),
)
def test_embedded_masks_match_upstream_calibration(camera, ignored_count):
    ignored = GripperMaskBank().ignored_pixels(camera, (213, 239, 3))
    assert ignored.dtype == np.bool_
    assert ignored.shape == (213, 239)
    assert int(ignored.sum()) == ignored_count


def test_masks_resize_and_return_independent_arrays():
    bank = GripperMaskBank()
    first = bank.ignored_pixels("center_camera", (1024, 1152, 3))
    second = bank.ignored_pixels("center_camera", (1024, 1152, 3))
    assert first.shape == (1024, 1152)
    assert first.sum() > 0
    first[:] = False
    assert second.sum() > 0


def test_gripper_pixels_cannot_win_board_component_selection():
    bank = GripperMaskBank()
    ignored = bank.ignored_pixels("center_camera", (213, 239, 3))
    image = np.full((213, 239, 3), 210, dtype=np.uint8)
    image[ignored] = (35, 35, 35)
    cv2.rectangle(image, (15, 45), (75, 100), (45, 45, 45), -1)

    unmasked = analyze_board(
        image, ignore_bottom_frac=0.0, margin_px=5, context_pad_frac=0.02
    )
    masked = analyze_board(
        image,
        ignore_bottom_frac=0.0,
        margin_px=5,
        context_pad_frac=0.02,
        ignore_mask=ignored,
    )

    assert unmasked.bbox is not None and unmasked.bbox[1] >= 150
    assert masked.bbox is not None and masked.bbox[1] < 60
    assert masked.bbox[3] < 110


def test_board_contact_with_gripper_mask_does_not_hide_complete_view():
    bank = GripperMaskBank()
    ignored = bank.ignored_pixels("center_camera", (213, 239, 3))
    image = np.full((213, 239, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (45, 70), (190, 195), (45, 45, 45), -1)

    report = analyze_board(
        image,
        ignore_bottom_frac=0.0,
        margin_px=5,
        context_pad_frac=0.02,
        ignore_mask=ignored,
    )

    assert report.seen and report.full
    assert report.artificial_bottom_contact
    assert "artificial_bottom_contact" not in report.failure_reasons


def test_unknown_camera_is_rejected():
    with pytest.raises(KeyError):
        GripperMaskBank().ignored_pixels("world_camera", (213, 239, 3))
