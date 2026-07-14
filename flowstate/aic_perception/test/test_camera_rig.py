from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from aic_perception.camera_rig import (
    CameraRig,
    ros_image_to_bgr,
    stamp_to_nanoseconds,
)
from aic_perception.config import CAMERA_NAMES


def image_message(array, encoding="bgr8", padding=0, stamp_ns=123):
    image = np.asarray(array, dtype=np.uint8)
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape
    row_bytes = width * channels
    step = row_bytes + padding
    data = np.zeros((height, step), dtype=np.uint8)
    data[:, :row_bytes] = image.reshape(height, row_bytes)
    return SimpleNamespace(
        height=height,
        width=width,
        step=step,
        encoding=encoding,
        data=data.tobytes(),
        header=SimpleNamespace(
            frame_id="camera/optical",
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            ),
        ),
    )


def make_rig():
    rig = CameraRig.__new__(CameraRig)
    rig._condition = threading.Condition()
    rig._sequences = {camera: 0 for camera in CAMERA_NAMES}
    rig._latest_frames = {}
    rig._force_xyz = None
    rig._force_received_at = None
    return rig


def test_bgr_conversion_preserves_pixels_and_ignores_padding():
    source = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    converted = ros_image_to_bgr(image_message(source, padding=7))
    np.testing.assert_array_equal(converted, source)
    assert converted.flags.c_contiguous


def test_rgb_conversion_swaps_red_and_blue():
    rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    converted = ros_image_to_bgr(image_message(rgb, encoding="rgb8"))
    np.testing.assert_array_equal(converted, [[[3, 2, 1]]])


def test_mono_conversion_returns_two_dimensions():
    mono = np.arange(12, dtype=np.uint8).reshape(3, 4)
    converted = ros_image_to_bgr(image_message(mono, encoding="mono8"))
    np.testing.assert_array_equal(converted, mono)


def test_unsupported_encoding_is_rejected():
    message = image_message(np.zeros((2, 2, 3), dtype=np.uint8))
    message.encoding = "16UC1"
    with pytest.raises(ValueError):
        ros_image_to_bgr(message)


def test_stamp_conversion():
    stamp = SimpleNamespace(sec=12, nanosec=345)
    assert stamp_to_nanoseconds(stamp) == 12_000_000_345


def test_grab_returns_only_frames_newer_than_call_start_and_latest_force():
    rig = make_rig()
    old = image_message(np.zeros((4, 5, 3), dtype=np.uint8), stamp_ns=1)
    rig._on_image("left_camera", old)
    wrench = SimpleNamespace(
        wrench=SimpleNamespace(force=SimpleNamespace(x=3.0, y=4.0, z=0.0))
    )
    rig._on_wrench(wrench)

    def publish_new_message():
        time.sleep(0.02)
        fresh = image_message(np.zeros((4, 5, 3), dtype=np.uint8), stamp_ns=2)
        rig._on_image("center_camera", fresh)

    publisher = threading.Thread(target=publish_new_message)
    publisher.start()
    snapshot = rig.grab(timeout_sec=1.5)
    publisher.join()
    assert snapshot is not None
    assert set(snapshot.frames) == {"center_camera"}
    assert snapshot.frames["center_camera"]["stamp_ns"] == 2
    assert snapshot.force_norm == 5.0
    assert rig.latest_force_xyz() == (3.0, 4.0, 0.0)
    assert rig.latest_force_norm() == 5.0


def test_grab_can_wait_for_multiple_fresh_cameras():
    rig = make_rig()

    def publish():
        time.sleep(0.01)
        message = image_message(np.zeros((2, 2, 3), dtype=np.uint8))
        rig._on_image("left_camera", message)
        rig._on_image("right_camera", message)

    publisher = threading.Thread(target=publish)
    publisher.start()
    snapshot = rig.grab(timeout_sec=0.5, min_cameras=2)
    publisher.join()
    assert snapshot is not None
    assert set(snapshot.frames) == {"left_camera", "right_camera"}


def test_grab_collects_all_three_cameras_during_grace_period():
    rig = make_rig()

    def publish():
        message = image_message(np.zeros((2, 2, 3), dtype=np.uint8))
        time.sleep(0.01)
        rig._on_image("right_camera", message)
        time.sleep(0.03)
        rig._on_image("center_camera", message)
        time.sleep(0.03)
        rig._on_image("left_camera", message)

    publisher = threading.Thread(target=publish)
    publisher.start()
    snapshot = rig.grab(timeout_sec=0.5, collection_grace_sec=0.15)
    publisher.join()
    assert snapshot is not None
    assert set(snapshot.frames) == set(CAMERA_NAMES)


def test_grab_returns_available_cameras_after_collection_grace():
    rig = make_rig()

    def publish():
        time.sleep(0.01)
        message = image_message(np.zeros((2, 2, 3), dtype=np.uint8))
        rig._on_image("left_camera", message)
        rig._on_image("center_camera", message)

    publisher = threading.Thread(target=publish)
    publisher.start()
    started = time.monotonic()
    snapshot = rig.grab(timeout_sec=1.0, collection_grace_sec=0.05)
    elapsed = time.monotonic() - started
    publisher.join()
    assert snapshot is not None
    assert set(snapshot.frames) == {"left_camera", "center_camera"}
    assert elapsed < 0.5


def test_grab_times_out_without_fresh_message():
    rig = make_rig()
    rig._on_image(
        "left_camera", image_message(np.zeros((2, 2, 3), dtype=np.uint8))
    )
    assert rig.grab(timeout_sec=0.01) is None


def test_stale_force_is_not_returned_as_motion_feedback():
    rig = make_rig()
    rig._force_xyz = (1.0, 2.0, 3.0)
    rig._force_received_at = time.monotonic() - 1.0
    assert rig.latest_force_xyz(max_age_sec=0.25) is None
    with pytest.raises(ValueError):
        rig.latest_force_xyz(max_age_sec=0.0)
