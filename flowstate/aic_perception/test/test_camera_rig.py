from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from aic_perception.camera_rig import (
    CameraSnapshot,
    CameraRig,
    approved_camera_message_frames,
    camera_info_to_calibration,
    frames_are_approved_camera_pair,
    ros_image_to_bgr,
    stamp_to_nanoseconds,
)
from aic_perception.config import CAMERA_NAMES, CAMERA_OPTICAL_FRAMES


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


def camera_info_message(
    *,
    width=5,
    height=4,
    camera="center_camera",
    stamp_ns=120,
):
    return SimpleNamespace(
        width=width,
        height=height,
        k=[500.0, 0.0, width / 2.0, 0.0, 501.0, height / 2.0, 0.0, 0.0, 1.0],
        d=[0.1, -0.2, 0.0, 0.0, 0.01],
        distortion_model="plumb_bob",
        header=SimpleNamespace(
            frame_id=CAMERA_OPTICAL_FRAMES[camera],
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
    rig._latest_calibrations = {}
    rig._camera_frames = dict(CAMERA_OPTICAL_FRAMES)
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
    with pytest.raises(ValueError):
        stamp_to_nanoseconds(SimpleNamespace(sec=0, nanosec=1_000_000_000))


def test_camera_info_conversion_validates_and_preserves_intrinsics():
    calibration = camera_info_to_calibration(
        camera_info_message(), CAMERA_OPTICAL_FRAMES["center_camera"]
    )
    assert calibration.width == 5
    assert calibration.height == 4
    assert calibration.frame_id == "center_camera/optical"
    assert calibration.distortion_model == "plumb_bob"
    np.testing.assert_allclose(
        calibration.camera_matrix,
        [[500.0, 0.0, 2.5], [0.0, 501.0, 2.0], [0.0, 0.0, 1.0]],
    )


def test_camera_info_rejects_wrong_frame_and_invalid_intrinsics():
    message = camera_info_message()
    message.header.frame_id = "task_board"
    with pytest.raises(ValueError):
        camera_info_to_calibration(
            message, CAMERA_OPTICAL_FRAMES["center_camera"]
        )


def test_basler_sensor_link_is_an_explicitly_accepted_optical_pair_only():
    optical = CAMERA_OPTICAL_FRAMES["center_camera"]
    assert approved_camera_message_frames(optical) == {
        "center_camera/optical",
        "center_camera/sensor_link",
    }
    message = camera_info_message()
    message.header.frame_id = "center_camera/sensor_link"
    calibration = camera_info_to_calibration(message, optical)
    assert calibration.frame_id == "center_camera/sensor_link"

    # This must stay an exact two-frame allowlist, not a camera-name prefix.
    for forbidden in ("center_camera/link", "left_camera/sensor_link", "task_board"):
        message.header.frame_id = forbidden
        with pytest.raises(ValueError):
            camera_info_to_calibration(message, optical)

    message = camera_info_message()
    message.k[0] = float("nan")
    with pytest.raises(ValueError):
        camera_info_to_calibration(
            message, CAMERA_OPTICAL_FRAMES["center_camera"]
        )


def test_image_and_camera_info_may_use_either_member_of_exact_pair():
    optical = CAMERA_OPTICAL_FRAMES["center_camera"]
    assert frames_are_approved_camera_pair(
        "center_camera/sensor_link", "center_camera/optical", optical
    )
    assert frames_are_approved_camera_pair(
        "center_camera/optical", "center_camera/sensor_link", optical
    )
    assert not frames_are_approved_camera_pair(
        "right_camera/sensor_link", "center_camera/optical", optical
    )


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


def test_grab_exposes_only_dimension_matched_available_calibration():
    rig = make_rig()
    rig._on_camera_info("center_camera", camera_info_message())
    rig._on_camera_info(
        "right_camera",
        camera_info_message(width=99, height=99, camera="right_camera"),
    )

    def publish():
        time.sleep(0.01)
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        center = image_message(frame, stamp_ns=125)
        center.header.frame_id = CAMERA_OPTICAL_FRAMES["center_camera"]
        right = image_message(frame, stamp_ns=130)
        right.header.frame_id = CAMERA_OPTICAL_FRAMES["right_camera"]
        rig._on_image("center_camera", center)
        rig._on_image("right_camera", right)

    publisher = threading.Thread(target=publish)
    publisher.start()
    snapshot = rig.grab(timeout_sec=0.5, min_cameras=2)
    publisher.join()
    assert snapshot is not None
    assert set(snapshot.calibrations) == {"center_camera"}
    assert snapshot.calibrations["center_camera"].stamp_ns == 120


@pytest.mark.parametrize(
    ("image_suffix", "info_suffix"),
    (("sensor_link", "optical"), ("optical", "sensor_link")),
)
def test_grab_accepts_exact_sensor_optical_frame_pair(
    image_suffix, info_suffix
):
    rig = make_rig()
    info = camera_info_message()
    info.header.frame_id = f"center_camera/{info_suffix}"
    rig._on_camera_info("center_camera", info)

    def publish():
        time.sleep(0.01)
        frame = image_message(np.zeros((4, 5, 3), dtype=np.uint8))
        frame.header.frame_id = f"center_camera/{image_suffix}"
        rig._on_image("center_camera", frame)

    publisher = threading.Thread(target=publish)
    publisher.start()
    snapshot = rig.grab(timeout_sec=0.5)
    publisher.join()
    assert snapshot is not None
    assert set(snapshot.calibrations) == {"center_camera"}


def test_grab_rejects_image_frame_outside_exact_camera_pair():
    rig = make_rig()
    rig._on_camera_info("center_camera", camera_info_message())

    def publish():
        time.sleep(0.01)
        frame = image_message(np.zeros((4, 5, 3), dtype=np.uint8))
        frame.header.frame_id = "right_camera/sensor_link"
        rig._on_image("center_camera", frame)

    publisher = threading.Thread(target=publish)
    publisher.start()
    snapshot = rig.grab(timeout_sec=0.5)
    publisher.join()
    assert snapshot is not None
    assert not snapshot.calibrations


def test_snapshot_reports_and_bounds_frame_timestamp_skew():
    snapshot = CameraSnapshot(
        frames={
            "left_camera": {"stamp_ns": 100},
            "center_camera": {"stamp_ns": 125},
            "right_camera": {"stamp_ns": 140},
        },
        force_xyz=None,
    )
    assert snapshot.frame_stamp_skew_ns == 40
    assert snapshot.frames_within_skew(40)
    assert not snapshot.frames_within_skew(39)
    with pytest.raises(ValueError):
        snapshot.frames_within_skew(-1)

    one_frame = CameraSnapshot(
        frames={"center_camera": {"stamp_ns": 100}},
        force_xyz=None,
    )
    assert one_frame.frame_stamp_skew_ns is None
    assert not one_frame.frames_within_skew(1_000)


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


def test_wait_for_force_returns_next_fresh_sample():
    rig = make_rig()
    rig._force_xyz = (9.0, 9.0, 9.0)
    rig._force_received_at = time.monotonic() - 1.0

    def publish_force():
        time.sleep(0.02)
        rig._on_wrench(
            SimpleNamespace(
                wrench=SimpleNamespace(
                    force=SimpleNamespace(x=3.0, y=4.0, z=0.0)
                )
            )
        )

    publisher = threading.Thread(target=publish_force)
    publisher.start()
    force = rig.wait_for_force_xyz(timeout_sec=0.5, max_age_sec=0.1)
    publisher.join()
    assert force == (3.0, 4.0, 0.0)


def test_wait_for_force_times_out_without_fresh_sample():
    rig = make_rig()
    assert rig.wait_for_force_xyz(timeout_sec=0.01) is None
    with pytest.raises(ValueError):
        rig.wait_for_force_xyz(timeout_sec=-0.1)
