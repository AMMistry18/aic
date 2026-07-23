"""Fresh wrist-camera frames and optional force data for one skill call."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np

from .config import CAMERA_NAMES, PerceptionConfig


@dataclass(frozen=True)
class CameraSnapshot:
    """A fresh set of approved wrist-camera frames and the latest force."""

    frames: dict[str, dict[str, Any]]
    force_xyz: tuple[float, float, float] | None

    @property
    def force_norm(self) -> float | None:
        if self.force_xyz is None:
            return None
        return float(np.linalg.norm(np.asarray(self.force_xyz, dtype=float)))


def ros_image_to_bgr(message: Any) -> np.ndarray:
    """Convert a ROS ``sensor_msgs/Image`` into a contiguous BGR/gray array."""
    import cv2

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError("image dimensions and step must be positive")

    channels_by_encoding = {
        "mono8": 1,
        "8uc1": 1,
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding: {message.encoding!r}")
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError("image step is shorter than one pixel row")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("image data is shorter than height * step")
    pixels = raw[: height * step].reshape(height, step)[:, :row_bytes]
    if channels == 1:
        return np.ascontiguousarray(pixels.reshape(height, width))

    image = np.ascontiguousarray(pixels.reshape(height, width, channels))
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def stamp_to_nanoseconds(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class CameraRig:
    """Subscribe only to the three approved images and wrist wrench topic."""

    def __init__(self, node: Any, config: PerceptionConfig):
        from geometry_msgs.msg import WrenchStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        self._condition = threading.Condition()
        self._sequences = {camera: 0 for camera in CAMERA_NAMES}
        self._latest_frames: dict[str, dict[str, Any]] = {}
        self._force_xyz: tuple[float, float, float] | None = None
        self._force_received_at: float | None = None
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscriptions = [
            node.create_subscription(
                Image,
                config.image_topics[camera],
                lambda message, name=camera: self._on_image(name, message),
                qos,
            )
            for camera in config.cameras
        ]
        self._subscriptions.append(
            node.create_subscription(
                WrenchStamped, config.wrench_topic, self._on_wrench, qos
            )
        )

    def _on_image(self, camera: str, message: Any) -> None:
        try:
            frame = {
                "image": ros_image_to_bgr(message),
                "stamp_ns": stamp_to_nanoseconds(message.header.stamp),
                "frame_id": str(getattr(message.header, "frame_id", "")),
            }
        except (AttributeError, TypeError, ValueError):
            return
        with self._condition:
            self._latest_frames[camera] = frame
            self._sequences[camera] += 1
            self._condition.notify_all()

    def _on_wrench(self, message: Any) -> None:
        try:
            force = message.wrench.force
            values = (float(force.x), float(force.y), float(force.z))
        except (AttributeError, TypeError, ValueError):
            return
        if not np.all(np.isfinite(np.asarray(values, dtype=float))):
            return
        with self._condition:
            self._force_xyz = values
            self._force_received_at = time.monotonic()
            self._condition.notify_all()

    def latest_force_xyz(
        self, max_age_sec: float | None = None
    ) -> tuple[float, float, float] | None:
        """Return the latest valid wrist force sample without waiting."""
        with self._condition:
            if max_age_sec is not None:
                if max_age_sec <= 0.0:
                    raise ValueError("max_age_sec must be positive")
                if (
                    self._force_received_at is None
                    or time.monotonic() - self._force_received_at > max_age_sec
                ):
                    return None
            return self._force_xyz

    def latest_force_norm(self) -> float | None:
        values = self.latest_force_xyz()
        if values is None:
            return None
        return float(np.linalg.norm(np.asarray(values, dtype=float)))

    def wait_for_force_xyz(
        self,
        timeout_sec: float,
        max_age_sec: float = 0.5,
    ) -> tuple[float, float, float] | None:
        """Wait for a valid force sample satisfying the freshness contract.

        Image and wrench topics are independent best-effort streams.  Waiting
        here closes the harmless race where a new image arrives just before
        the next wrench callback, without ever accepting an old force value.
        """

        if timeout_sec < 0.0:
            raise ValueError("timeout_sec must be non-negative")
        if max_age_sec <= 0.0:
            raise ValueError("max_age_sec must be positive")
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while True:
                if (
                    self._force_xyz is not None
                    and self._force_received_at is not None
                    and time.monotonic() - self._force_received_at
                    <= max_age_sec
                ):
                    return self._force_xyz
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def grab(
        self,
        timeout_sec: float,
        min_cameras: int = 1,
        collection_grace_sec: float = 0.25,
    ) -> CameraSnapshot | None:
        """Collect a fresh multi-camera snapshot.

        Once the minimum camera count arrives, keep collecting briefly so the
        other wrist cameras can join the same decision. Return immediately if
        all configured cameras arrive, or after the grace/overall deadline.
        """
        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        if not 1 <= min_cameras <= len(CAMERA_NAMES):
            raise ValueError("min_cameras is outside the camera count")
        if collection_grace_sec < 0.0:
            raise ValueError("collection_grace_sec must be non-negative")
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            baseline = dict(self._sequences)
            collection_deadline = None
            while True:
                fresh = [
                    camera
                    for camera in CAMERA_NAMES
                    if self._sequences[camera] > baseline[camera]
                ]
                now = time.monotonic()
                if len(fresh) >= min_cameras and collection_deadline is None:
                    collection_deadline = min(
                        deadline, now + collection_grace_sec
                    )
                ready = len(fresh) == len(CAMERA_NAMES) or (
                    len(fresh) >= min_cameras
                    and collection_deadline is not None
                    and now >= collection_deadline
                )
                if ready:
                    frames = {
                        camera: self._latest_frames[camera]
                        for camera in fresh
                        if camera in self._latest_frames
                    }
                    force_xyz = self._force_xyz
                    if (
                        self._force_received_at is None
                        or time.monotonic() - self._force_received_at > 0.5
                    ):
                        force_xyz = None
                    return CameraSnapshot(frames=frames, force_xyz=force_xyz)
                wait_until = deadline
                if collection_deadline is not None:
                    wait_until = min(wait_until, collection_deadline)
                remaining = wait_until - now
                if remaining <= 0.0:
                    if len(fresh) >= min_cameras:
                        frames = {
                            camera: self._latest_frames[camera]
                            for camera in fresh
                            if camera in self._latest_frames
                        }
                        force_xyz = self._force_xyz
                        if (
                            self._force_received_at is None
                            or time.monotonic() - self._force_received_at > 0.5
                        ):
                            force_xyz = None
                        return CameraSnapshot(
                            frames=frames, force_xyz=force_xyz
                        )
                    return None
                self._condition.wait(timeout=remaining)
