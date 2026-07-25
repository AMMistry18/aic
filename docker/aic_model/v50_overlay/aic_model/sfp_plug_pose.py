"""Fail-closed multiview SFP plug-pose estimation.

This module is intentionally independent of ROS.  The controller supplies
camera images, intrinsics, camera poses, and image timestamps; the estimator
runs a *separate* SFP plug YOLO-pose model, triangulates its eight keypoints,
and rigidly fits the simulator's ``sfp_tip_link`` frame.

There is deliberately no fixed-grasp/bias fallback.  A missing model, stale
frames, fewer than two usable cameras, or inconsistent geometry returns no
pose (or raises for malformed configuration) so a caller can stop safely.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .sfp_plug_pose_geometry import (
    SFP_PLUG_LOCAL_KEYPOINTS_M,
    validate_transform,
)


def stamp_to_seconds(stamp: Any) -> float:
    """Convert a ROS-like image stamp or numeric seconds to finite seconds.

    Supported inputs are numeric seconds, objects with ``nanoseconds``, ROS
    ``builtin_interfaces/Time`` objects with ``sec``/``nanosec``, and objects
    whose ``to_msg()`` returns the latter.  The clock domain is not changed;
    callers must compare ROS stamps with ROS time, not ``time.monotonic()``.
    """

    if stamp is None:
        raise ValueError("image timestamp is required")
    if isinstance(stamp, (int, float, np.integer, np.floating)):
        seconds = float(stamp)
    elif hasattr(stamp, "nanoseconds"):
        seconds = float(stamp.nanoseconds) * 1e-9
    elif hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    elif hasattr(stamp, "to_msg"):
        return stamp_to_seconds(stamp.to_msg())
    else:
        raise TypeError(f"unsupported timestamp type: {type(stamp)!r}")
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"invalid image timestamp: {seconds}")
    return seconds


@dataclass(frozen=True)
class PlugPoseView:
    """One synchronized camera observation.

    ``T_world_from_camera`` maps camera-optical coordinates into the controller
    world/base frame.  Existing code that stores ``T_cam_from_base`` must invert
    it before constructing this object.
    """

    camera_name: str
    image_bgr: np.ndarray
    K: np.ndarray
    T_world_from_camera: np.ndarray
    stamp_s: float
    frame_id: str

    def validated(self) -> "PlugPoseView":
        image = np.asarray(self.image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError(f"{self.camera_name}: image_bgr must be nonempty HxWx3")
        intrinsics = np.asarray(self.K, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise ValueError(f"{self.camera_name}: K must be a finite 3x3 matrix")
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise ValueError(f"{self.camera_name}: focal lengths must be positive")
        validate_transform(
            np.asarray(self.T_world_from_camera, dtype=np.float64),
            name=f"{self.camera_name}.T_world_from_camera",
        )
        stamp_to_seconds(self.stamp_s)
        if not self.camera_name:
            raise ValueError("camera_name is required")
        if not self.frame_id:
            raise ValueError(f"{self.camera_name}: frame_id is required")
        return self


@dataclass(frozen=True)
class PlugKeypointDetection:
    camera_name: str
    keypoints_px: np.ndarray
    keypoint_confidences: np.ndarray
    box_confidence: float


@dataclass(frozen=True)
class PlugPoseEstimate:
    """World-frame pose of ``sfp_tip_link`` plus auditable quality fields."""

    position_world: np.ndarray
    rotation_world_from_plug: np.ndarray
    quaternion_wxyz: np.ndarray
    axis_world: np.ndarray
    confidence: float
    stamp_s: float
    age_s: float | None
    view_count: int
    source_frame_ids: tuple[str, ...]
    source_camera_names: tuple[str, ...]
    reprojection_error_px: float
    keypoint_rmse_m: float
    triangulated_keypoint_count: int


@dataclass(frozen=True)
class RelativePlugPoseEstimate:
    """Plug pose expressed in the perceived port frame."""

    translation_port: np.ndarray
    rotation_port_from_plug: np.ndarray
    axis_port: np.ndarray
    confidence: float
    stamp_s: float
    age_s: float
    view_count: int
    source_frame_ids: tuple[str, ...]
    reprojection_error_px: float
    keypoint_rmse_m: float
    world_pose: PlugPoseEstimate


def _projection_matrix(view: PlugPoseView) -> np.ndarray:
    camera_from_world = np.linalg.inv(np.asarray(view.T_world_from_camera, dtype=np.float64))
    return np.asarray(view.K, dtype=np.float64) @ camera_from_world[:3, :]


def triangulate_dlt(
    pixels: Sequence[np.ndarray],
    projection_matrices: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted linear DLT triangulation in the projection matrices' world frame."""

    if len(pixels) != len(projection_matrices) or len(pixels) < 2:
        raise ValueError("triangulation needs matching observations from at least two views")
    if weights is None:
        weights = [1.0] * len(pixels)
    if len(weights) != len(pixels):
        raise ValueError("weights length mismatch")
    rows = []
    for pixel, matrix, weight in zip(pixels, projection_matrices, weights):
        uv = np.asarray(pixel, dtype=np.float64).reshape(2)
        P = np.asarray(matrix, dtype=np.float64)
        if P.shape != (3, 4) or not np.all(np.isfinite(P)):
            raise ValueError("projection matrices must be finite 3x4 arrays")
        scale = math.sqrt(max(float(weight), 1e-6))
        rows.append(scale * (uv[0] * P[2] - P[0]))
        rows.append(scale * (uv[1] * P[2] - P[1]))
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) < 1e-10:
        raise ValueError("degenerate triangulation")
    point = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(point)):
        raise ValueError("non-finite triangulated point")
    return point


def fit_rigid_transform(
    local_points: np.ndarray,
    world_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted Kabsch fit mapping object-local points into world coordinates."""

    source = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 4:
        raise ValueError("rigid fit needs at least four paired 3D points")
    if weights is None:
        w = np.ones(len(source), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(source):
            raise ValueError("rigid-fit weights length mismatch")
        if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError("rigid-fit weights must be finite and positive")
    w /= np.sum(w)
    source_center = np.sum(source * w[:, None], axis=0)
    target_center = np.sum(target * w[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * w[:, None]).T @ target_zero
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    fitted = (rotation @ source.T).T + translation
    residuals = np.linalg.norm(fitted - target, axis=1)
    rmse = float(np.sqrt(np.sum(w * residuals**2)))
    return rotation, translation, rmse


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a canonical unit wxyz quaternion."""

    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * scale,
                (R[2, 1] - R[1, 2]) / scale,
                (R[0, 2] - R[2, 0]) / scale,
                (R[1, 0] - R[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(R)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            q = np.array(
                [
                    (R[2, 1] - R[1, 2]) / scale,
                    0.25 * scale,
                    (R[0, 1] + R[1, 0]) / scale,
                    (R[0, 2] + R[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            q = np.array(
                [
                    (R[0, 2] - R[2, 0]) / scale,
                    (R[0, 1] + R[1, 0]) / scale,
                    0.25 * scale,
                    (R[1, 2] + R[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            q = np.array(
                [
                    (R[1, 0] - R[0, 1]) / scale,
                    (R[0, 2] + R[2, 0]) / scale,
                    (R[1, 2] + R[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("could not convert rotation to quaternion")
    q /= norm
    if q[0] < 0.0:
        q = -q
    return q


def _reprojection_errors(
    rotation_world_from_plug: np.ndarray,
    position_world: np.ndarray,
    local_points: np.ndarray,
    views: Sequence[PlugPoseView],
    detections: Sequence[PlugKeypointDetection],
    min_keypoint_confidence: float,
) -> list[float]:
    world_points = (
        np.asarray(rotation_world_from_plug) @ np.asarray(local_points).T
    ).T + np.asarray(position_world)
    homogeneous = np.column_stack([world_points, np.ones(len(world_points))])
    errors: list[float] = []
    by_camera = {detection.camera_name: detection for detection in detections}
    for view in views:
        detection = by_camera.get(view.camera_name)
        if detection is None:
            continue
        projected = (_projection_matrix(view) @ homogeneous.T).T
        valid_depth = projected[:, 2] > 1e-7
        pixels = projected[:, :2] / np.where(valid_depth, projected[:, 2], 1.0)[:, None]
        valid = valid_depth & (
            np.asarray(detection.keypoint_confidences) >= min_keypoint_confidence
        )
        if np.any(valid):
            delta = pixels[valid] - np.asarray(detection.keypoints_px)[valid]
            errors.extend(np.linalg.norm(delta, axis=1).tolist())
    return errors


def fuse_multiview_keypoints(
    views: Sequence[PlugPoseView],
    detections: Sequence[PlugKeypointDetection],
    *,
    local_keypoints_m: np.ndarray = SFP_PLUG_LOCAL_KEYPOINTS_M,
    min_keypoint_confidence: float = 0.15,
    min_triangulated_keypoints: int = 6,
) -> tuple[np.ndarray, np.ndarray, float, float, int, float]:
    """Fuse matched detections into pose and geometric quality statistics.

    Returns ``(position, rotation, keypoint_rmse_m, reprojection_error_px,
    triangulated_count, mean_keypoint_confidence)``.
    """

    view_by_name = {view.camera_name: view.validated() for view in views}
    if len(view_by_name) < 2:
        raise ValueError("at least two unique camera views are required")
    detection_by_name = {detection.camera_name: detection for detection in detections}
    common_names = sorted(set(view_by_name) & set(detection_by_name))
    if len(common_names) < 2:
        raise ValueError("detections from at least two camera views are required")

    object_points = np.asarray(local_keypoints_m, dtype=np.float64).reshape(-1, 3)
    triangulated: list[np.ndarray] = []
    local_used: list[np.ndarray] = []
    fit_weights: list[float] = []
    for keypoint_index in range(len(object_points)):
        pixels: list[np.ndarray] = []
        matrices: list[np.ndarray] = []
        weights: list[float] = []
        for name in common_names:
            detection = detection_by_name[name]
            keypoints = np.asarray(detection.keypoints_px, dtype=np.float64)
            confidences = np.asarray(detection.keypoint_confidences, dtype=np.float64)
            if keypoints.shape != (len(object_points), 2) or confidences.shape != (
                len(object_points),
            ):
                raise ValueError(f"{name}: unexpected plug keypoint shape")
            confidence = float(confidences[keypoint_index])
            pixel = keypoints[keypoint_index]
            if confidence < min_keypoint_confidence or not np.all(np.isfinite(pixel)):
                continue
            pixels.append(pixel)
            matrices.append(_projection_matrix(view_by_name[name]))
            weights.append(confidence)
        if len(pixels) < 2:
            continue
        triangulated.append(triangulate_dlt(pixels, matrices, weights))
        local_used.append(object_points[keypoint_index])
        fit_weights.append(float(np.mean(weights)))

    if len(triangulated) < int(min_triangulated_keypoints):
        raise ValueError(
            f"only {len(triangulated)} keypoints triangulated; "
            f"need {min_triangulated_keypoints}"
        )
    rotation, position, rmse = fit_rigid_transform(
        np.asarray(local_used), np.asarray(triangulated), np.asarray(fit_weights)
    )
    errors = _reprojection_errors(
        rotation,
        position,
        object_points,
        [view_by_name[name] for name in common_names],
        [detection_by_name[name] for name in common_names],
        min_keypoint_confidence,
    )
    if not errors:
        raise ValueError("no valid reprojections")
    return (
        position,
        rotation,
        rmse,
        float(np.mean(errors)),
        len(triangulated),
        float(np.mean(fit_weights)),
    )


class SfpPlugPoseEstimator:
    """Separate YOLO-pose detector and strict multiview pose fuser.

    The fusing maths is object-agnostic: ``local_keypoints_m`` selects which
    rigid body is being fitted, and defaults to the SFP plug so existing
    callers are unaffected.  ``aic_model.sc_plug_pose.ScPlugPoseEstimator``
    reuses this class with the SC plug's keypoints rather than forking it, so
    both plugs share one audited fail-closed path.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        imgsz: int = 960,
        conf_threshold: float = 0.25,
        min_keypoint_confidence: float = 0.15,
        min_pose_confidence: float = 0.35,
        max_sync_spread_s: float = 0.12,
        max_reprojection_error_px: float = 6.0,
        max_keypoint_rmse_m: float = 0.0035,
        min_views: int = 2,
        device: str | None = None,
        model: Any | None = None,
        local_keypoints_m: np.ndarray | None = None,
    ):
        if model is None:
            if weights_path is None:
                raise ValueError("SFP plug-pose weights path is required")
            path = Path(weights_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"SFP plug-pose weights not found: {path}")
            self._weights_path: Path | None = path
        else:
            self._weights_path = (
                Path(weights_path).expanduser().resolve() if weights_path is not None else None
            )
        self._model = model
        keypoints = np.asarray(
            SFP_PLUG_LOCAL_KEYPOINTS_M if local_keypoints_m is None else local_keypoints_m,
            dtype=np.float64,
        )
        if keypoints.ndim != 2 or keypoints.shape[1] != 3 or len(keypoints) < 4:
            raise ValueError("local_keypoints_m must be at least four Nx3 points")
        if not np.all(np.isfinite(keypoints)):
            raise ValueError("local_keypoints_m must be finite")
        self.local_keypoints_m = keypoints
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self.min_pose_confidence = float(min_pose_confidence)
        self.max_sync_spread_s = float(max_sync_spread_s)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.max_keypoint_rmse_m = float(max_keypoint_rmse_m)
        self.min_views = max(2, int(min_views))
        self.device = device

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self._weights_path))
        return self._model

    def _predict(self, views: Sequence[PlugPoseView]) -> list[PlugKeypointDetection]:
        model = self._load_model()
        kwargs: dict[str, Any] = {
            "verbose": False,
            "conf": self.conf_threshold,
            "imgsz": self.imgsz,
        }
        if self.device:
            kwargs["device"] = self.device
        results = model([view.image_bgr for view in views], **kwargs)
        if len(results) != len(views):
            return []

        detections: list[PlugKeypointDetection] = []
        expected_count = len(self.local_keypoints_m)
        for view, result in zip(views, results):
            if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
                continue
            boxes_conf = np.asarray(result.boxes.conf.cpu().numpy(), dtype=np.float64)
            keypoints = np.asarray(result.keypoints.xy.cpu().numpy(), dtype=np.float64)
            raw_kp_conf = getattr(result.keypoints, "conf", None)
            if raw_kp_conf is None:
                kp_conf = np.repeat(boxes_conf[:, None], expected_count, axis=1)
            else:
                kp_conf = np.asarray(raw_kp_conf.cpu().numpy(), dtype=np.float64)
            candidates = []
            for index in range(min(len(boxes_conf), len(keypoints), len(kp_conf))):
                if keypoints[index].shape != (expected_count, 2):
                    continue
                score = float(boxes_conf[index]) * math.sqrt(
                    max(float(np.mean(kp_conf[index])), 0.0)
                )
                candidates.append((score, index))
            if not candidates:
                continue
            _, best = max(candidates)
            detections.append(
                PlugKeypointDetection(
                    camera_name=view.camera_name,
                    keypoints_px=keypoints[best].copy(),
                    keypoint_confidences=kp_conf[best].copy(),
                    box_confidence=float(boxes_conf[best]),
                )
            )
        return detections

    def detect_views(
        self, views: Sequence[PlugPoseView]
    ) -> list[PlugKeypointDetection]:
        """Run the plug model once and return the best detection per camera."""

        validated = [view.validated() for view in views]
        return self._predict(validated)

    def estimate_multiview(
        self,
        views: Sequence[PlugPoseView],
        *,
        now_s: float | None = None,
        max_age_s: float | None = None,
        min_stamp_s: float | None = None,
        detections: Sequence[PlugKeypointDetection] | None = None,
    ) -> PlugPoseEstimate | None:
        """Estimate ``sfp_tip_link`` in world coordinates or fail closed.

        ``now_s`` and ``min_stamp_s`` must share the image stamps' clock
        domain (normally ROS simulation time).  Runtime callers should provide
        both ``now_s`` and ``max_age_s``; omitting them is intended only for
        offline held-out evaluation.
        """

        try:
            unique: dict[str, PlugPoseView] = {}
            for view in views:
                validated = view.validated()
                if validated.camera_name in unique:
                    return None
                unique[validated.camera_name] = validated
            selected = list(unique.values())
            if len(selected) < self.min_views:
                return None
            stamps = np.array([stamp_to_seconds(view.stamp_s) for view in selected])
            if float(np.ptp(stamps)) > self.max_sync_spread_s:
                return None
            estimate_stamp = float(np.max(stamps))
            if min_stamp_s is not None and estimate_stamp <= float(min_stamp_s):
                return None
            age: float | None = None
            if now_s is not None:
                now = stamp_to_seconds(now_s)
                # Camera headers can be epoch-stamped while the policy node's
                # simulation clock is elapsed time.  A negative cross-domain
                # age is not evidence that the image is stale, so clamp it to
                # zero instead of rejecting it as clock skew.  Positive ages
                # still pass through the normal stale-frame guard below.
                age = max(0.0, now - estimate_stamp)
                if max_age_s is not None and age > float(max_age_s):
                    return None
            elif max_age_s is not None:
                raise ValueError("now_s is required when max_age_s is provided")

            detected = list(detections) if detections is not None else self._predict(selected)
            detected_names = {detection.camera_name for detection in detected}
            used_views = [view for view in selected if view.camera_name in detected_names]
            if len(used_views) < self.min_views:
                return None
            position, rotation, rmse, reprojection, count, kp_conf = fuse_multiview_keypoints(
                used_views,
                detected,
                local_keypoints_m=self.local_keypoints_m,
                min_keypoint_confidence=self.min_keypoint_confidence,
            )
            if reprojection > self.max_reprojection_error_px:
                return None
            if rmse > self.max_keypoint_rmse_m:
                return None
            detection_conf = float(
                np.mean([d.box_confidence for d in detected if d.camera_name in detected_names])
            )
            reprojection_quality = math.exp(-reprojection / max(self.max_reprojection_error_px, 1e-6))
            shape_quality = math.exp(-rmse / max(self.max_keypoint_rmse_m, 1e-9))
            view_quality = 0.85 if len(used_views) == 2 else 1.0
            confidence = float(
                detection_conf
                * math.sqrt(max(kp_conf, 0.0))
                * math.sqrt(reprojection_quality)
                * math.sqrt(shape_quality)
                * view_quality
            )
            if confidence < self.min_pose_confidence:
                return None
            source_ids = tuple(view.frame_id for view in used_views)
            if len(set(source_ids)) != len(source_ids):
                return None
            quaternion = rotation_matrix_to_quaternion_wxyz(rotation)
            return PlugPoseEstimate(
                position_world=position,
                rotation_world_from_plug=rotation,
                quaternion_wxyz=quaternion,
                axis_world=rotation[:, 2].copy(),
                confidence=confidence,
                stamp_s=estimate_stamp,
                age_s=age,
                view_count=len(used_views),
                source_frame_ids=source_ids,
                source_camera_names=tuple(view.camera_name for view in used_views),
                reprojection_error_px=reprojection,
                keypoint_rmse_m=rmse,
                triangulated_keypoint_count=count,
            )
        except (
            ArithmeticError,
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            return None

    def estimate_relative_to_port(
        self,
        views: Sequence[PlugPoseView],
        port_position_world: np.ndarray,
        port_rotation_world: np.ndarray,
        *,
        now_s: float,
        max_age_s: float = 0.35,
        min_stamp_s: float | None = None,
    ) -> RelativePlugPoseEstimate | None:
        """Return a fresh plug pose in port coordinates, with no fallback."""

        world_pose = self.estimate_multiview(
            views,
            now_s=now_s,
            max_age_s=max_age_s,
            min_stamp_s=min_stamp_s,
        )
        if world_pose is None or world_pose.age_s is None:
            return None
        try:
            port_position = np.asarray(port_position_world, dtype=np.float64).reshape(3)
            port_rotation = np.asarray(port_rotation_world, dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(port_position)):
                return None
            port_transform = np.eye(4, dtype=np.float64)
            port_transform[:3, :3] = port_rotation
            port_transform[:3, 3] = port_position
            validate_transform(port_transform, name="world_from_port")
            rotation_port_from_plug = port_rotation.T @ world_pose.rotation_world_from_plug
            translation_port = port_rotation.T @ (world_pose.position_world - port_position)
            return RelativePlugPoseEstimate(
                translation_port=translation_port,
                rotation_port_from_plug=rotation_port_from_plug,
                axis_port=rotation_port_from_plug[:, 2].copy(),
                confidence=world_pose.confidence,
                stamp_s=world_pose.stamp_s,
                age_s=world_pose.age_s,
                view_count=world_pose.view_count,
                source_frame_ids=world_pose.source_frame_ids,
                reprojection_error_px=world_pose.reprojection_error_px,
                keypoint_rmse_m=world_pose.keypoint_rmse_m,
                world_pose=world_pose,
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None


__all__ = [
    "PlugKeypointDetection",
    "PlugPoseEstimate",
    "PlugPoseView",
    "RelativePlugPoseEstimate",
    "SfpPlugPoseEstimator",
    "fit_rigid_transform",
    "fuse_multiview_keypoints",
    "rotation_matrix_to_quaternion_wxyz",
    "stamp_to_seconds",
    "triangulate_dlt",
]
