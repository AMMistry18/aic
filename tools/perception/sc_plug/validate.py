#!/usr/bin/env python3
"""Validate SC plug tip-pose estimation against simulator ground truth.

Two modes:

``--mode dataset`` (the real validation)
    Runs the trained SC plug-pose model over a held-out split of the
    ``DataCollectorScPlugPoseGT`` dataset, fuses the multiview detections with
    :class:`ScPlugPoseEstimator`, and compares the fitted ``sc_tip_link`` pose
    against the simulator TF ground truth recorded alongside each frame.
    Requires a collected dataset and trained weights.

``--mode synthetic`` (error budget, runs with no data)
    Rebuilds the real three-camera wrist rig from the robot description,
    projects the SC plug keypoints, and measures how keypoint pixel noise
    propagates into tip-position error.  This answers "how accurate must the
    network's keypoints be to land inside the port clearance?" before any data
    exists, and it exercises the same label-formatting path the collector uses.

The SC port opening leaves 0.725 mm of vertical clearance per side (vertical is
the binding axis) and 1.205 mm lateral, so the working target for the estimated
tip is roughly 0.4 mm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

# Allow the checked-out script to run before the ROS package is rebuilt.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AIC_MODEL_SOURCE = _REPO_ROOT / "aic_model"
if str(_AIC_MODEL_SOURCE) not in sys.path:
    sys.path.insert(0, str(_AIC_MODEL_SOURCE))

from aic_model.sc_plug_pose import ScPlugPoseEstimator  # noqa: E402
from aic_model.sc_plug_pose_geometry import (  # noqa: E402
    SC_PLUG_LOCAL_KEYPOINTS_M,
    format_yolo_pose_label,
    padded_bbox,
    project_keypoints,
    visibility_flags,
)
from aic_model.sfp_plug_pose import (  # noqa: E402
    PlugKeypointDetection,
    PlugPoseView,
    fuse_multiview_keypoints,
)


# --------------------------------------------------------------------------
# Wrist camera rig, transcribed from the robot description.
#
# aic_description/urdf/ur_gz.urdf.xacro attaches three Basler cameras to
# cam_mount_link, and aic_assets/models/Basler Camera/basler_camera_macro.xacro
# carries camera_link -> sensor_link -> optical.  Image size and horizontal FOV
# come from the same macro's <camera> block.
# --------------------------------------------------------------------------
IMAGE_WIDTH = 1152
IMAGE_HEIGHT = 1024
HORIZONTAL_FOV_RAD = 0.8718

# cam_mount_link -> tcp, summed along the wrist chain:
#   cam_mount_link -> ati base_link      +0.0265  (ur_gz.urdf.xacro)
#   ati base_link  -> ati tool_link      +0.0245  (axia80_m20_macro.xacro adapter_offset)
#   ati tool_link  -> hande_base_link     0.0     (ur_gz.urdf.xacro attach_gripper)
#   hande_base_link-> gripper tcp        +0.172   (robotiq_hande_macro.xacro)
MOUNT_TO_TCP_Z_M = 0.0265 + 0.0245 + 0.172

# cam_mount_link -> camera_link, as (xyz, rpy).
CAMERA_MOUNTS = {
    "left_camera": ((-0.09326, -0.053843, -0.007188), (0.0, -1.30899630, 0.523599027)),
    "center_camera": ((0.0, -0.1077, -0.00719), (0.0, -1.30899630, 1.57079623)),
    "right_camera": ((0.09326, -0.053843, -0.007188), (0.0, -1.30899630, 2.61799343)),
}
# camera_link -> sensor_link, then sensor_link -> optical (z forward).
SENSOR_IN_CAMERA_XYZ = (0.02174, 0.0, 0.0145)
OPTICAL_IN_SENSOR_RPY = (-1.5708, 0.0, -1.5708)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw to a rotation matrix (Rz @ Ry @ Rx)."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _transform(xyz, rpy) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rpy_to_matrix(*rpy)
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def camera_intrinsics() -> np.ndarray:
    """Pinhole K for the Basler wrist cameras."""

    fx = (IMAGE_WIDTH * 0.5) / math.tan(HORIZONTAL_FOV_RAD * 0.5)
    return np.array(
        [[fx, 0.0, IMAGE_WIDTH * 0.5], [0.0, fx, IMAGE_HEIGHT * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rig_optical_poses() -> dict[str, np.ndarray]:
    """Return T_mount_from_optical for each wrist camera."""

    sensor_in_camera = _transform(SENSOR_IN_CAMERA_XYZ, (0.0, 0.0, 0.0))
    optical_in_sensor = _transform((0.0, 0.0, 0.0), OPTICAL_IN_SENSOR_RPY)
    poses = {}
    for name, (xyz, rpy) in CAMERA_MOUNTS.items():
        poses[name] = _transform(xyz, rpy) @ sensor_in_camera @ optical_in_sensor
    return poses


def rig_convergence_point(poses: dict[str, np.ndarray]) -> np.ndarray:
    """Least-squares intersection of the three optical axes.

    The cameras are toed in toward the gripper's working volume, so where their
    axes converge is the distance the rig was designed to view the held plug
    from.  Using it keeps the synthetic study at a realistic scale instead of
    an invented one.
    """

    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for pose in poses.values():
        origin = pose[:3, 3]
        direction = pose[:3, 2] / np.linalg.norm(pose[:3, 2])
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ origin
    return np.linalg.solve(a, b)


def nominal_tip_pose_in_mount() -> np.ndarray:
    """Nominal held-plug pose ``T_mount_from_tip`` for the synthetic study.

    The grasp transform used here is the SFP one, because the SC transform is
    exactly what is not calibrated -- that is the problem this work removes.
    For an error budget only the viewing geometry matters (about 280 mm from
    each camera, all eight keypoints inside the frame), and the SFP grasp puts
    the plug in the right place to a few millimetres, which changes the
    resulting error scale by well under the precision of this study.
    """

    from aic_model.rl_insert_contract import (  # noqa: PLC0415
        SFP_TIP_IN_TCP_POS,
        SFP_TIP_IN_TCP_QUAT,
    )

    w, x, y, z = np.asarray(SFP_TIP_IN_TCP_QUAT, dtype=np.float64)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    tip_in_tcp = np.eye(4, dtype=np.float64)
    tip_in_tcp[:3, :3] = rotation
    tip_in_tcp[:3, 3] = np.asarray(SFP_TIP_IN_TCP_POS, dtype=np.float64)
    tcp_in_mount = np.eye(4, dtype=np.float64)
    tcp_in_mount[2, 3] = MOUNT_TO_TCP_Z_M
    return tcp_in_mount @ tip_in_tcp


def _summary(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _rotation_error_deg(predicted: np.ndarray, truth: np.ndarray) -> float:
    delta = np.asarray(truth).T @ np.asarray(predicted)
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _axis_error_deg(predicted: np.ndarray, truth: np.ndarray) -> float:
    cosine = float(
        np.clip(np.dot(np.asarray(predicted)[:, 2], np.asarray(truth)[:, 2]), -1.0, 1.0)
    )
    return math.degrees(math.acos(cosine))


def _stamp_seconds(stamp: dict) -> float:
    return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9


def _split_error_mm(delta_m: np.ndarray, truth_rotation: np.ndarray) -> tuple[float, float]:
    """Split a tip-position error into lateral and axial millimetres.

    Local +Z of ``sc_tip_link`` is the insertion axis, so the component along
    it is depth-into-the-bore error while the perpendicular component is what
    has to fit inside the port opening's clearance.  Decomposing in the plug's
    own frame keeps the number meaningful regardless of how the wrist is
    oriented at the moment of measurement.
    """

    delta = np.asarray(delta_m, dtype=np.float64).reshape(3)
    axis = np.asarray(truth_rotation, dtype=np.float64)[:, 2]
    axis = axis / np.linalg.norm(axis)
    axial = float(np.dot(delta, axis))
    lateral = float(np.linalg.norm(delta - axial * axis))
    return lateral * 1000.0, abs(axial) * 1000.0


# --------------------------------------------------------------------------
# Synthetic error budget
# --------------------------------------------------------------------------
def run_synthetic(args) -> dict:
    rng = np.random.default_rng(args.seed)
    K = camera_intrinsics()
    poses = rig_optical_poses()
    nominal = nominal_tip_pose_in_mount()
    focus = nominal[:3, 3]
    convergence = rig_convergence_point(poses)
    view_distances = {
        name: float(np.linalg.norm(focus - pose[:3, 3])) for name, pose in poses.items()
    }
    baselines = {}
    names = sorted(poses)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            baselines[f"{first}|{second}"] = float(
                np.linalg.norm(poses[first][:3, 3] - poses[second][:3, 3])
            )

    label_checks = {"frames": 0, "valid_labels": 0, "min_visible_keypoints": 8}
    results_by_sigma: dict[str, dict] = {}

    for sigma in args.pixel_sigmas:
        position_errors_mm: list[float] = []
        lateral_errors_mm: list[float] = []
        axial_errors_mm: list[float] = []
        axis_errors_deg: list[float] = []
        rotation_errors_deg: list[float] = []
        rejected = 0

        for _ in range(args.trials):
            # A plausible held-plug pose: near the rig's convergence point with
            # a few millimetres of grasp slop and a few degrees of tilt.
            truth = np.eye(4, dtype=np.float64)
            truth[:3, :3] = nominal[:3, :3] @ rpy_to_matrix(
                *(rng.uniform(-args.tilt_rad, args.tilt_rad, size=3))
            )
            truth[:3, 3] = focus + rng.uniform(-args.jitter_m, args.jitter_m, size=3)

            views: list[PlugPoseView] = []
            detections: list[PlugKeypointDetection] = []
            for name, optical in poses.items():
                camera_from_tip = np.linalg.inv(optical) @ truth
                pixels, in_front = project_keypoints(
                    SC_PLUG_LOCAL_KEYPOINTS_M, camera_from_tip, K
                )
                flags = visibility_flags(pixels, in_front, IMAGE_WIDTH, IMAGE_HEIGHT)
                visible = int(np.count_nonzero(flags == 2))
                label_checks["min_visible_keypoints"] = min(
                    label_checks["min_visible_keypoints"], visible
                )
                bbox = padded_bbox(
                    pixels, in_front, IMAGE_WIDTH, IMAGE_HEIGHT, padding=0.28
                )
                label_checks["frames"] += 1
                if bbox is not None and visible >= 6:
                    label = format_yolo_pose_label(
                        bbox, pixels, flags, IMAGE_WIDTH, IMAGE_HEIGHT
                    )
                    if len(label.split()) == 29:
                        label_checks["valid_labels"] += 1

                noisy = pixels + rng.normal(0.0, sigma, size=pixels.shape)
                views.append(
                    PlugPoseView(
                        camera_name=name,
                        image_bgr=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
                        K=K,
                        T_world_from_camera=optical,
                        stamp_s=1.0,
                        frame_id=f"{name}/optical",
                    )
                )
                detections.append(
                    PlugKeypointDetection(
                        camera_name=name,
                        keypoints_px=noisy,
                        keypoint_confidences=np.ones(len(noisy)),
                        box_confidence=1.0,
                    )
                )

            try:
                position, rotation, _, _, _, _ = fuse_multiview_keypoints(
                    views, detections, local_keypoints_m=SC_PLUG_LOCAL_KEYPOINTS_M
                )
            except (ValueError, np.linalg.LinAlgError):
                rejected += 1
                continue
            delta = position - truth[:3, 3]
            lateral, axial = _split_error_mm(delta, truth[:3, :3])
            position_errors_mm.append(float(np.linalg.norm(delta) * 1000.0))
            lateral_errors_mm.append(lateral)
            axial_errors_mm.append(axial)
            axis_errors_deg.append(_axis_error_deg(rotation, truth[:3, :3]))
            rotation_errors_deg.append(_rotation_error_deg(rotation, truth[:3, :3]))

        results_by_sigma[f"{sigma:g}"] = {
            "keypoint_sigma_px": sigma,
            "trials": args.trials,
            "rejected": rejected,
            "position_error_mm": _summary(position_errors_mm),
            "lateral_error_mm": _summary(lateral_errors_mm),
            "axial_error_mm": _summary(axial_errors_mm),
            "axis_error_deg": _summary(axis_errors_deg),
            "rotation_error_deg": _summary(rotation_errors_deg),
        }

    # Largest sigma whose median tip error still fits the working target.
    affordable = [
        entry["keypoint_sigma_px"]
        for entry in results_by_sigma.values()
        if entry["position_error_mm"].get("median", math.inf) <= args.target_mm
    ]
    return {
        "mode": "synthetic",
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "focal_length_px": float(K[0, 0]),
        "nominal_tip_position_mount_frame_m": focus.tolist(),
        "rig_convergence_point_mount_frame_m": convergence.tolist(),
        "camera_to_plug_distance_m": view_distances,
        "stereo_baselines_m": baselines,
        "label_path_check": label_checks,
        "target_position_mm": args.target_mm,
        "vertical_clearance_mm": args.vertical_clearance_mm,
        "by_keypoint_sigma": results_by_sigma,
        "max_keypoint_sigma_px_within_target": max(affordable) if affordable else None,
    }


# --------------------------------------------------------------------------
# Held-out dataset validation against simulator TF ground truth
# --------------------------------------------------------------------------
def _choose_device(requested: str) -> str:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    if requested != "auto":
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def _load_groups(dataset_root: Path, split: str, limit: int | None):
    groups = []
    for metadata_path in sorted((dataset_root / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("split") != split:
            continue
        groups.append((metadata_path, metadata))
        if limit is not None and len(groups) >= limit:
            break
    return groups


def run_dataset(args) -> dict:
    import cv2

    weights = args.weights.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    dataset_root = data_yaml.parent
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")
    groups = _load_groups(dataset_root, args.split, args.limit)
    if not groups:
        raise RuntimeError(f"no {args.split!r} metadata groups under {dataset_root}")

    device = _choose_device(args.device)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # min_pose_confidence is relaxed to 0 so that the report measures the
    # geometry the model actually produces.  The deployed estimator keeps its
    # fail-closed gates; this run is for measuring, not for driving.
    estimator = ScPlugPoseEstimator(
        weights,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        min_pose_confidence=0.0,
        device=device,
        model=model,
    )

    position_errors_mm: list[float] = []
    lateral_errors_mm: list[float] = []
    axial_errors_mm: list[float] = []
    rotation_errors_deg: list[float] = []
    axis_errors_deg: list[float] = []
    keypoint_errors_px: list[float] = []
    reprojection_errors_px: list[float] = []
    inference_seconds: list[float] = []
    missed_groups = 0
    camera_images = 0
    missed_camera_detections = 0
    per_group = []

    for metadata_path, metadata in groups:
        views = []
        camera_meta_by_name = {}
        for camera_name, camera_meta in metadata.get("cameras", {}).items():
            image = cv2.imread(str(dataset_root / camera_meta["image"]))
            if image is None:
                continue
            camera_images += 1
            camera_meta_by_name[camera_name] = camera_meta
            views.append(
                PlugPoseView(
                    camera_name=camera_name,
                    image_bgr=image,
                    K=np.asarray(camera_meta["K"], dtype=np.float64),
                    T_world_from_camera=np.asarray(
                        camera_meta["T_world_from_camera"], dtype=np.float64
                    ),
                    stamp_s=_stamp_seconds(camera_meta["stamp"]),
                    frame_id=camera_meta["camera_frame"],
                )
            )
        if len(views) < 2:
            missed_groups += 1
            continue

        started = time.monotonic()
        detections = estimator.detect_views(views)
        inference_seconds.append(time.monotonic() - started)
        detections_by_name = {d.camera_name: d for d in detections}
        missed_camera_detections += len(views) - len(detections_by_name)
        for camera_name, detection in detections_by_name.items():
            truth_px = np.asarray(
                camera_meta_by_name[camera_name]["keypoints_px"], dtype=np.float64
            )
            confidence = np.asarray(detection.keypoint_confidences, dtype=np.float64)
            valid = confidence >= estimator.min_keypoint_confidence
            if np.any(valid):
                delta = np.asarray(detection.keypoints_px)[valid] - truth_px[valid]
                keypoint_errors_px.extend(np.linalg.norm(delta, axis=1).tolist())

        estimate = estimator.estimate_multiview(views, detections=detections)
        if estimate is None:
            missed_groups += 1
            per_group.append({"metadata": str(metadata_path), "pose_estimated": False})
            continue

        reference = camera_meta_by_name[estimate.source_camera_names[0]]
        truth_transform = np.asarray(reference["T_world_from_tip"], dtype=np.float64)
        delta = estimate.position_world - truth_transform[:3, 3]
        position_error = float(np.linalg.norm(delta) * 1000.0)
        lateral_error, axial_error = _split_error_mm(delta, truth_transform[:3, :3])
        rotation_error = _rotation_error_deg(
            estimate.rotation_world_from_plug, truth_transform[:3, :3]
        )
        axis_error = _axis_error_deg(
            estimate.rotation_world_from_plug, truth_transform[:3, :3]
        )
        position_errors_mm.append(position_error)
        lateral_errors_mm.append(lateral_error)
        axial_errors_mm.append(axial_error)
        rotation_errors_deg.append(rotation_error)
        axis_errors_deg.append(axis_error)
        reprojection_errors_px.append(estimate.reprojection_error_px)
        per_group.append(
            {
                "metadata": str(metadata_path),
                "pose_estimated": True,
                "position_error_mm": position_error,
                "lateral_error_mm": lateral_error,
                "axial_error_mm": axial_error,
                "rotation_error_deg": rotation_error,
                "axis_error_deg": axis_error,
                "confidence": estimate.confidence,
                "reprojection_error_px": estimate.reprojection_error_px,
                "keypoint_rmse_m": estimate.keypoint_rmse_m,
                "views": estimate.view_count,
            }
        )

    group_miss_rate = missed_groups / len(groups)
    position_summary = _summary(position_errors_mm)
    lateral_summary = _summary(lateral_errors_mm)
    axis_summary = _summary(axis_errors_deg)
    gates = {
        "group_miss_rate": {
            "value": group_miss_rate,
            "limit": args.max_miss_rate,
            "pass": group_miss_rate <= args.max_miss_rate,
        },
        "position_median_mm": {
            "value": position_summary.get("median"),
            "limit": args.target_mm,
            "pass": bool(position_summary)
            and position_summary["median"] <= args.target_mm,
        },
        "lateral_p95_mm": {
            "value": lateral_summary.get("p95"),
            "limit": args.vertical_clearance_mm,
            "pass": bool(lateral_summary)
            and lateral_summary["p95"] <= args.vertical_clearance_mm,
        },
        "axis_p95_deg": {
            "value": axis_summary.get("p95"),
            "limit": args.max_p95_axis_deg,
            "pass": bool(axis_summary) and axis_summary["p95"] <= args.max_p95_axis_deg,
        },
    }
    return {
        "mode": "dataset",
        "weights": str(weights),
        "data": str(data_yaml),
        "split": args.split,
        "device": device,
        "groups_total": len(groups),
        "groups_with_pose": len(position_errors_mm),
        "group_miss_rate": group_miss_rate,
        "camera_images": camera_images,
        "camera_detection_miss_rate": (
            missed_camera_detections / camera_images if camera_images else 1.0
        ),
        "position_error_mm": position_summary,
        "lateral_error_mm": lateral_summary,
        "axial_error_mm": _summary(axial_errors_mm),
        "rotation_error_deg": _summary(rotation_errors_deg),
        "axis_error_deg": axis_summary,
        "keypoint_error_px": _summary(keypoint_errors_px),
        "fusion_reprojection_error_px": _summary(reprojection_errors_px),
        "inference_seconds_per_group": _summary(inference_seconds),
        "gates": gates,
        "all_gates_pass": all(gate["pass"] for gate in gates.values()),
        "per_group": per_group,
    }


def _parse_args():
    dataset_root = Path.home() / "aic_perception_data" / "sc_plug_pose"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dataset", "synthetic"), default="dataset")
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--data", type=Path, default=dataset_root / "aic_sc_plug_pose.yaml"
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--target-mm", type=float, default=0.4)
    parser.add_argument("--vertical-clearance-mm", type=float, default=0.725)
    parser.add_argument("--max-miss-rate", type=float, default=0.01)
    parser.add_argument("--max-p95-axis-deg", type=float, default=3.0)
    # synthetic-only
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--jitter-m", type=float, default=0.010)
    parser.add_argument("--tilt-rad", type=float, default=0.12)
    parser.add_argument(
        "--pixel-sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if args.mode == "dataset" and args.weights is None:
        parser.error("--weights is required for --mode dataset")
    if args.report is None:
        args.report = dataset_root / "reports" / f"validate_sc_plug_pose_{args.mode}.json"
    return args


def main():
    args = _parse_args()
    report = run_synthetic(args) if args.mode == "synthetic" else run_dataset(args)
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = {key: value for key, value in report.items() if key != "per_group"}
    print(json.dumps(printable, indent=2))
    print(f"Full report: {report_path}")
    if args.enforce and not report.get("all_gates_pass", True):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
