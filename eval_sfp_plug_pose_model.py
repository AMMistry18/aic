#!/usr/bin/env python3
"""Held-out geometric evaluation for the separate SFP plug-pose model."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Allow the checked-out script to run before the ROS package is rebuilt.
_REPO_ROOT = Path(__file__).resolve().parent
_AIC_MODEL_SOURCE = _REPO_ROOT / "aic_model"
if str(_AIC_MODEL_SOURCE) not in sys.path:
    sys.path.insert(0, str(_AIC_MODEL_SOURCE))

from aic_model.sfp_plug_pose import PlugPoseView, SfpPlugPoseEstimator


def _parse_args():
    dataset_root = Path.home() / "aic_perception_data" / "sfp_plug_pose"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=dataset_root / "aic_sfp_plug_pose.yaml",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=dataset_root / "reports" / "eval_sfp_plug_pose_test.json",
    )
    parser.add_argument("--max-miss-rate", type=float, default=0.01)
    parser.add_argument("--max-p95-position-mm", type=float, default=2.0)
    parser.add_argument("--max-p95-axis-deg", type=float, default=3.0)
    parser.add_argument("--max-p95-keypoint-px", type=float, default=4.0)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="exit nonzero when held-out gates fail",
    )
    return parser.parse_args()


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


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
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
        np.clip(
            np.dot(np.asarray(predicted)[:, 2], np.asarray(truth)[:, 2]),
            -1.0,
            1.0,
        )
    )
    return math.degrees(math.acos(cosine))


def _stamp_seconds(stamp: dict) -> float:
    return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9


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


def main():
    args = _parse_args()
    weights = args.weights.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    dataset_root = data_yaml.parent
    report_path = args.report.expanduser().resolve()
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
    estimator = SfpPlugPoseEstimator(
        weights,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        min_pose_confidence=0.0,
        device=device,
        model=model,
    )
    validation = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        verbose=False,
    )
    yolo_metrics = {}
    if hasattr(validation, "results_dict"):
        yolo_metrics = {
            key: float(value) for key, value in validation.results_dict.items()
        }

    position_errors_mm: list[float] = []
    rotation_errors_deg: list[float] = []
    axis_errors_deg: list[float] = []
    keypoint_errors_px: list[float] = []
    reprojection_errors_px: list[float] = []
    pose_confidences: list[float] = []
    inference_seconds: list[float] = []
    missed_groups = 0
    missed_camera_detections = 0
    camera_images = 0
    evaluated_groups = []

    for metadata_path, metadata in groups:
        views = []
        camera_meta_by_name = {}
        for camera_name, camera_meta in metadata.get("cameras", {}).items():
            image_path = dataset_root / camera_meta["image"]
            image = cv2.imread(str(image_path))
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
                    frame_id=camera_meta["frame_id"],
                )
            )
        if len(views) < 2:
            missed_groups += 1
            continue
        started = time.monotonic()
        detections = estimator.detect_views(views)
        inference_seconds.append(time.monotonic() - started)
        detections_by_name = {detection.camera_name: detection for detection in detections}
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
            evaluated_groups.append(
                {"metadata": str(metadata_path), "pose_estimated": False}
            )
            continue
        reference = camera_meta_by_name[estimate.source_camera_names[0]]
        truth_transform = np.asarray(reference["T_world_from_tip"], dtype=np.float64)
        position_error = float(
            np.linalg.norm(estimate.position_world - truth_transform[:3, 3]) * 1000.0
        )
        rotation_error = _rotation_error_deg(
            estimate.rotation_world_from_plug, truth_transform[:3, :3]
        )
        axis_error = _axis_error_deg(
            estimate.rotation_world_from_plug, truth_transform[:3, :3]
        )
        position_errors_mm.append(position_error)
        rotation_errors_deg.append(rotation_error)
        axis_errors_deg.append(axis_error)
        reprojection_errors_px.append(estimate.reprojection_error_px)
        pose_confidences.append(estimate.confidence)
        evaluated_groups.append(
            {
                "metadata": str(metadata_path),
                "pose_estimated": True,
                "position_error_mm": position_error,
                "rotation_error_deg": rotation_error,
                "axis_error_deg": axis_error,
                "confidence": estimate.confidence,
                "reprojection_error_px": estimate.reprojection_error_px,
                "keypoint_rmse_m": estimate.keypoint_rmse_m,
                "views": estimate.view_count,
            }
        )

    group_miss_rate = missed_groups / len(groups)
    camera_miss_rate = (
        missed_camera_detections / camera_images if camera_images else 1.0
    )
    position_summary = _summary(position_errors_mm)
    axis_summary = _summary(axis_errors_deg)
    keypoint_summary = _summary(keypoint_errors_px)
    gates = {
        "group_miss_rate": {
            "value": group_miss_rate,
            "limit": args.max_miss_rate,
            "pass": group_miss_rate <= args.max_miss_rate,
        },
        "position_p95_mm": {
            "value": position_summary.get("p95"),
            "limit": args.max_p95_position_mm,
            "pass": bool(position_summary)
            and position_summary["p95"] <= args.max_p95_position_mm,
        },
        "axis_p95_deg": {
            "value": axis_summary.get("p95"),
            "limit": args.max_p95_axis_deg,
            "pass": bool(axis_summary)
            and axis_summary["p95"] <= args.max_p95_axis_deg,
        },
        "keypoint_p95_px": {
            "value": keypoint_summary.get("p95"),
            "limit": args.max_p95_keypoint_px,
            "pass": bool(keypoint_summary)
            and keypoint_summary["p95"] <= args.max_p95_keypoint_px,
        },
    }
    report = {
        "weights": str(weights),
        "data": str(data_yaml),
        "split": args.split,
        "device": device,
        "groups_total": len(groups),
        "groups_with_pose": len(position_errors_mm),
        "group_miss_rate": group_miss_rate,
        "camera_images": camera_images,
        "camera_detection_miss_rate": camera_miss_rate,
        "position_error_mm": position_summary,
        "rotation_error_deg": _summary(rotation_errors_deg),
        "axis_error_deg": axis_summary,
        "keypoint_error_px": keypoint_summary,
        "fusion_reprojection_error_px": _summary(reprojection_errors_px),
        "pose_confidence": _summary(pose_confidences),
        "inference_seconds_per_three_view_group": _summary(inference_seconds),
        "inference_seconds_total": sum(inference_seconds),
        "yolo_metrics": yolo_metrics,
        "gates": gates,
        "all_gates_pass": all(gate["pass"] for gate in gates.values()),
        "per_group": evaluated_groups,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_group"}, indent=2))
    print(f"Full held-out report: {report_path}")
    if args.enforce and not report["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
