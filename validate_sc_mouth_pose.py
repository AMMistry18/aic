#!/usr/bin/env python3
"""Evaluate a physical SC-mouth pose checkpoint against held-out sim TF data.

Unlike the legacy SC port evaluator, this uses the collector's camera-matrix
and entrance-frame metadata to report both 2-D mouth-centre error and
single-view PnP translation/rotation error.  It is an evaluation gate for a
candidate checkpoint, not permission to replace the deployed legacy weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

_SOURCE_ROOT = Path(__file__).resolve().parent / "aic_example_policies"
if _SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(_SOURCE_ROOT))

from aic_example_policies.ros.sc_mouth_pose_geometry import LOCAL_SC_FRONT_MOUTH_KPS_M


EXPECTED_KEYPOINTS = 5
EXPECTED_TOKENS = 5 + 3 * EXPECTED_KEYPOINTS


def _summary(values: list[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _parse_labels(path: Path, width: int, height: int) -> list[dict]:
    labels: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) != EXPECTED_TOKENS:
            raise ValueError(f"{path} has a {len(tokens)}-token label, expected {EXPECTED_TOKENS}")
        values = np.asarray([float(value) for value in tokens[1:]], dtype=np.float64)
        bbox = values[:4].copy()
        bbox[[0, 2]] *= width
        bbox[[1, 3]] *= height
        keypoints = np.empty((EXPECTED_KEYPOINTS, 2), dtype=np.float64)
        visibility = np.empty(EXPECTED_KEYPOINTS, dtype=np.int32)
        for index in range(EXPECTED_KEYPOINTS):
            offset = 4 + index * 3
            keypoints[index] = (values[offset] * width, values[offset + 1] * height)
            visibility[index] = int(values[offset + 2])
        labels.append({"bbox_cxcywh": bbox, "kps": keypoints, "vis": visibility})
    return labels


def _box_xyxy(label: dict) -> np.ndarray:
    cx, cy, width, height = label["bbox_cxcywh"]
    return np.array([cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0])


def _best_corner_order(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve only cyclic drawing order; KP4 remains the centre in all cases."""

    best_prediction = prediction
    best_errors = np.linalg.norm(prediction - target, axis=1)
    best_score = float(np.mean(best_errors))
    for shift in range(1, 4):
        reordered = prediction.copy()
        reordered[:4] = np.roll(prediction[:4], -shift, axis=0)
        errors = np.linalg.norm(reordered - target, axis=1)
        score = float(np.mean(errors))
        if score < best_score:
            best_prediction, best_errors, best_score = reordered, errors, score
    return best_prediction, best_errors


def _rotation_error_deg(predicted_rotation: np.ndarray, target_rotation: np.ndarray) -> float:
    trace = float(np.trace(predicted_rotation @ target_rotation.T))
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pnp_error(prediction: np.ndarray, camera_matrix: np.ndarray, target_transform: np.ndarray):
    ok, rvec, tvec = cv2.solvePnP(
        LOCAL_SC_FRONT_MOUTH_KPS_M[:4],
        prediction[:4].astype(np.float64),
        camera_matrix.astype(np.float64),
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        return None
    predicted_rotation, _ = cv2.Rodrigues(rvec)
    translation_mm = float(np.linalg.norm(tvec.reshape(3) - target_transform[:3, 3]) * 1000.0)
    return translation_mm, _rotation_error_deg(predicted_rotation, target_transform[:3, :3])


def _metadata_by_image(metadata_root: Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for path in sorted(metadata_root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for camera in data.get("cameras", {}).values():
            image = camera.get("image")
            if image:
                output[image] = camera
    return output


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _evaluate(
    model, root: Path, split: str, conf: float, imgsz: int, device: str, limit: int | None
) -> dict:
    images = sorted((root / "images" / split).glob("*.png"))
    if limit is not None:
        images = images[:limit]
    metadata = _metadata_by_image(root / "metadata")
    centre_errors_px: list[float] = []
    corner_errors_px: list[float] = []
    translation_errors_mm: list[float] = []
    rotation_errors_deg: list[float] = []
    inference_ms: list[float] = []
    matched = missed = pnp_failures = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        camera_data = metadata.get(image_path.name)
        if image is None or camera_data is None:
            continue
        height, width = image.shape[:2]
        labels = _parse_labels(root / "labels" / split / f"{image_path.stem}.txt", width, height)
        target_metadata = camera_data.get("labels", [])
        if len(labels) != len(target_metadata):
            raise ValueError(f"metadata/label count mismatch for {image_path.name}")
        _sync_cuda()
        started = time.perf_counter()
        result = model(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        _sync_cuda()
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        predicted = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else np.empty((0, 5, 2))
        if predicted.ndim != 3 or predicted.shape[1:] != (EXPECTED_KEYPOINTS, 2):
            predicted = np.empty((0, EXPECTED_KEYPOINTS, 2))
        if len(labels) == 0 or len(predicted) == 0:
            missed += len(labels)
            continue
        costs = np.empty((len(labels), len(predicted)), dtype=np.float64)
        gates = np.empty(len(labels), dtype=np.float64)
        for gt_index, label in enumerate(labels):
            target_centre = label["kps"][4]
            box = _box_xyxy(label)
            gates[gt_index] = max(8.0, 0.75 * float(np.linalg.norm(box[2:] - box[:2])))
            costs[gt_index] = np.linalg.norm(predicted[:, 4] - target_centre, axis=1)
        gt_indices, pred_indices = linear_sum_assignment(costs)
        accepted: set[int] = set()
        for gt_index, prediction_index in zip(gt_indices, pred_indices):
            if costs[gt_index, prediction_index] > gates[gt_index]:
                continue
            accepted.add(int(gt_index))
            reordered, errors = _best_corner_order(predicted[prediction_index], labels[gt_index]["kps"])
            corner_errors_px.extend(errors[:4].tolist())
            centre_errors_px.append(float(errors[4]))
            camera_matrix = np.asarray(camera_data["camera_matrix"], dtype=np.float64)
            target_transform = np.asarray(target_metadata[gt_index]["T_camera_mouth"], dtype=np.float64)
            pnp = _pnp_error(reordered, camera_matrix, target_transform)
            if pnp is None:
                pnp_failures += 1
            else:
                translation_errors_mm.append(pnp[0])
                rotation_errors_deg.append(pnp[1])
            matched += 1
        missed += len(labels) - len(accepted)

    total = matched + missed
    return {
        "split": split,
        "images_considered": len(images),
        "matched_mouths": matched,
        "missed_mouths": missed,
        "miss_rate": float(missed / total) if total else None,
        "pnp_failures": pnp_failures,
        "mouth_center_error_px": _summary(centre_errors_px),
        "corner_error_px": _summary(corner_errors_px),
        "single_view_translation_error_mm": _summary(translation_errors_mm),
        "single_view_rotation_error_deg": _summary(rotation_errors_deg),
        "inference_ms": _summary(inference_ms),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    data_yaml = args.data.expanduser().resolve()
    root = data_yaml.parent
    if "kpt_shape: [5, 3]" not in data_yaml.read_text(encoding="utf-8"):
        raise RuntimeError(f"{data_yaml} is not a physical SC-mouth dataset")
    model = YOLO(str(args.weights.expanduser().resolve()))
    report = {
        "weights": str(args.weights),
        "data": str(data_yaml),
        "label_convention": "physical_front_mouth_22.407x8.100mm_plus_center",
        "evaluation": _evaluate(
            model, root, args.split, args.conf, args.imgsz, args.device, args.limit
        ),
    }
    args.report.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.expanduser().resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
