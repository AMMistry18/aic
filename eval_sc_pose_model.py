#!/usr/bin/env python3
"""Evaluate SC YOLO pose weights on held-out validation split."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# This repository uses a nested Python package.  Prefer the worktree source over
# a potentially stale package already installed in the Pixi environment.
_SOURCE_ROOT = Path(__file__).resolve().parent / "aic_example_policies"
if _SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(_SOURCE_ROOT))

from aic_example_policies.ros.perception_core import PerceptionCore


def parse_label_file(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            rows.append(parts)
    return rows


def bbox_center(parts, w, h):
    return np.array([float(parts[1]) * w, float(parts[2]) * h], dtype=np.float64)


def bbox_xyxy(parts, w, h):
    cx, cy = bbox_center(parts, w, h)
    bw, bh = float(parts[3]) * w, float(parts[4]) * h
    return np.array([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0])


def visible_keypoints(parts, w, h):
    """Return four GT keypoints and their YOLO visibility mask."""
    keypoints = np.zeros((4, 2), dtype=np.float64)
    visible = np.zeros(4, dtype=bool)
    for index in range(4):
        base = 5 + 3 * index
        if len(parts) <= base + 2:
            continue
        keypoints[index] = (float(parts[base]) * w, float(parts[base + 1]) * h)
        visible[index] = float(parts[base + 2]) > 0.0
    return keypoints, visible


def _sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _summary(values):
    if not values:
        return {}
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def cyclic_keypoint_errors(predicted, ground_truth, visible):
    """Return the lowest-error cyclic corner correspondence.

    The live multiview controller deliberately accepts cyclic corner relabeling,
    so the evaluator must not penalize a prediction that differs only in which
    physical corner YOLO called keypoint zero.
    """
    predicted = np.asarray(predicted, dtype=np.float64)[:4, :2]
    ground_truth = np.asarray(ground_truth, dtype=np.float64)[:4, :2]
    visible = np.asarray(visible, dtype=bool)[:4]
    best = None
    for roll in range(4):
        rolled = np.roll(predicted, -roll, axis=0)
        errors = np.linalg.norm(rolled[visible] - ground_truth[visible], axis=1)
        score = float(errors.mean()) if len(errors) else float("inf")
        if best is None or score < best[0]:
            best = (score, roll, errors)
    return best[1], best[2]


def globally_match_detections(gts, detections, width, height):
    """Globally match GT and detections by pose-keypoint centre distance."""
    if not gts or not detections:
        return []
    costs = np.empty((len(gts), len(detections)), dtype=np.float64)
    gates = np.empty(len(gts), dtype=np.float64)
    for gt_index, gt in enumerate(gts):
        gt_kps, visible = visible_keypoints(gt, width, height)
        gt_centre = (
            gt_kps[visible].mean(axis=0)
            if np.any(visible)
            else bbox_center(gt, width, height)
        )
        gt_box = bbox_xyxy(gt, width, height)
        gates[gt_index] = max(8.0, 0.75 * float(np.linalg.norm(gt_box[2:] - gt_box[:2])))
        for det_index, det in enumerate(detections):
            det_kps = np.asarray(det.get("kps", []), dtype=np.float64)
            det_centre = (
                det_kps[:4, :2].mean(axis=0)
                if det_kps.ndim == 2 and len(det_kps) >= 4
                else np.asarray(det["centroid"], dtype=np.float64)
            )
            costs[gt_index, det_index] = np.linalg.norm(det_centre - gt_centre)
    gt_indices, det_indices = linear_sum_assignment(costs)
    return [
        (int(gt_index), int(det_index), float(costs[gt_index, det_index]))
        for gt_index, det_index in zip(gt_indices, det_indices)
        if costs[gt_index, det_index] <= gates[gt_index]
    ]


def evaluate_runtime_variant(
    model, images, labels_dir, conf, *, crop_refine, pad_scale, warmup
):
    """Headless 2D runtime-path evaluation against saved YOLO-pose labels.

    The dataset contains native camera PNGs and TF-projected 2D labels, so no
    simulator, Gazebo renderer, or ROS graph is needed.  It deliberately
    measures association/miss and keypoint error instead of claiming a 3D
    result: this dataset's metadata does not contain K/extrinsics.
    """
    core = PerceptionCore(sc_weights="in_memory.pt")
    core._sc_yolo = model
    for image_path in images[:warmup]:
        image = cv2.imread(str(image_path))
        if image is not None:
            core.detect_sc_pose(
                image,
                conf_thresh=conf,
                crop_refine=crop_refine,
                crop_pad_scale=pad_scale,
            )

    keypoint_errors = []
    centre_errors = []
    inference_ms = []
    matched = 0
    misses = 0
    associations_rejected = 0
    detection_count = 0
    labeled_images = 0
    nonzero_roll_matches = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gts = parse_label_file(labels_dir / f"{image_path.stem}.txt")
        if not gts:
            continue
        labeled_images += 1
        _sync_cuda()
        started = time.perf_counter()
        detections = core.detect_sc_pose(
            image,
            conf_thresh=conf,
            crop_refine=crop_refine,
            crop_pad_scale=pad_scale,
        )
        _sync_cuda()
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        detection_count += len(detections)

        pairs = globally_match_detections(gts, detections, w, h)
        for gt_index, det_index, distance in pairs:
            gt = gts[gt_index]
            det = detections[det_index]
            if "kps" not in det or len(det["kps"]) < 4:
                associations_rejected += 1
                continue
            gt_kps, visible = visible_keypoints(gt, w, h)
            if not np.any(visible):
                associations_rejected += 1
                continue
            pred_kps = np.asarray(det["kps"], dtype=np.float64)[:4]
            roll, errors = cyclic_keypoint_errors(pred_kps, gt_kps, visible)
            nonzero_roll_matches += int(roll != 0)
            keypoint_errors.extend(errors.tolist())
            centre_errors.append(distance)
            matched += 1
        misses += len(gts) - len(pairs)

    evaluated_labels = matched + misses + associations_rejected
    return {
        "mode": "crop_refine" if crop_refine else "baseline",
        "crop_pad_scale": pad_scale if crop_refine else None,
        "labeled_images": labeled_images,
        "matched_labels": matched,
        "missed_labels": misses,
        "association_rejected": associations_rejected,
        "miss_rate": (
            float(misses + associations_rejected) / evaluated_labels
            if evaluated_labels
            else None
        ),
        "nonzero_cyclic_roll_matches": nonzero_roll_matches,
        "detections_per_image": (
            float(detection_count) / labeled_images if labeled_images else 0.0
        ),
        "keypoint_error_px": _summary(keypoint_errors),
        "pose_center_error_px": _summary(centre_errors),
        "inference_ms": _summary(inference_ms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--data",
        default=str(Path(os.path.expanduser("~/aic_perception_data/pose_sc/aic_sc_pose.yaml"))),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument(
        "--report",
        default=str(Path(__file__).resolve().parent / "outputs" / "sc_pose_pipeline" / "eval_sc_pose.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--crop-pad-scales",
        default="",
        help="comma-separated opt-in native-crop pad scales; skips model.val and sweeps 2D labels",
    )
    parser.add_argument("--warmup", type=int, default=20, help="untimed crop-refine warmup images")
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    pad_scales = [float(value) for value in args.crop_pad_scales.split(",") if value.strip()]
    # model.val evaluates only full-frame YOLO metrics; it cannot exercise the
    # two-stage path and would double a crop-sweep's runtime for no added signal.
    val_metrics = None if pad_scales else model.val(data=args.data, split=args.split, conf=args.conf, verbose=False)
    metrics = {}
    if hasattr(val_metrics, "results_dict"):
        metrics = {k: float(v) for k, v in val_metrics.results_dict.items()}

    data_root = Path(args.data).resolve().parent
    images_dir = data_root / "images" / args.split
    labels_dir = data_root / "labels" / args.split
    images = sorted(images_dir.glob("*.png"))
    if args.limit is not None:
        images = images[: args.limit]

    if pad_scales:
        warmup = max(0, args.warmup)
        report = {
            "weights": args.weights,
            "data": args.data,
            "split": args.split,
            "images_evaluated": len(images),
            "baseline": evaluate_runtime_variant(
                model,
                images,
                labels_dir,
                args.conf,
                crop_refine=False,
                pad_scale=None,
                warmup=warmup,
            ),
            "crop_refine_sweeps": [
                evaluate_runtime_variant(
                    model,
                    images,
                    labels_dir,
                    args.conf,
                    crop_refine=True,
                    pad_scale=scale,
                    warmup=warmup,
                )
                for scale in pad_scales
            ],
            "note": "2D-only: saved pose_sc metadata lacks camera intrinsics/extrinsics for 3D error.",
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"\nSaved report: {report_path}")
        return

    center_errors = []
    misses = 0
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = parse_label_file(labels_dir / f"{image_path.stem}.txt")
        if not gts:
            continue
        result = model(img, conf=args.conf, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            misses += len(gts)
            continue
        pred_centers = result.boxes.xywh.cpu().numpy()[:, :2]
        used = set()
        for gt in gts:
            gc = bbox_center(gt, w, h)
            best_idx = None
            best_dist = 1e9
            for idx, pc in enumerate(pred_centers):
                if idx in used:
                    continue
                d = float(np.linalg.norm(gc - pc))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx is None:
                misses += 1
                continue
            used.add(best_idx)
            center_errors.append(best_dist)

    center_summary = {}
    if center_errors:
        arr = np.array(center_errors)
        center_summary = {
            "mean_px": float(arr.mean()),
            "median_px": float(np.median(arr)),
            "p95_px": float(np.percentile(arr, 95)),
            "max_px": float(arr.max()),
        }

    report = {
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "ultralytics_metrics": metrics,
        "center_error": center_summary,
        "missed_labels": misses,
        "images_evaluated": len(images),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
