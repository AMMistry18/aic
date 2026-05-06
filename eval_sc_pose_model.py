#!/usr/bin/env python3
"""Evaluate SC YOLO pose weights on held-out validation split."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    val_metrics = model.val(data=args.data, split=args.split, conf=args.conf, verbose=False)
    metrics = {}
    if hasattr(val_metrics, "results_dict"):
        metrics = {k: float(v) for k, v in val_metrics.results_dict.items()}

    data_root = Path(args.data).resolve().parent
    images_dir = data_root / "images" / args.split
    labels_dir = data_root / "labels" / args.split
    images = sorted(images_dir.glob("*.png"))
    if args.limit is not None:
        images = images[: args.limit]

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
