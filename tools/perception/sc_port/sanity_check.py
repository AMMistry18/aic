#!/usr/bin/env python3
"""
Audit SC GT labels against the HSV SC detector.

For each labeled sample, this script computes:
- GT center in pixels from visible keypoints (fallback bbox center).
- HSV center from PerceptionCore.detect_sc().
- Center error in pixels + normalized by image diagonal.

It writes:
- Detailed per-label audit CSV + JSON.
- Summary JSON + Markdown report.
- Failing-samples JSON.
- Optional cleaned dataset with updated YAML.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "aic_example_policies"))

from aic_example_policies.ros.perception_core import PerceptionCore  # noqa: E402


@dataclass
class Thresholds:
    max_center_err_px: float
    max_center_err_norm_diag: float
    min_iou: float
    max_axis_angle_err_deg: float
    use_axis_gate: bool
    outlier_factor_mad: float
    min_abs_outlier_px: float


def parse_yolo_pose_label(line: str, img_w: int, img_h: int):
    parts = line.strip().split()
    if len(parts) < 5 + 12:
        return None

    cls_id = int(parts[0])
    cx = float(parts[1]) * img_w
    cy = float(parts[2]) * img_h
    bw = float(parts[3]) * img_w
    bh = float(parts[4]) * img_h

    kps = []
    offset = 5
    for _ in range(4):
        kx = float(parts[offset]) * img_w
        ky = float(parts[offset + 1]) * img_h
        vis = int(float(parts[offset + 2]))
        kps.append((kx, ky, vis))
        offset += 3

    visible = [(kx, ky) for (kx, ky, vis) in kps if vis > 0]
    if visible:
        gt_cx = float(np.mean([p[0] for p in visible]))
        gt_cy = float(np.mean([p[1] for p in visible]))
        center_source = "keypoints"
    else:
        gt_cx = cx
        gt_cy = cy
        center_source = "bbox"

    return {
        "class_id": cls_id,
        "line_raw": line.strip(),
        "bbox_center": (cx, cy),
        "gt_center": (gt_cx, gt_cy),
        "center_source": center_source,
        "bbox_xyxy": (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0),
        "kps": kps,
    }


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 1e-9:
        return 0.0
    return inter / denom


def axis_angle_deg(axis):
    (x0, y0), (x1, y1) = axis
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def wrap_deg(delta):
    d = (delta + 180.0) % 360.0 - 180.0
    return abs(d)


def gt_axis_from_visible_kps(gt_kps):
    vis = [kp for kp in gt_kps if kp[2] > 0]
    if len(vis) < 2:
        return None
    pts = np.array([[p[0], p[1]] for p in vis], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = box[np.argsort(box[:, 1])]
    top = box[:2]
    bot = box[2:]
    return (
        tuple(((top[0] + top[1]) * 0.5).tolist()),
        tuple(((bot[0] + bot[1]) * 0.5).tolist()),
    )


def summarize(values):
    if not values:
        return None
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def robust_outlier_mask(values, factor):
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-9:
        mad = 1e-9
    robust_z = 0.6745 * (arr - med) / mad
    mask = np.abs(robust_z) > factor
    return mask.tolist()


def choose_blob_for_gt(preds, gt):
    if not preds:
        return None, None

    gt_center = np.array(gt["gt_center"], dtype=np.float64)
    gt_bbox = gt["bbox_xyxy"]
    scored = []
    for idx, pred in enumerate(preds):
        px, py, pw, ph = pred["bbox"]
        pred_bbox = (float(px), float(py), float(px + pw), float(py + ph))
        iou = bbox_iou(gt_bbox, pred_bbox)
        dist = float(np.linalg.norm(np.array(pred["centroid"], dtype=np.float64) - gt_center))
        scored.append(
            {
                "idx": idx,
                "iou": iou,
                "dist": dist,
                "pred": pred,
            }
        )

    best_iou = max(scored, key=lambda s: s["iou"])
    if best_iou["iou"] > 0.0:
        return best_iou["pred"], "best_iou"
    best_nearest = min(scored, key=lambda s: s["dist"])
    return best_nearest["pred"], "nearest_center"


def evaluate_match(gt, pred, img_w, img_h):
    gt_cx, gt_cy = gt["gt_center"]
    pred_cx, pred_cy = pred["centroid"]
    center_err_px = float(np.linalg.norm(np.array([gt_cx - pred_cx, gt_cy - pred_cy])))
    img_diag = float(np.hypot(img_w, img_h))
    center_err_norm_diag = center_err_px / max(img_diag, 1e-9)

    px, py, pw, ph = pred["bbox"]
    pred_bbox = (float(px), float(py), float(px + pw), float(py + ph))
    iou = bbox_iou(gt["bbox_xyxy"], pred_bbox)

    axis_err_deg = None
    gt_axis = gt_axis_from_visible_kps(gt["kps"])
    pred_axis = pred.get("major_axis")
    if gt_axis is not None and pred_axis is not None:
        err = wrap_deg(axis_angle_deg(gt_axis) - axis_angle_deg(pred_axis))
        axis_err_deg = float(min(err, abs(180.0 - err)))

    return {
        "center_err_px": center_err_px,
        "center_err_norm_diag": center_err_norm_diag,
        "iou": iou,
        "axis_err_deg": axis_err_deg,
    }


def audit_dataset(pc, dataset_root: Path):
    records = []
    splits = ("train", "val")
    for split in splits:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in sorted(images_dir.glob("*.png")):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            bgr = cv2.imread(str(image_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            hsv_preds = pc.detect_sc(bgr)

            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            for line_idx, line in enumerate(lines):
                gt = parse_yolo_pose_label(line, w, h)
                if gt is None:
                    continue

                matched, match_method = choose_blob_for_gt(hsv_preds, gt)
                if matched is None:
                    rec = {
                        "split": split,
                        "image": image_path.name,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "label_line_idx": line_idx,
                        "gt_class_id": gt["class_id"],
                        "img_w": w,
                        "img_h": h,
                        "num_hsv_blobs": 0,
                        "gt_center_x": float(gt["gt_center"][0]),
                        "gt_center_y": float(gt["gt_center"][1]),
                        "gt_bbox_center_x": float(gt["bbox_center"][0]),
                        "gt_bbox_center_y": float(gt["bbox_center"][1]),
                        "gt_center_source": gt["center_source"],
                        "hsv_center_x": None,
                        "hsv_center_y": None,
                        "hsv_bbox_x": None,
                        "hsv_bbox_y": None,
                        "hsv_bbox_w": None,
                        "hsv_bbox_h": None,
                        "hsv_area": None,
                        "match_method": None,
                        "center_err_px": None,
                        "center_err_norm_diag": None,
                        "iou": None,
                        "axis_err_deg": None,
                        "gate_center_px": False,
                        "gate_center_norm_diag": False,
                        "gate_iou": False,
                        "gate_axis": False,
                        "gate_outlier": False,
                        "pass": False,
                        "fail_reason": "no_hsv_blob",
                        "label_line_raw": gt["line_raw"],
                    }
                    records.append(rec)
                    continue

                metrics = evaluate_match(gt, matched, w, h)
                x, y, bw, bh = matched["bbox"]
                axis_gate = metrics["axis_err_deg"] is None
                rec = {
                    "split": split,
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_line_idx": line_idx,
                    "gt_class_id": gt["class_id"],
                    "img_w": w,
                    "img_h": h,
                    "num_hsv_blobs": len(hsv_preds),
                    "gt_center_x": float(gt["gt_center"][0]),
                    "gt_center_y": float(gt["gt_center"][1]),
                    "gt_bbox_center_x": float(gt["bbox_center"][0]),
                    "gt_bbox_center_y": float(gt["bbox_center"][1]),
                    "gt_center_source": gt["center_source"],
                    "hsv_center_x": float(matched["centroid"][0]),
                    "hsv_center_y": float(matched["centroid"][1]),
                    "hsv_bbox_x": int(x),
                    "hsv_bbox_y": int(y),
                    "hsv_bbox_w": int(bw),
                    "hsv_bbox_h": int(bh),
                    "hsv_area": int(matched["area"]),
                    "match_method": match_method,
                    "center_err_px": metrics["center_err_px"],
                    "center_err_norm_diag": metrics["center_err_norm_diag"],
                    "iou": metrics["iou"],
                    "axis_err_deg": metrics["axis_err_deg"],
                    "gate_center_px": False,
                    "gate_center_norm_diag": False,
                    "gate_iou": False,
                    "gate_axis": axis_gate,
                    "gate_outlier": False,
                    "pass": False,
                    "fail_reason": "",
                    "label_line_raw": gt["line_raw"],
                }
                records.append(rec)

    return records


def apply_quality_gate(records, thresholds: Thresholds):
    measured_indices = [i for i, r in enumerate(records) if r["center_err_px"] is not None]
    center_errors = [records[i]["center_err_px"] for i in measured_indices]
    outlier_flags = robust_outlier_mask(center_errors, thresholds.outlier_factor_mad)

    outlier_map = {idx: outlier_flags[pos] for pos, idx in enumerate(measured_indices)}

    for idx, rec in enumerate(records):
        if rec["center_err_px"] is None:
            rec["pass"] = False
            if not rec["fail_reason"]:
                rec["fail_reason"] = "no_hsv_blob"
            continue

        rec["gate_center_px"] = rec["center_err_px"] <= thresholds.max_center_err_px
        rec["gate_center_norm_diag"] = (
            rec["center_err_norm_diag"] <= thresholds.max_center_err_norm_diag
        )
        rec["gate_iou"] = rec["iou"] >= thresholds.min_iou
        if not thresholds.use_axis_gate:
            rec["gate_axis"] = True
        elif rec["axis_err_deg"] is None:
            rec["gate_axis"] = True
        else:
            rec["gate_axis"] = rec["axis_err_deg"] <= thresholds.max_axis_angle_err_deg

        robust_outlier = bool(outlier_map.get(idx, False)) and (
            rec["center_err_px"] >= thresholds.min_abs_outlier_px
        )
        rec["gate_outlier"] = not robust_outlier

        passed = (
            rec["gate_center_px"]
            and rec["gate_center_norm_diag"]
            and rec["gate_iou"]
            and rec["gate_axis"]
            and rec["gate_outlier"]
        )
        rec["pass"] = passed

        if not passed and not rec["fail_reason"]:
            reasons = []
            if not rec["gate_center_px"]:
                reasons.append("center_px")
            if not rec["gate_center_norm_diag"]:
                reasons.append("center_norm")
            if not rec["gate_iou"]:
                reasons.append("iou")
            if not rec["gate_axis"]:
                reasons.append("axis")
            if not rec["gate_outlier"]:
                reasons.append("outlier")
            rec["fail_reason"] = ",".join(reasons)

    return records


def build_summary(records, thresholds: Thresholds):
    total = len(records)
    passed = sum(1 for r in records if r["pass"])
    failed = total - passed
    pass_rate = (passed / total) if total else 0.0

    measured = [r for r in records if r["center_err_px"] is not None]
    center_px = [r["center_err_px"] for r in measured]
    center_norm = [r["center_err_norm_diag"] for r in measured]
    ious = [r["iou"] for r in measured]
    axis_vals = [r["axis_err_deg"] for r in measured if r["axis_err_deg"] is not None]

    by_split = {}
    for split in ("train", "val"):
        subset = [r for r in records if r["split"] == split]
        if not subset:
            continue
        split_pass = sum(1 for r in subset if r["pass"])
        by_split[split] = {
            "count": len(subset),
            "pass_count": split_pass,
            "fail_count": len(subset) - split_pass,
            "pass_rate": split_pass / len(subset),
            "center_err_px": summarize([r["center_err_px"] for r in subset if r["center_err_px"] is not None]),
            "center_err_norm_diag": summarize([r["center_err_norm_diag"] for r in subset if r["center_err_norm_diag"] is not None]),
            "iou": summarize([r["iou"] for r in subset if r["iou"] is not None]),
        }

    fail_reasons = {}
    for r in records:
        if r["pass"]:
            continue
        key = r["fail_reason"] or "unknown"
        fail_reasons[key] = fail_reasons.get(key, 0) + 1

    return {
        "counts": {
            "labels_total": total,
            "labels_with_hsv_match": len(measured),
            "pass_count": passed,
            "fail_count": failed,
            "pass_rate": pass_rate,
        },
        "thresholds": {
            "max_center_err_px": thresholds.max_center_err_px,
            "max_center_err_norm_diag": thresholds.max_center_err_norm_diag,
            "min_iou": thresholds.min_iou,
            "max_axis_angle_err_deg": thresholds.max_axis_angle_err_deg,
            "use_axis_gate": thresholds.use_axis_gate,
            "outlier_factor_mad": thresholds.outlier_factor_mad,
            "min_abs_outlier_px": thresholds.min_abs_outlier_px,
        },
        "metrics": {
            "center_err_px": summarize(center_px),
            "center_err_norm_diag": summarize(center_norm),
            "iou": summarize(ious),
            "axis_err_deg": summarize(axis_vals),
        },
        "splits": by_split,
        "match_method_counts": {
            "best_iou": sum(1 for r in records if r["match_method"] == "best_iou"),
            "nearest_center": sum(1 for r in records if r["match_method"] == "nearest_center"),
            "no_match": sum(1 for r in records if r["match_method"] is None),
        },
        "fail_reason_counts": fail_reasons,
    }


def write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, records):
    if not records:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_markdown_summary(path: Path, summary: dict):
    c = summary["counts"]
    m = summary["metrics"]
    t = summary["thresholds"]
    center = m["center_err_px"] or {}
    norm = m["center_err_norm_diag"] or {}
    iou = m["iou"] or {}

    lines = []
    lines.append("# SC GT vs HSV Audit Summary")
    lines.append("")
    lines.append("## Quality Gate")
    lines.append(f"- max_center_err_px: `{t['max_center_err_px']}`")
    lines.append(f"- max_center_err_norm_diag: `{t['max_center_err_norm_diag']}`")
    lines.append(f"- min_iou: `{t['min_iou']}`")
    lines.append(f"- max_axis_angle_err_deg: `{t['max_axis_angle_err_deg']}`")
    lines.append(f"- use_axis_gate: `{t['use_axis_gate']}`")
    lines.append(
        f"- robust outlier: `|robust_z| <= {t['outlier_factor_mad']}` "
        f"or `center_err_px < {t['min_abs_outlier_px']}`"
    )
    lines.append("")
    lines.append("## Counts")
    lines.append(f"- labels_total: `{c['labels_total']}`")
    lines.append(f"- labels_with_hsv_match: `{c['labels_with_hsv_match']}`")
    lines.append(f"- pass_count: `{c['pass_count']}`")
    lines.append(f"- fail_count: `{c['fail_count']}`")
    lines.append(f"- pass_rate: `{c['pass_rate']:.4f}`")
    lines.append("")
    lines.append("## Metric Distribution")
    if center:
        lines.append(
            f"- center_err_px mean/median/p95/max: "
            f"`{center['mean']:.3f}/{center['median']:.3f}/{center['p95']:.3f}/{center['max']:.3f}`"
        )
    if norm:
        lines.append(
            f"- center_err_norm_diag mean/median/p95/max: "
            f"`{norm['mean']:.6f}/{norm['median']:.6f}/{norm['p95']:.6f}/{norm['max']:.6f}`"
        )
    if iou:
        lines.append(
            f"- iou mean/median/p95/max: "
            f"`{iou['mean']:.3f}/{iou['median']:.3f}/{iou['p95']:.3f}/{iou['max']:.3f}`"
        )
    lines.append("")
    lines.append("## Match Logic")
    mm = summary["match_method_counts"]
    lines.append(f"- best_iou: `{mm['best_iou']}`")
    lines.append(f"- nearest_center fallback: `{mm['nearest_center']}`")
    lines.append(f"- no_match: `{mm['no_match']}`")
    lines.append("")
    lines.append("## Fail Reasons")
    for k, v in sorted(summary["fail_reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_dataset_yaml(dataset_path: Path):
    text = f"""path: {dataset_path}
train: images/train
val: images/val
nc: 1
names: ['sc_port']
kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
"""
    with open(dataset_path / "aic_sc_pose.yaml", "w", encoding="utf-8") as f:
        f.write(text)


def maybe_clean_dataset(records, input_root: Path, output_root: Path):
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    keep_map = {}
    for rec in records:
        key = (rec["split"], rec["image"])
        keep_map.setdefault(key, set())
        if rec["pass"]:
            keep_map[key].add(rec["label_line_idx"])

    for split in ("train", "val"):
        in_images = input_root / "images" / split
        in_labels = input_root / "labels" / split
        if not in_images.exists() or not in_labels.exists():
            continue
        out_images = output_root / "images" / split
        out_labels = output_root / "labels" / split

        for image_path in sorted(in_images.glob("*.png")):
            label_path = in_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            key = (split, image_path.name)
            keep_indices = keep_map.get(key, set())
            if not keep_indices:
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            kept_lines = [ln for idx, ln in enumerate(lines) if idx in keep_indices]
            if not kept_lines:
                continue

            shutil.copy2(image_path, out_images / image_path.name)
            with open(out_labels / label_path.name, "w", encoding="utf-8") as f:
                f.write("\n".join(kept_lines))

    # copy metadata and root counters if present
    metadata_src = input_root / "metadata"
    if metadata_src.exists():
        shutil.copytree(metadata_src, output_root / "metadata")
    run_counter = input_root / ".run_counter"
    if run_counter.exists():
        shutil.copy2(run_counter, output_root / ".run_counter")

    write_dataset_yaml(output_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(Path(os.path.expanduser("~/aic_perception_data/pose_sc_gt_raw"))),
        help="Raw SC GT dataset root.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(REPO_ROOT / "outputs" / "sc_pose_pipeline" / "sanity"),
        help="Directory for audit artifacts.",
    )
    parser.add_argument(
        "--cleaned-output",
        default=str(Path(os.path.expanduser("~/aic_perception_data/pose_sc_gt_cleaned"))),
        help="Cleaned dataset path when --write-cleaned-dataset is set.",
    )
    parser.add_argument("--max-center-err-px", type=float, default=26.0)
    parser.add_argument("--max-center-err-norm-diag", type=float, default=0.018)
    parser.add_argument("--min-iou", type=float, default=0.12)
    parser.add_argument("--max-axis-err-deg", type=float, default=35.0)
    parser.add_argument(
        "--use-axis-gate",
        action="store_true",
        help="Require axis-angle consistency between GT and HSV blob.",
    )
    parser.add_argument("--outlier-factor-mad", type=float, default=4.5)
    parser.add_argument("--min-abs-outlier-px", type=float, default=12.0)
    parser.add_argument(
        "--write-cleaned-dataset",
        action="store_true",
        help="Write cleaned dataset containing only passing labels.",
    )
    args = parser.parse_args()

    input_root = Path(args.input)
    report_dir = Path(args.report_dir)
    cleaned_output = Path(args.cleaned_output)

    report_dir.mkdir(parents=True, exist_ok=True)

    thresholds = Thresholds(
        max_center_err_px=args.max_center_err_px,
        max_center_err_norm_diag=args.max_center_err_norm_diag,
        min_iou=args.min_iou,
        max_axis_angle_err_deg=args.max_axis_err_deg,
        use_axis_gate=args.use_axis_gate,
        outlier_factor_mad=args.outlier_factor_mad,
        min_abs_outlier_px=args.min_abs_outlier_px,
    )

    pc = PerceptionCore()
    records = audit_dataset(pc, input_root)
    records = apply_quality_gate(records, thresholds)
    summary = build_summary(records, thresholds)

    failures = [r for r in records if not r["pass"]]
    fail_samples = sorted({f"{r['split']}/{r['image']}#{r['label_line_idx']}" for r in failures})

    audit_json = report_dir / "sc_gt_hsv_audit.json"
    audit_csv = report_dir / "sc_gt_hsv_audit.csv"
    summary_json = report_dir / "sc_gt_hsv_summary.json"
    summary_md = report_dir / "sc_gt_hsv_summary.md"
    failures_json = report_dir / "sc_gt_hsv_failures.json"
    fail_list_txt = report_dir / "sc_gt_hsv_failing_samples.txt"

    write_json(audit_json, records)
    write_csv(audit_csv, records)
    write_json(summary_json, summary)
    write_markdown_summary(summary_md, summary)
    write_json(failures_json, failures)
    with open(fail_list_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(fail_samples))

    cleaned_written = False
    if args.write_cleaned_dataset:
        maybe_clean_dataset(records, input_root, cleaned_output)
        cleaned_written = True

    out = {
        "input_root": str(input_root),
        "report_dir": str(report_dir),
        "thresholds": summary["thresholds"],
        "counts": summary["counts"],
        "metrics": summary["metrics"],
        "match_method_counts": summary["match_method_counts"],
        "fail_reason_counts": summary["fail_reason_counts"],
        "artifacts": {
            "audit_json": str(audit_json),
            "audit_csv": str(audit_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "failures_json": str(failures_json),
            "failing_samples_txt": str(fail_list_txt),
        },
        "cleaned_dataset": {
            "written": cleaned_written,
            "path": str(cleaned_output) if cleaned_written else None,
            "yaml": str(cleaned_output / "aic_sc_pose.yaml") if cleaned_written else None,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
