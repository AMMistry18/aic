#!/usr/bin/env python3
"""
Evaluate classical HSV-based SC port detection against ground truth.

Workflow:
  1. Load each image in pose_sc/images/{train,val}
  2. HSV mask for blue, find connected components
  3. For each blob, compute centroid (predicted port center in pixels)
  4. Read ground truth from the matching YOLO label file
     - bbox center in normalized coords = GT port center
  5. Match predicted blobs to GT centers by nearest-neighbor
  6. Report pixel error statistics

Usage:
    pixi run python3 ~/eval_color_sc.py
    pixi run python3 ~/eval_color_sc.py --save-viz ~/sc_color_viz --limit 20
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.path.expanduser("~/aic_perception_data/pose_sc"))

# HSV range — same as filter_no_blue. Tune if needed.
BLUE_LOWER = np.array([90, 80, 60], dtype=np.uint8)
BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)

# Minimum blob area in pixels to count as an SC port candidate
MIN_BLOB_AREA = 15
MAX_BLOB_AREA = 50000  # reject huge blobs (false positives from background)


def detect_sc_blobs(img_bgr):
    """
    Return list of (cx, cy, area, bbox) for blue blobs in the image.
    bbox is (x, y, w, h).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    # Morphological cleanup — close gaps between the 2 duplex circles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs = []
    for i in range(1, n_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        blobs.append((float(cx), float(cy), int(area), (int(x), int(y), int(w), int(h))))

    return blobs, mask


def parse_gt_label(label_path, img_w, img_h):
    """
    Parse YOLO-pose label file. Return list of (cx, cy, w, h) bbox tuples
    in pixel coords — one per ground truth SC port.
    """
    gts = []
    if not label_path.exists():
        return gts
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            gts.append((cx, cy, w, h))
    return gts


def match_blobs_to_gt(blobs, gts):
    """
    Greedy nearest-neighbor match. Returns list of (pred_cx, pred_cy, gt_cx, gt_cy, error_px).
    Unmatched GTs contribute a 'missed' record with error=None.
    """
    matched = []
    unmatched_gt = []
    used_blobs = set()

    for gt_cx, gt_cy, gt_w, gt_h in gts:
        best = None
        best_dist = float('inf')
        for i, (bcx, bcy, _, _) in enumerate(blobs):
            if i in used_blobs:
                continue
            d = np.sqrt((bcx - gt_cx) ** 2 + (bcy - gt_cy) ** 2)
            if d < best_dist:
                best_dist = d
                best = i
        if best is not None and best_dist < max(gt_w, gt_h) * 2:
            bcx, bcy, _, _ = blobs[best]
            matched.append((bcx, bcy, gt_cx, gt_cy, best_dist))
            used_blobs.add(best)
        else:
            unmatched_gt.append((gt_cx, gt_cy))

    false_positives = [b for i, b in enumerate(blobs) if i not in used_blobs]
    return matched, unmatched_gt, false_positives


def process_split(split, save_viz_dir, limit):
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split

    if not img_dir.exists():
        print(f"[{split}] missing, skipping")
        return [], 0, 0, 0

    images = sorted(img_dir.glob("*.png"))
    if limit:
        images = images[:limit]

    all_errors = []
    total_gt = 0
    total_matched = 0
    total_fp = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gts = parse_gt_label(lbl_dir / (img_path.stem + ".txt"), w, h)
        if not gts:
            continue
        total_gt += len(gts)

        blobs, mask = detect_sc_blobs(img)
        matched, missed, fps = match_blobs_to_gt(blobs, gts)
        total_matched += len(matched)
        total_fp += len(fps)

        for _, _, _, _, err in matched:
            all_errors.append(err)

        if save_viz_dir is not None:
            viz = img.copy()
            # GT = green
            for cx, cy, bw, bh in gts:
                cv2.rectangle(viz,
                              (int(cx - bw/2), int(cy - bh/2)),
                              (int(cx + bw/2), int(cy + bh/2)),
                              (0, 255, 0), 2)
                cv2.circle(viz, (int(cx), int(cy)), 4, (0, 255, 0), -1)
            # Predicted blobs = red
            for bcx, bcy, area, (x, y, bw, bh) in blobs:
                cv2.rectangle(viz, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.circle(viz, (int(bcx), int(bcy)), 4, (0, 0, 255), -1)
            # Matched lines = yellow
            for bcx, bcy, gcx, gcy, err in matched:
                cv2.line(viz, (int(bcx), int(bcy)), (int(gcx), int(gcy)),
                         (0, 255, 255), 1)
                cv2.putText(viz, f"{err:.1f}px",
                            (int((bcx + gcx) / 2) + 5, int((bcy + gcy) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            out_path = Path(save_viz_dir) / split / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), viz)

    return all_errors, total_gt, total_matched, total_fp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-viz", type=str, default=None,
                   help="directory to save visualization images")
    p.add_argument("--limit", type=int, default=None,
                   help="max images per split (for quick testing)")
    args = p.parse_args()

    all_errors = []
    grand_gt = 0
    grand_match = 0
    grand_fp = 0

    for split in ["train", "val"]:
        print(f"\n=== {split} ===")
        errs, ngt, nmatch, nfp = process_split(split, args.save_viz, args.limit)
        if not errs:
            print(f"  no GT labels found")
            continue
        arr = np.array(errs)
        print(f"  ground truth ports: {ngt}")
        print(f"  matched:            {nmatch}  ({100*nmatch/ngt:.1f}%)")
        print(f"  missed:             {ngt - nmatch}")
        print(f"  false positives:    {nfp}")
        print(f"  pixel error — mean: {arr.mean():.2f}  median: {np.median(arr):.2f}  "
              f"p95: {np.percentile(arr, 95):.2f}  max: {arr.max():.2f}")
        all_errors.extend(errs)
        grand_gt += ngt
        grand_match += nmatch
        grand_fp += nfp

    if all_errors:
        arr = np.array(all_errors)
        print(f"\n=== COMBINED ===")
        print(f"  total GT:     {grand_gt}")
        print(f"  matched:      {grand_match}  ({100*grand_match/grand_gt:.1f}%)")
        print(f"  false pos:    {grand_fp}")
        print(f"  mean px err:  {arr.mean():.2f}")
        print(f"  median:       {np.median(arr):.2f}")
        print(f"  p95:          {np.percentile(arr, 95):.2f}")

        if args.save_viz:
            print(f"\n  visualizations saved to {args.save_viz}/")


if __name__ == "__main__":
    main()