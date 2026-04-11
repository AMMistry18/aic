#!/usr/bin/env python3
"""
Filter out SC port images that don't contain visible blue ports.
Deletes both the image and its matching YOLO label file.

Usage:
    python filter_no_blue.py                  # dry run, shows what would be deleted
    python filter_no_blue.py --apply          # actually delete
    python filter_no_blue.py --apply --min-blue-pct 0.05
"""
import argparse
import os
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(os.path.expanduser("~/aic_perception_data/pose_sc"))

# HSV range for SC port blue. SC ports in sim are a saturated cyan/blue.
# Tune these if needed — run with --debug to see blue masks of a few samples.
BLUE_LOWER = np.array([90, 80, 60], dtype=np.uint8)
BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)


def blue_fraction(img_bgr):
    """Return fraction of image pixels that fall in the SC blue HSV range."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    return mask.sum() / 255.0 / mask.size


def process_split(split, min_blue_pct, apply, debug):
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split

    if not img_dir.exists():
        print(f"[{split}] directory does not exist, skipping")
        return 0, 0

    imgs = sorted(img_dir.glob("*.png"))
    kept = 0
    deleted = 0
    blue_pcts = []

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] could not read {img_path.name}")
            continue

        pct = blue_fraction(img)
        blue_pcts.append(pct)

        if pct < min_blue_pct:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if apply:
                img_path.unlink()
                if lbl_path.exists():
                    lbl_path.unlink()
            deleted += 1
            if debug:
                print(f"  DELETE {img_path.name}  blue={pct*100:.3f}%")
        else:
            kept += 1

    if blue_pcts:
        arr = np.array(blue_pcts)
        print(f"[{split}] blue% stats: "
              f"min={arr.min()*100:.3f}  max={arr.max()*100:.3f}  "
              f"mean={arr.mean()*100:.3f}  median={np.median(arr)*100:.3f}")
    print(f"[{split}] kept={kept}  deleted={deleted}  total={len(imgs)}")
    return kept, deleted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-blue-pct", type=float, default=0.001,
                   help="min fraction of blue pixels to keep image (default 0.001 = 0.1%%)")
    p.add_argument("--apply", action="store_true",
                   help="actually delete files (default is dry run)")
    p.add_argument("--debug", action="store_true",
                   help="print every deleted file with its blue%%")
    args = p.parse_args()

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== {mode} | threshold: {args.min_blue_pct*100:.3f}% blue pixels ===")
    print(f"=== Data root: {ROOT} ===\n")

    total_kept = 0
    total_deleted = 0
    for split in ["train", "val"]:
        k, d = process_split(split, args.min_blue_pct, args.apply, args.debug)
        total_kept += k
        total_deleted += d
        print()

    print(f"=== TOTAL: kept={total_kept}  deleted={total_deleted} ===")
    if not args.apply:
        print("DRY RUN — rerun with --apply to actually delete")


if __name__ == "__main__":
    main()