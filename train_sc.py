#!/usr/bin/env python3
"""
Train YOLO-pose model for SC port detection.
Matches hyperparameters from the NIC training recipe with tuning for SC's
smaller keypoint count (4 vs 8) and tiny port size.

Usage:
    pixi run python3 ~/train_sc.py

Output:
    Weights: ~/bestSC.pt
    Run logs: ~/runs/aic_sc_robotic_precision/
"""
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

HOME = Path(os.path.expanduser("~"))
DATA_YAML = HOME / "aic_perception_data" / "pose_sc" / "aic_sc_pose.yaml"
RUN_ROOT = HOME / "runs"
RUN_NAME = "aic_sc_robotic_precision"
FINAL_WEIGHTS = HOME / "bestSC.pt"


def main():
    assert DATA_YAML.exists(), f"Dataset YAML not found: {DATA_YAML}"
    print(f"Training on: {DATA_YAML}")

    model = YOLO("yolov8m-pose.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=150,

        # Stability & speed
        imgsz=960,
        batch=16,
        multi_scale=False,
        cache=True,

        # Precision weights — same as NIC recipe
        box=10.0,
        pose=12.0,
        kobj=2.0,
        dfl=3.0,
        cls=0.5,

        # HSV color augmentation (Option B — invariance without hardcoding filter)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Keep geometric augmentations conservative since ports are tiny
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # Schedule
        patience=50,
        cos_lr=True,
        close_mosaic=20,

        # Output
        project=str(RUN_ROOT),
        name=RUN_NAME,
        exist_ok=True,
        device=0,
        verbose=True,
    )

    # Copy the best weights to ~/bestSC.pt for easy access
    best_src = RUN_ROOT / RUN_NAME / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy(best_src, FINAL_WEIGHTS)
        print(f"\n✓ Best weights copied to {FINAL_WEIGHTS}")
    else:
        print(f"\n[WARN] best.pt not found at {best_src}")

    # Print final metrics
    print("\n=== Training complete ===")
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()