#!/usr/bin/env python3
"""
Train YOLO-pose model for SC port detection.
Matches hyperparameters from the NIC training recipe with tuning for SC's
smaller keypoint count (4 vs 8) and tiny port size.

Usage:
    pixi run python3 tools/perception/sc_port/train.py

Output:
    Weights: ~/bestSC.pt
    Run logs: ~/runs/aic_sc_robotic_precision/
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

HOME = Path(os.path.expanduser("~"))
REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(HOME / "aic_perception_data" / "pose_sc" / "aic_sc_pose.yaml"),
        help="SC pose dataset yaml",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--model", default="yolov8m-pose.pt")
    parser.add_argument("--project", default=str(HOME / "runs"))
    parser.add_argument("--name", default="aic_sc_robotic_precision")
    parser.add_argument("--final-weights", default=str(HOME / "bestSC.pt"))
    parser.add_argument(
        "--metrics-out",
        default=str(REPO_ROOT / "outputs" / "sc_pose_pipeline" / "train_metrics.json"),
        help="path to save summarized training metrics json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(args.data)
    run_root = Path(args.project)
    run_name = args.name
    final_weights = Path(args.final_weights)
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    assert data_yaml.exists(), f"Dataset YAML not found: {data_yaml}"
    print(f"Training on: {data_yaml}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,

        # Stability & speed
        imgsz=args.imgsz,
        batch=args.batch,
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
        project=str(run_root),
        name=run_name,
        exist_ok=True,
        device=args.device,
        verbose=True,
    )

    # Copy the best weights to ~/bestSC.pt for easy access
    best_src = run_root / run_name / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy(best_src, final_weights)
        print(f"\n✓ Best weights copied to {final_weights}")
    else:
        print(f"\n[WARN] best.pt not found at {best_src}")

    # Print final metrics
    print("\n=== Training complete ===")
    metrics = {}
    if hasattr(results, "results_dict"):
        metrics = {k: float(v) for k, v in results.results_dict.items()}
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": str(data_yaml),
                "weights": str(best_src if best_src.exists() else ""),
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"\nMetrics saved: {metrics_out}")


if __name__ == "__main__":
    main()
