#!/usr/bin/env python3
"""Train the separate SFP plug-pose network locally, preferring Apple MPS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time


def _parse_args():
    repo_root = Path(__file__).resolve().parent
    dataset_root = Path.home() / "aic_perception_data" / "sfp_plug_pose"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=dataset_root / "aic_sfp_plug_pose.yaml",
    )
    parser.add_argument("--model", default="yolo11s-pose.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto prefers mps, then CUDA, then CPU; pass mps/cpu/0 explicitly",
    )
    parser.add_argument("--project", type=Path, default=dataset_root / "runs")
    parser.add_argument("--name", default="sfp_plug_pose_mps")
    parser.add_argument(
        "--final-weights",
        type=Path,
        default=(
            repo_root
            / "aic_example_policies"
            / "aic_example_policies"
            / "ros"
            / "weights"
            / "best_sfp_plug_pose.pt"
        ),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=dataset_root / "reports" / "train_sfp_plug_pose.json",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _choose_device(requested: str) -> tuple[str, str]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    if requested != "auto":
        return requested, torch.__version__
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.__version__
    if torch.cuda.is_available():
        return "0", torch.__version__
    return "cpu", torch.__version__


def _dataset_counts(data_yaml: Path) -> dict[str, int]:
    root = data_yaml.parent
    counts = {}
    for split in ("train", "val", "test"):
        images = root / "images" / split
        labels = root / "labels" / split
        image_paths = sorted(images.glob("*.png")) if images.is_dir() else []
        counts[f"{split}_images"] = len(image_paths)
        counts[f"{split}_labels"] = sum(
            (labels / f"{path.stem}.txt").is_file() for path in image_paths
        )
    return counts


def main():
    args = _parse_args()
    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")
    counts = _dataset_counts(data_yaml)
    if counts["train_images"] == 0 or counts["val_images"] == 0:
        raise RuntimeError(f"dataset needs nonempty train and val splits: {counts}")
    if counts["train_labels"] != counts["train_images"]:
        raise RuntimeError(f"one or more train images lack labels: {counts}")

    # In this conda/PyPI Mac environment, importing the PyTorch wheel before
    # OpenCV can initialize two libomp copies.  Ultralytics loads OpenCV first,
    # which selects one runtime cleanly; do not use KMP_DUPLICATE_LIB_OK.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from ultralytics import YOLO
    device, torch_version = _choose_device(args.device)

    args.project = args.project.expanduser().resolve()
    args.final_weights = args.final_weights.expanduser().resolve()
    args.metrics_out = args.metrics_out.expanduser().resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    args.final_weights.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {data_yaml}")
    print(f"Counts: {counts}")
    print(f"Training device: {device} (torch {torch_version})")

    model = YOLO(args.model)
    started = time.monotonic()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        seed=args.seed,
        deterministic=True,
        cache="disk",
        amp=device not in ("mps", "cpu"),
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        patience=args.patience,
        close_mosaic=15,
        box=8.0,
        pose=14.0,
        kobj=2.0,
        cls=0.5,
        dfl=2.0,
        hsv_h=0.02,
        hsv_s=0.65,
        hsv_v=0.45,
        degrees=12.0,
        translate=0.10,
        scale=0.35,
        shear=2.0,
        perspective=0.0002,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.20,
        plots=True,
        verbose=True,
    )
    elapsed = time.monotonic() - started
    run_dir = args.project / args.name
    best_source = run_dir / "weights" / "best.pt"
    if not best_source.is_file():
        raise FileNotFoundError(f"training completed without best.pt: {best_source}")
    shutil.copy2(best_source, args.final_weights)

    metrics = {}
    if hasattr(results, "results_dict"):
        metrics = {key: float(value) for key, value in results.results_dict.items()}
    report = {
        "data": str(data_yaml),
        "dataset_counts": counts,
        "initial_model": args.model,
        "best_source": str(best_source),
        "final_weights": str(args.final_weights),
        "device": device,
        "torch_version": torch_version,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }
    args.metrics_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
