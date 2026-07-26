#!/usr/bin/env python3
"""Quantify per-camera SFP detection asymmetry, offline or from field logs.

Two independent modes, either usable without the other:

  --images DIR   run the port and plug pose models over a dump of preview
                  images and compare detection quality across cameras.
  --log FILE     parse PLUG_POSE_INPUT / PORT_POSE_INPUT lines out of a
                  service log and build the same per-camera table from a
                  real run instead of an offline re-inference.

Kept dependency-light: numpy + ultralytics only, and ultralytics is imported
lazily inside --images mode so --log mode works without it installed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_PORT_WEIGHTS = Path(
    "/home/Anshul/AIC_Phase_1/aic_0/aic/aic_example_policies/aic_example_policies/"
    "ros/weights/best.pt"
)
DEFAULT_PLUG_WEIGHTS = Path(
    "/home/Anshul/AIC_Phase_1/aic_0/aic/aic_example_policies/aic_example_policies/"
    "ros/weights/best_sfp_plug_pose.pt"
)

_CAMERA_RE = re.compile(r"(center|left|right)_camera")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--images", type=Path, help="directory of preview images to re-run")
    mode.add_argument("--log", type=Path, help="service log to parse PLUG/PORT_POSE_INPUT lines from")
    parser.add_argument("--port-weights", type=Path, default=DEFAULT_PORT_WEIGHTS)
    parser.add_argument("--plug-weights", type=Path, default=DEFAULT_PLUG_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _camera_from_filename(path: Path) -> str | None:
    m = _CAMERA_RE.search(path.name)
    return f"{m.group(1)}_camera" if m else None


def _collect_images(root: Path) -> dict[str, list[Path]]:
    """Recursively bucket every image under root by camera identity."""
    by_camera: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in _IMAGE_EXTS or not path.is_file():
            continue
        camera = _camera_from_filename(path)
        if camera is None:
            continue
        by_camera.setdefault(camera, []).append(path)
    return by_camera


def _pairwise_spread(xy: np.ndarray) -> float:
    """Mean pairwise keypoint distance -- a quad-shrink proxy: a squashed quad
    (camera looking down its own axis, or clipped by the frame edge) collapses
    this toward zero relative to a well-posed view of the same part."""
    if xy.shape[0] < 2:
        return float("nan")
    diffs = xy[:, None, :] - xy[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(xy.shape[0], k=1)
    return float(np.mean(dists[iu])) if iu[0].size else float("nan")


def _run_model_over_images(model, images: list[Path], imgsz: int) -> dict:
    """Per-camera-batch stats for one YOLO pose model over one image set."""
    box_confs = []
    kp_confs_by_index: dict[int, list[float]] = {}
    spreads = []
    n_detected = 0
    for path in images:
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        result = model(bgr, imgsz=imgsz, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        confs = result.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        box_confs.append(float(confs[best]))
        n_detected += 1
        if result.keypoints is not None:
            xy = result.keypoints.xy.cpu().numpy()[best]
            spreads.append(_pairwise_spread(xy))
            if result.keypoints.conf is not None:
                kp_conf = result.keypoints.conf.cpu().numpy()[best]
                for idx, c in enumerate(kp_conf):
                    kp_confs_by_index.setdefault(idx, []).append(float(c))

    n_images = len(images)
    stats = {
        "n_images": n_images,
        "n_detected": n_detected,
        "detection_rate": (n_detected / n_images) if n_images else float("nan"),
        "box_conf_mean": float(np.mean(box_confs)) if box_confs else float("nan"),
        "box_conf_median": float(np.median(box_confs)) if box_confs else float("nan"),
        "box_conf_p10": float(np.percentile(box_confs, 10)) if box_confs else float("nan"),
        "keypoint_conf_by_index": {
            idx: float(np.mean(vals)) for idx, vals in sorted(kp_confs_by_index.items())
        },
        "mean_keypoint_spread": float(np.mean(spreads)) if spreads else float("nan"),
    }
    return stats


def _run_images_mode(args) -> dict:
    from ultralytics import YOLO

    by_camera = _collect_images(args.images)
    if not by_camera:
        print(f"[asymmetry] no camera-tagged images found under {args.images}")

    models = {}
    for label, weights in (("port", args.port_weights), ("plug", args.plug_weights)):
        if not weights.is_file():
            print(f"[asymmetry] {label} weights not found at {weights}; skipping")
            continue
        models[label] = YOLO(str(weights))

    result = {}
    for camera, images in sorted(by_camera.items()):
        result[camera] = {}
        for label, model in models.items():
            result[camera][label] = _run_model_over_images(model, images, args.imgsz)
    return result


def _parse_log_mode(log_path: Path) -> dict:
    """PLUG_POSE_INPUT camera=... box_conf=... usable_kp=a/b kp_conf=[...]
    PORT_POSE_INPUT camera=... box_conf=... kp_conf_mean=... kp_conf_min=..."""
    plug_re = re.compile(
        r"PLUG_POSE_INPUT camera=(\S+).*?box_conf=([\d.]+).*?usable_kp=(\d+)/(\d+)"
        r".*?kp_conf=\[([^\]]*)\]"
    )
    port_re = re.compile(
        r"PORT_POSE_INPUT camera=(\S+).*?box_conf=([\d.]+).*?"
        r"kp_conf_mean=([\d.]+).*?kp_conf_min=([\d.]+)"
    )

    by_camera: dict[str, dict[str, list]] = {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = plug_re.search(line)
        if m:
            camera, box_conf, usable, total, kp_conf_raw = m.groups()
            bucket = by_camera.setdefault(camera, {"plug": {"box_conf": [], "usable_frac": [], "kp_conf": []}})
            bucket.setdefault("plug", {"box_conf": [], "usable_frac": [], "kp_conf": []})
            bucket["plug"]["box_conf"].append(float(box_conf))
            bucket["plug"]["usable_frac"].append(float(usable) / float(total))
            kp_vals = [float(v) for v in kp_conf_raw.replace(",", " ").split()]
            bucket["plug"]["kp_conf"].append(kp_vals)
            continue
        m = port_re.search(line)
        if m:
            camera, box_conf, kp_mean, kp_min = m.groups()
            bucket = by_camera.setdefault(camera, {})
            bucket.setdefault("port", {"box_conf": [], "kp_conf_mean": [], "kp_conf_min": []})
            bucket["port"]["box_conf"].append(float(box_conf))
            bucket["port"]["kp_conf_mean"].append(float(kp_mean))
            bucket["port"]["kp_conf_min"].append(float(kp_min))

    result = {}
    for camera, models in sorted(by_camera.items()):
        result[camera] = {}
        for label, fields in models.items():
            entry = {"n_images": len(fields["box_conf"]),
                     "detection_rate": 1.0,  # a logged line implies a detection happened
                     "box_conf_mean": float(np.mean(fields["box_conf"])),
                     "box_conf_median": float(np.median(fields["box_conf"])),
                     "box_conf_p10": float(np.percentile(fields["box_conf"], 10))}
            if label == "plug":
                all_kp = [row for row in fields["kp_conf"] if row]
                by_index: dict[int, list[float]] = {}
                for row in all_kp:
                    for idx, v in enumerate(row):
                        by_index.setdefault(idx, []).append(v)
                entry["keypoint_conf_by_index"] = {
                    idx: float(np.mean(vals)) for idx, vals in sorted(by_index.items())
                }
            else:
                entry["kp_conf_mean"] = float(np.mean(fields["kp_conf_mean"]))
                entry["kp_conf_min"] = float(np.mean(fields["kp_conf_min"]))
            result[camera][label] = entry
    return result


def _print_table(result: dict):
    for camera, models in result.items():
        print(f"\n=== {camera} ===")
        for label, stats in models.items():
            print(f"  [{label}]")
            for key, value in stats.items():
                print(f"    {key}: {value}")


def main():
    args = _parse_args()

    if args.images is not None:
        result = _run_images_mode(args)
    else:
        result = _parse_log_mode(args.log)

    _print_table(result)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[asymmetry] wrote {args.json}")


if __name__ == "__main__":
    main()
