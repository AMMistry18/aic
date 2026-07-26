#!/usr/bin/env python3
"""Harvest the sim's full TF inventory without running an insertion.

_probe_tf_frames_for_tip() only ever runs mid-grasp, so a frame that only
matters for calibration planning -- "is there a port frame at all", "which
frames actually move" -- has never been visible on its own.  This script
attaches a bare TF listener to a live sim, lets the buffer fill, and prints
every frame it knows about: distance from the TCP, position, and how far it
moved between two samples a few seconds apart.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Allow the checked-out script to run before the ROS package is rebuilt.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_AIC_MODEL_SOURCE = _REPO_ROOT / "aic_model"
if str(_AIC_MODEL_SOURCE) not in sys.path:
    sys.path.insert(0, str(_AIC_MODEL_SOURCE))

try:
    from aic_model.sc_controller import parse_tf_frame_names
except Exception:
    # Mirrors sc_controller.parse_tf_frame_names: accept either tf2 dump
    # format ("all_frames_as_yaml()" puts each frame at column 0;
    # "all_frames_as_string()" emits "Frame X exists with parent Y.").
    def parse_tf_frame_names(text: str) -> list[str]:
        names = {line.split(":", 1)[0].strip()
                 for line in text.splitlines()
                 if ":" in line and line[:1] not in ("", " ", "\t", "-")}
        names |= set(re.findall(r"Frame (\S+?) exists", text))
        return sorted(n for n in names if n)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--tcp-frame", default="arm_link_tip")
    parser.add_argument("--settle", type=float, default=5.0,
                         help="seconds to let the TF buffer fill before the first sample")
    parser.add_argument("--gap", type=float, default=3.0,
                         help="seconds between the two samples used to detect motion")
    parser.add_argument("--json", type=Path, default=None,
                         help="dump the full result to this path for offline diffing")
    return parser.parse_args()


def _all_frame_names(buffer) -> list[str]:
    text = ""
    for method in ("all_frames_as_yaml", "all_frames_as_string"):
        fn = getattr(buffer, method, None)
        if fn is None:
            continue
        try:
            text = str(fn() or "")
        except Exception:
            continue
        if text.strip():
            break
    return parse_tf_frame_names(text)


def _sample(buffer, names, base_frame, tcp_frame):
    """One pass: resolve every frame against base_frame and, if present, tcp_frame."""
    import rclpy.time
    import rclpy.duration

    now = rclpy.time.Time()
    timeout = rclpy.duration.Duration(seconds=0.05)
    tcp_pos = None
    if tcp_frame in names:
        try:
            tf = buffer.lookup_transform(base_frame, tcp_frame, now, timeout)
            tr = tf.transform.translation
            tcp_pos = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        except Exception:
            tcp_pos = None

    frames = {}
    for frame in names:
        try:
            tf = buffer.lookup_transform(base_frame, frame, now, timeout)
        except Exception:
            continue
        tr = tf.transform.translation
        pos = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        frames[frame] = pos
    return frames, tcp_pos


def main():
    args = _parse_args()

    import rclpy
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener

    rclpy.init()
    node = rclpy.create_node("tf_frame_enumerator")
    buffer = Buffer()
    listener = TransformListener(buffer=buffer, node=node, spin_thread=True)

    print(f"[enumerate-tf] settling {args.settle:.1f}s for the TF buffer to fill...")
    time.sleep(args.settle)

    names = _all_frame_names(buffer)
    if not names:
        print("[enumerate-tf] TF buffer reported no frames at all", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
    print(f"[enumerate-tf] buffer knows {len(names)} frames")

    frames_a, tcp_pos = _sample(buffer, names, args.base_frame, args.tcp_frame)
    has_tcp = tcp_pos is not None
    if not has_tcp:
        print(f"[enumerate-tf] TCP frame {args.tcp_frame!r} not resolvable; "
              f"degrading to base-only table (distances are from {args.base_frame!r})")
        tcp_pos = np.zeros(3, dtype=np.float64)

    print(f"[enumerate-tf] waiting {args.gap:.1f}s before the second sample...")
    time.sleep(args.gap)
    frames_b, tcp_pos_b = _sample(buffer, names, args.base_frame, args.tcp_frame)
    if has_tcp and tcp_pos_b is not None:
        tcp_pos = tcp_pos_b

    rows = []
    for frame, pos_a in frames_a.items():
        pos_b = frames_b.get(frame)
        moved_mm = float(np.linalg.norm(pos_b - pos_a) * 1000.0) if pos_b is not None else float("nan")
        rows.append({
            "frame": frame,
            "dist_mm": float(np.linalg.norm(pos_a - tcp_pos) * 1000.0),
            "pos": pos_a.tolist(),
            "moved_mm": moved_mm,
        })
    rows.sort(key=lambda r: r["dist_mm"])

    dist_label = "TCP" if has_tcp else args.base_frame
    print(f"\n=== ALL FRAMES BY DISTANCE FROM {dist_label} ({len(rows)} resolved / {len(names)} total) ===")
    for r in rows:
        print(f"  {r['dist_mm']:8.1f}mm  moved={r['moved_mm']:6.2f}mm  {r['frame']}  "
              f"pos={np.round(r['pos'], 4).tolist()}")

    truth_rows = [r for r in rows
                  if any(k in r["frame"].lower() for k in ("port", "sfp", "sc"))]
    print(f"\n=== TRUTH-FRAME HARVEST (name matches port/sfp/sc, {len(truth_rows)}) ===")
    for r in truth_rows:
        print(f"  {r['dist_mm']:8.1f}mm  moved={r['moved_mm']:6.2f}mm  {r['frame']}")

    band_rows = [r for r in rows if 20.0 <= r["dist_mm"] <= 120.0]
    print(f"\n=== HELD-PLUG BAND (20-120mm from {dist_label}, {len(band_rows)}) ===")
    for r in band_rows:
        print(f"  {r['dist_mm']:8.1f}mm  moved={r['moved_mm']:6.2f}mm  {r['frame']}")

    if args.json is not None:
        result = {
            "base_frame": args.base_frame,
            "tcp_frame": args.tcp_frame,
            "has_tcp": has_tcp,
            "settle_s": args.settle,
            "gap_s": args.gap,
            "all_names": names,
            "frames": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[enumerate-tf] wrote {args.json}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
