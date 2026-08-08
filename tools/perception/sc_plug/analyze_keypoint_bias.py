"""Per-keypoint signed error: is the pose head biased, and where?

Compares predicted keypoints against the GT labels the collector wrote, per
keypoint index.  A near-zero mean with large spread is noise; a consistent
non-zero mean is bias, which triangulation cannot average away.

Keypoints 0-3 are the near plane (visible ferrule end).  Keypoints 4-7 are the
rear plane, 12 mm back, which sits inside the gripper jaws -- the model has to
infer those from context rather than observe them.
"""
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

weights, data_root, split = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
img_dir = data_root / "images" / split
lbl_dir = data_root / "labels" / split
images = sorted(img_dir.glob("*.png"))
print(f"{len(images)} images in {split}")

model = YOLO(weights)
N = 8
dx = [[] for _ in range(N)]
dy = [[] for _ in range(N)]
matched = 0

for i in range(0, len(images), 16):
    batch = images[i : i + 16]
    results = model.predict(batch, imgsz=960, verbose=False, device=0)
    for img_path, res in zip(batch, results):
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists() or res.keypoints is None or len(res.boxes) == 0:
            continue
        h, w = res.orig_shape
        toks = lbl.read_text().split()
        gt = np.array(toks[5:], dtype=float).reshape(N, 3)
        gt_px = np.stack([gt[:, 0] * w, gt[:, 1] * h], axis=1)
        best = int(np.argmax(res.boxes.conf.cpu().numpy()))
        pr_px = res.keypoints.xy.cpu().numpy()[best]
        matched += 1
        for k in range(N):
            dx[k].append(pr_px[k, 0] - gt_px[k, 0])
            dy[k].append(pr_px[k, 1] - gt_px[k, 1])

print(f"matched {matched} images\n")
hdr = "%4s %5s %9s %9s %8s %8s %8s %7s" % (
    "kpt", "plane", "mean_dx", "mean_dy", "std_dx", "std_dy", "|bias|", "rms",
)
print(hdr)
rows = []
for k in range(N):
    a, b = np.array(dx[k]), np.array(dy[k])
    bias = float(np.hypot(a.mean(), b.mean()))
    rms = float(np.sqrt((a ** 2 + b ** 2).mean()))
    plane = "near" if k < 4 else "REAR"
    print(
        "%4d %5s %9.3f %9.3f %8.3f %8.3f %8.3f %7.3f"
        % (k, plane, a.mean(), b.mean(), a.std(), b.std(), bias, rms)
    )
    rows.append(
        dict(
            kpt=k,
            plane=plane,
            mean_dx=float(a.mean()),
            mean_dy=float(b.mean()),
            std_dx=float(a.std()),
            std_dy=float(b.std()),
            bias_px=bias,
            rms_px=rms,
        )
    )

near = float(np.mean([r["bias_px"] for r in rows[:4]]))
rear = float(np.mean([r["bias_px"] for r in rows[4:]]))
near_rms = float(np.mean([r["rms_px"] for r in rows[:4]]))
rear_rms = float(np.mean([r["rms_px"] for r in rows[4:]]))
print(f"\nmean |bias| near plane (0-3): {near:.3f} px   (rms {near_rms:.3f})")
print(f"mean |bias| REAR plane (4-7): {rear:.3f} px   (rms {rear_rms:.3f})  <- occluded")
if near > 0:
    print(f"rear/near bias ratio: {rear / near:.2f}x")

out = data_root.parent / "training" / "reports" / "keypoint_bias.json"
json.dump(rows, open(out, "w"), indent=2)
print(f"\nwrote {out}")
