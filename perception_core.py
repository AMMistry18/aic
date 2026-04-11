"""
perception_core.py
Standalone perception helpers for AIC port detection + triangulation.
No ROS dependencies — can be imported and tested from plain Python.

Usage as library:
    from perception_core import PerceptionCore
    pc = PerceptionCore(nic_weights='path/to/best.pt')
    nics = pc.detect_nic(bgr_image)       # list of {kps, bbox, conf}
    scs = pc.detect_sc(bgr_image)         # list of {centroid, bbox, area}
    xyz = pc.triangulate([p1, p2, p3], [P1, P2, P3])  # 3D point

Usage as CLI sanity check:
    python perception_core.py --image path/to/img.png --kind sc
    python perception_core.py --image path/to/img.png --kind nic --weights best.pt
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# Default SC port blue HSV range
SC_BLUE_LOWER = np.array([90, 80, 60], dtype=np.uint8)
SC_BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)

SC_MIN_AREA = 15
SC_MAX_AREA = 50000

# NIC port keypoint layout from DataCollectorPose2: 8 kps = 4 corners Port0 + 4 corners Port1
NIC_KPS_PORT0 = slice(0, 4)
NIC_KPS_PORT1 = slice(4, 8)


class PerceptionCore:
    def __init__(self, nic_weights: str | None = None):
        self._yolo = None
        self._nic_weights = nic_weights

    def _load_yolo(self):
        if self._yolo is None:
            if self._nic_weights is None:
                raise RuntimeError("NIC weights path not provided")
            from ultralytics import YOLO
            self._yolo = YOLO(self._nic_weights)
        return self._yolo

    # ─── SC port detection via HSV blob ────────────────────────────────────

    def detect_sc(self, bgr: np.ndarray) -> list[dict]:
        """
        Detect SC ports via HSV blue blob. Returns list of:
            {centroid: (cx, cy), bbox: (x, y, w, h), area: int}
        Sorted by area descending.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SC_BLUE_LOWER, SC_BLUE_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < SC_MIN_AREA or area > SC_MAX_AREA:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]
            out.append({
                "centroid": (float(cx), float(cy)),
                "bbox": (x, y, w, h),
                "area": area,
            })
        out.sort(key=lambda d: -d["area"])
        return out

    # ─── NIC card detection via YOLO ───────────────────────────────────────

    def detect_nic(self, bgr: np.ndarray, conf_thresh: float = 0.3) -> list[dict]:
        """
        Run YOLO-pose on the image. Returns list of:
            {kps: np.ndarray[8,2], bbox: (x1,y1,x2,y2), conf: float}
        Sorted by confidence descending.
        """
        model = self._load_yolo()
        r = model(bgr, verbose=False, conf=conf_thresh)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kps_all = r.keypoints.xy.cpu().numpy()  # (N, 8, 2)
        for i in range(len(boxes)):
            out.append({
                "kps": kps_all[i],
                "bbox": tuple(boxes[i].tolist()),
                "conf": float(confs[i]),
            })
        out.sort(key=lambda d: -d["conf"])
        return out

    # ─── Linear DLT triangulation ──────────────────────────────────────────

    @staticmethod
    def triangulate(points_2d: list[tuple[float, float]],
                    proj_mats: list[np.ndarray]) -> np.ndarray:
        """
        Linear DLT triangulation from N views of the same 3D point.

        points_2d: list of (u, v) pixel coordinates, one per view
        proj_mats: list of 3x4 projection matrices P = K @ [R|t] where
                   [R|t] transforms world -> camera_optical

        Returns: (3,) numpy array — the 3D point in world (base_link) frame.
        """
        if len(points_2d) != len(proj_mats):
            raise ValueError("points_2d and proj_mats length mismatch")
        if len(points_2d) < 2:
            raise ValueError("need at least 2 views to triangulate")

        A = []
        for (u, v), P in zip(points_2d, proj_mats):
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X = X_h[:3] / X_h[3]
        return X

    @staticmethod
    def build_projection_matrix(K: np.ndarray, T_cam_from_world: np.ndarray) -> np.ndarray:
        """
        Build a 3x4 projection matrix.

        K: 3x3 camera intrinsics
        T_cam_from_world: 4x4 transform such that X_cam = T_cam_from_world @ X_world
                          (i.e. this is base_link -> camera_optical inverse direction)

        Returns: 3x4 projection matrix P
        """
        Rt = T_cam_from_world[:3, :4]
        return K @ Rt

    @staticmethod
    def invert_transform(T: np.ndarray) -> np.ndarray:
        """Invert a 4x4 rigid transform efficiently."""
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv


# ─── Visualization helpers ────────────────────────────────────────────────

def draw_sc(bgr, detections):
    out = bgr.copy()
    for d in detections:
        cx, cy = d["centroid"]
        x, y, w, h = d["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(out, f"a={d['area']}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def draw_nic(bgr, detections):
    out = bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i, (kx, ky) in enumerate(d["kps"]):
            color = (0, 0, 255) if i < 4 else (255, 0, 0)  # P0 red, P1 blue
            cv2.circle(out, (int(kx), int(ky)), 4, color, -1)
            cv2.putText(out, str(i), (int(kx) + 5, int(ky)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(out, f"{d['conf']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


# ─── CLI sanity check ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to image or directory")
    ap.add_argument("--kind", choices=["sc", "nic", "both"], default="sc")
    ap.add_argument("--weights", default=None, help="YOLO weights for NIC")
    ap.add_argument("--out", default="/tmp/perception_viz",
                    help="output dir for visualizations")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    pc = PerceptionCore(nic_weights=args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = Path(args.image)
    images = [p] if p.is_file() else sorted(p.glob("*.png"))[:args.limit]

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[SKIP] {img_path}")
            continue
        viz = bgr.copy()
        if args.kind in ("sc", "both"):
            sc = pc.detect_sc(bgr)
            viz = draw_sc(viz, sc)
            print(f"{img_path.name}: SC blobs={len(sc)} "
                  f"{[d['centroid'] for d in sc[:3]]}")
        if args.kind in ("nic", "both"):
            nic = pc.detect_nic(bgr)
            viz = draw_nic(viz, nic)
            top_conf = f"{nic[0]['conf']:.2f}" if nic else "none"
            print(f"{img_path.name}: NIC dets={len(nic)} top_conf={top_conf}")
        cv2.imwrite(str(out_dir / img_path.name), viz)

    # DLT self-test with synthetic data
    if args.kind == "sc":
        print("\n=== DLT triangulation self-test ===")
        X_true = np.array([0.5, 0.2, 1.0])
        K = np.array([[1236.63, 0, 576],
                      [0, 1236.63, 512],
                      [0, 0, 1]])
        # 3 synthetic cameras looking at X_true from different angles
        T1 = np.eye(4)
        T2 = np.eye(4); T2[0, 3] = -0.1
        T3 = np.eye(4); T3[0, 3] = 0.1
        Ps = [PerceptionCore.build_projection_matrix(K, T) for T in [T1, T2, T3]]
        pts = []
        for P in Ps:
            X_h = np.append(X_true, 1.0)
            x = P @ X_h
            pts.append((x[0] / x[2], x[1] / x[2]))
        X_est = PerceptionCore.triangulate(pts, Ps)
        err = np.linalg.norm(X_est - X_true)
        print(f"  true:  {X_true}")
        print(f"  est:   {X_est}")
        print(f"  error: {err:.6f} m  {'PASS' if err < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()