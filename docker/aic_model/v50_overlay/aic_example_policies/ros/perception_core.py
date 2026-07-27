"""
perception_core.py
Standalone perception helpers for AIC port detection + triangulation.
No ROS dependencies — can be imported and tested from plain Python.

Usage as library:
    from perception_core import PerceptionCore
    pc = PerceptionCore(nic_weights='path/to/best.pt')
    nics = pc.detect_nic(bgr_image)       # list of {kps, bbox, conf, cls}
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
    def __init__(self, nic_weights: str | None = None, sc_weights: str | None = None):
        self._yolo = None
        self._sc_yolo = None
        self._nic_weights = nic_weights
        self._sc_weights = sc_weights

    def _load_yolo(self):
        if self._yolo is None:
            if self._nic_weights is None:
                raise RuntimeError("NIC weights path not provided")
            from ultralytics import YOLO
            self._yolo = YOLO(self._nic_weights)
        return self._yolo

    def _load_sc_yolo(self):
        if self._sc_yolo is None:
            if self._sc_weights is None:
                raise RuntimeError("SC weights path not provided")
            from ultralytics import YOLO
            self._sc_yolo = YOLO(self._sc_weights)
        return self._sc_yolo

    # ─── SC port detection via HSV blob ────────────────────────────────────

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        """Return corners in [top-left, top-right, bottom-right, bottom-left]."""
        if pts.shape != (4, 2):
            raise ValueError("expected 4x2 corner array")
        s = pts[:, 0] + pts[:, 1]
        d = pts[:, 0] - pts[:, 1]
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]  # top-left
        ordered[2] = pts[np.argmax(s)]  # bottom-right
        ordered[1] = pts[np.argmin(d)]  # top-right
        ordered[3] = pts[np.argmax(d)]  # bottom-left
        return ordered

    def detect_sc(self, bgr: np.ndarray) -> list[dict]:
        """
        Detect SC ports via HSV blue blob. Returns list of:
            {
              centroid: (cx, cy),
              bbox: (x, y, w, h),
              area: int,
              corners: [(x, y) * 4],  # ordered TL,TR,BR,BL from minAreaRect
              major_axis: ((x0, y0), (x1, y1)),  # long-axis endpoints in pixels
            }
        Sorted by area descending.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SC_BLUE_LOWER, SC_BLUE_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        out = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < SC_MIN_AREA or area > SC_MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            moms = cv2.moments(cnt)
            if abs(moms["m00"]) < 1e-6:
                continue
            cx = moms["m10"] / moms["m00"]
            cy = moms["m01"] / moms["m00"]

            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = rect
            if rw < 1.0 or rh < 1.0:
                continue
            corners = self._order_quad_points(cv2.boxPoints(rect).astype(np.float32))
            tl, tr, br, bl = corners
            if rw >= rh:
                major_axis = (
                    ((tl + bl) * 0.5).tolist(),
                    ((tr + br) * 0.5).tolist(),
                )
            else:
                major_axis = (
                    ((tl + tr) * 0.5).tolist(),
                    ((bl + br) * 0.5).tolist(),
                )

            out.append({
                "centroid": (float(cx), float(cy)),
                "bbox": (x, y, w, h),
                "area": area,
                "corners": [tuple(map(float, p)) for p in corners],
                "major_axis": (
                    (float(major_axis[0][0]), float(major_axis[0][1])),
                    (float(major_axis[1][0]), float(major_axis[1][1])),
                ),
            })
        out.sort(key=lambda d: -d["area"])
        return out

    @staticmethod
    def _sc_pose_record(
        xyxy: np.ndarray,
        conf: float,
        kps: np.ndarray | None,
    ) -> dict:
        """Build one public SC detection while retaining precise box pixels.

        ``_xyxy`` is intentionally private to this module: the public ``bbox``
        retains its historical integer ``(x, y, w, h)`` contract, while crop
        refinement needs the unrounded coarse box to choose and form a crop.
        """
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        det = {
            "centroid": ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
            "bbox": (
                int(round(x1)), int(round(y1)),
                int(round(x2 - x1)), int(round(y2 - y1)),
            ),
            "conf": float(conf),
        }
        if kps is not None:
            kps = np.asarray(kps, dtype=np.float32)
            det["kps"] = kps
            if kps.shape[0] >= 4:
                corners = PerceptionCore._order_quad_points(kps[:4, :2])
                tl, tr, br, bl = corners
                if np.linalg.norm(tr - tl) >= np.linalg.norm(bl - tl):
                    major_axis = (
                        ((tl + bl) * 0.5).tolist(),
                        ((tr + br) * 0.5).tolist(),
                    )
                else:
                    major_axis = (
                        ((tl + tr) * 0.5).tolist(),
                        ((bl + br) * 0.5).tolist(),
                    )
                det["corners"] = [tuple(map(float, p)) for p in corners]
                det["major_axis"] = (
                    (float(major_axis[0][0]), float(major_axis[0][1])),
                    (float(major_axis[1][0]), float(major_axis[1][1])),
                )
        return det

    @classmethod
    def _sc_pose_records_from_result(cls, result) -> list[dict]:
        """Decode a YOLO result into internal detections in that image's frame."""
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes_xyxy = np.asarray(result.boxes.xyxy.cpu().numpy(), dtype=np.float64)
        confs = np.asarray(result.boxes.conf.cpu().numpy(), dtype=np.float64)
        kps_all = (
            np.asarray(result.keypoints.xy.cpu().numpy(), dtype=np.float64)
            if result.keypoints is not None
            else None
        )
        count = min(len(boxes_xyxy), len(confs))
        records = []
        for index in range(count):
            xyxy = boxes_xyxy[index]
            if xyxy.shape != (4,) or not np.all(np.isfinite(xyxy)):
                continue
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            kps = kps_all[index] if kps_all is not None and index < len(kps_all) else None
            records.append({
                "xyxy": xyxy.copy(),
                "conf": float(confs[index]),
                "kps": None if kps is None else np.asarray(kps, dtype=np.float64).copy(),
            })
        return records

    @staticmethod
    def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        """IoU for two finite ``xyxy`` boxes in the same pixel frame."""
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 1e-9 else 0.0

    @classmethod
    def _refine_sc_pose_on_crops(
        cls,
        bgr: np.ndarray,
        coarse: list[dict],
        model,
        kwargs: dict,
        crop_pad_scale: float,
    ) -> list[dict]:
        """Refine every coarse port on a native-resolution crop.

        Each crop is tied to its originating full-frame box.  A second-pass
        detection must overlap that box (in crop pixels) *and* have a nearby
        centre before it may replace the coarse result.  This deliberately
        fails closed for crowded port images rather than letting a nearby,
        higher-confidence port change an identity before multiview matching.
        """
        height, width = bgr.shape[:2]
        crops: list[np.ndarray] = []
        metadata: list[tuple[int, int, int, np.ndarray]] = []
        for index, record in enumerate(coarse):
            box = np.asarray(record["xyxy"], dtype=np.float64)
            centre_x = 0.5 * (box[0] + box[2])
            centre_y = 0.5 * (box[1] + box[3])
            half = max(4.0, 0.5 * crop_pad_scale * max(box[2] - box[0], box[3] - box[1]))
            x0 = int(max(0, np.floor(centre_x - half)))
            y0 = int(max(0, np.floor(centre_y - half)))
            x1 = int(min(width, np.ceil(centre_x + half)))
            y1 = int(min(height, np.ceil(centre_y + half)))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crops.append(bgr[y0:y1, x0:x1])
            metadata.append((index, x0, y0, box - np.array([x0, y0, x0, y0])))

        if not crops:
            return coarse
        try:
            crop_results = model(crops, **kwargs)
        except Exception:
            # Refinement is optional accuracy work.  A crop batch failure must
            # not discard the already-valid full-frame detections.
            return coarse
        if len(crop_results) != len(crops):
            return coarse

        refined = list(coarse)
        for (index, x0, y0, coarse_local), result in zip(metadata, crop_results):
            candidates = cls._sc_pose_records_from_result(result)
            if not candidates:
                continue
            reference_centre = 0.5 * (coarse_local[:2] + coarse_local[2:])
            reference_diag = float(np.linalg.norm(coarse_local[2:] - coarse_local[:2]))
            valid = []
            for candidate in candidates:
                candidate_box = candidate["xyxy"]
                iou = cls._box_iou_xyxy(coarse_local, candidate_box)
                candidate_centre = 0.5 * (candidate_box[:2] + candidate_box[2:])
                centre_distance = float(np.linalg.norm(candidate_centre - reference_centre))
                # Both checks are intentional.  IoU alone can select a broad
                # neighbouring box; centre alone can select a tiny false pose.
                if iou < 0.10 or centre_distance > max(4.0, 0.5 * reference_diag):
                    continue
                valid.append((iou, -centre_distance, float(candidate["conf"]), candidate))
            if not valid:
                continue
            candidate = max(valid, key=lambda item: item[:3])[3]
            offset = np.array([x0, y0, x0, y0], dtype=np.float64)
            refined_record = dict(candidate)
            refined_record["xyxy"] = candidate["xyxy"] + offset
            if candidate["kps"] is not None:
                refined_record["kps"] = candidate["kps"] + np.array([x0, y0], dtype=np.float64)
            refined[index] = refined_record
        return refined

    def detect_sc_pose(
        self,
        bgr: np.ndarray,
        conf_thresh: float = 0.2,
        *,
        crop_refine: bool | None = None,
        crop_pad_scale: float | None = None,
    ) -> list[dict]:
        """
        Run YOLO pose model for SC ports.

        Returns list of:
            {
              centroid: (cx, cy),
              bbox: (x, y, w, h),
              conf: float,
              kps: np.ndarray[N,2],
              corners: [(x, y) * 4] when 4+ keypoints are available,
              major_axis: ((x0, y0), (x1, y1)) when 4+ keypoints are available
            }
        """
        model = self._load_sc_yolo()
        # The SC pose model was trained at imgsz=960 (train_sc.py). Without an
        # explicit imgsz, ultralytics letterboxes to its 640 default, silently
        # downscaling the 1152x1024 camera frames ~1.8x AND running the model
        # off its training scale. Match the training resolution here.
        imgsz = int(os.environ.get("AIC_SC_POSE_IMGSZ", "960"))
        kwargs = {"verbose": False, "conf": conf_thresh, "imgsz": imgsz}
        results = model(bgr, **kwargs)
        if not results:
            return []
        records = self._sc_pose_records_from_result(results[0])
        if not records:
            return []

        # TACC evaluation on TF-labelled native camera frames selected a broad
        # 24x context crop: it roughly halved median pose-centre error without
        # increasing misses.  Keep an explicit opt-out for field comparison.
        if crop_refine is None:
            crop_refine = os.environ.get("AIC_SC_POSE_CROP_REFINE", "1") == "1"
        if crop_refine:
            if crop_pad_scale is None:
                try:
                    crop_pad_scale = float(os.environ.get("AIC_SC_POSE_CROP_PAD", "24"))
                except ValueError:
                    crop_pad_scale = 24.0
            records = self._refine_sc_pose_on_crops(
                bgr, records, model, kwargs, max(1.1, float(crop_pad_scale))
            )

        out = [self._sc_pose_record(record["xyxy"], record["conf"], record["kps"])
               for record in records]
        out.sort(key=lambda det: -det["conf"])
        return out

    # ─── NIC card detection via YOLO ───────────────────────────────────────

    def detect_nic(self, bgr: np.ndarray, conf_thresh: float = 0.3) -> list[dict]:
        """
        Run YOLO-pose on the image. Returns list of:
            {kps: np.ndarray[8,2], bbox: (x1,y1,x2,y2), conf: float, cls: int, kp_conf: np.ndarray[8]}
        Sorted by confidence descending.
        """
        model = self._load_yolo()
        # The NIC pose model shares the same training recipe as the SC pose
        # model (train_sc.py notes it matches the NIC recipe); match imgsz=960
        # here too, per docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md Tier-0.
        r = model(bgr, imgsz=960, verbose=False, conf=conf_thresh)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        kps_all = r.keypoints.xy.cpu().numpy()  # (N, 8, 2)
        kp_confs_all = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        for i in range(len(boxes)):
            kp_conf = np.asarray(kp_confs_all[i], dtype=np.float32) if kp_confs_all is not None else np.ones(8, dtype=np.float32)
            out.append({
                "kps": kps_all[i],
                "bbox": tuple(boxes[i].tolist()),
                "conf": float(confs[i]),
                "cls": int(classes[i]),
                "kp_conf": kp_conf,
            })
        out.sort(key=lambda d: -d["conf"])
        return out

    @staticmethod
    def nic_sfp_yaw_delta_world_z_from_triangulated_kps_board_invariant(
        kp_3d_8: np.ndarray,
        sfp_port_0: bool,
        min_edge_xy_m: float = 2e-4,
        min_port_sep_xy_m: float = 2e-3,
    ) -> float | None:
        """
        Yaw-only (about +base_link Z) from triangulated YOLO keypoints only — no TF.

        Reference axis is perpendicular (in XY) to the line between triangulated
        port-0 and port-1 centers on the same NIC. That rotates with the task board,
        so pure board yaw does not inject a false correction (unlike a fixed world-XY ref).

        Top edge in label order: port0 → (KP1−KP0); port1 → (KP5−KP4) (DataCollectorPose2).
        """
        if kp_3d_8.shape != (8, 3):
            raise ValueError("kp_3d_8 must be (8, 3)")
        if not np.all(np.isfinite(kp_3d_8)):
            return None
        c0 = kp_3d_8[0:4].mean(axis=0)
        c1 = kp_3d_8[4:8].mean(axis=0)
        v_lat = (c1 - c0)[:2].astype(np.float64)
        nlat = float(np.linalg.norm(v_lat))
        if nlat < min_port_sep_xy_m:
            return None
        v_lat /= nlat
        ref = np.array([-v_lat[1], v_lat[0]], dtype=np.float64)
        ref /= max(float(np.linalg.norm(ref)), 1e-9)
        if sfp_port_0:
            d = kp_3d_8[1] - kp_3d_8[0]
        else:
            d = kp_3d_8[5] - kp_3d_8[4]
        d_xy = np.asarray(d[:2], dtype=np.float64).reshape(2)
        nd = float(np.linalg.norm(d_xy))
        if nd < min_edge_xy_m:
            return None
        d_xy /= nd
        return float(np.arctan2(
            ref[0] * d_xy[1] - ref[1] * d_xy[0],
            ref[0] * d_xy[0] + ref[1] * d_xy[1],
        ))

    @staticmethod
    def nic_sfp_yaw_world_z_from_triangulated_kps_absolute(
        kp_3d_8: np.ndarray,
        sfp_port_0: bool,
        min_edge_xy_m: float = 2e-4,
    ) -> float | None:
        """
        Absolute world yaw (about +base_link Z) from triangulated YOLO keypoints only.

        Uses top-edge direction in label order:
          - port0: KP1 - KP0
          - port1: KP5 - KP4

        Returns atan2(dy, dx) in radians, wrapped by caller as needed.
        """
        if kp_3d_8.shape != (8, 3):
            raise ValueError("kp_3d_8 must be (8, 3)")
        if not np.all(np.isfinite(kp_3d_8)):
            return None
        d = (kp_3d_8[1] - kp_3d_8[0]) if sfp_port_0 else (kp_3d_8[5] - kp_3d_8[4])
        d_xy = np.asarray(d[:2], dtype=np.float64).reshape(2)
        nd = float(np.linalg.norm(d_xy))
        if nd < min_edge_xy_m:
            return None
        d_xy /= nd
        return float(np.arctan2(d_xy[1], d_xy[0]))

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
        if "corners" in d:
            pts = np.array(d["corners"], dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        if "major_axis" in d:
            (x0, y0), (x1, y1) = d["major_axis"]
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 2)
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
