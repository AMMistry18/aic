"""Development-only equivariance audit of the ORIGINAL frozen plug networks.

Rotate pixels, invert the pixel transform without relabeling landmarks, and fit
all camera-pair hypotheses. Simulator truth is used only after fits are saved.
This is not a production selector or a robustness benchmark.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'aic_model'))
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import SfpPlugPoseEstimator, PlugPoseView, PlugKeypointDetection, fuse_multiview_keypoints


def inverse_quarter_turn(points, turns, height, width):
    p = np.asarray(points).copy()
    x, y = p[:, 0].copy(), p[:, 1].copy()
    if turns == 1:
        p[:, 0], p[:, 1] = width - 1 - y, x
    elif turns == 2:
        p[:, 0], p[:, 1] = width - 1 - x, height - 1 - y
    elif turns == 3:
        p[:, 0], p[:, 1] = y, height - 1 - x
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--connector', choices=['sc', 'sfp'], required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    audit = json.loads(args.audit.read_text())
    views = [PlugPoseView(r['camera'], cv2.imread(str(args.audit.parent / f'{r["camera"]}_raw.png')),
                         np.array(r['K']), np.linalg.inv(r['T_camera_from_base_exact']), r['stamp_s'], r['camera'])
             for r in audit['views']]
    cls = ScPlugPoseEstimator if args.connector == 'sc' else SfpPlugPoseEstimator
    est = cls(ROOT / f'aic_example_policies/aic_example_policies/ros/weights/best_{args.connector}_plug_pose.pt',
              device='cpu', crop_refine=False, refinement_strategy='fixed')
    alternatives = {v.camera_name: [] for v in views}
    start = time.monotonic()
    for turns, crop in itertools.product(range(4), [False, True]):
        rotated = [PlugPoseView(v.camera_name, np.ascontiguousarray(np.rot90(v.image_bgr, turns)),
                               v.K, v.T_world_from_camera, v.stamp_s, v.frame_id) for v in views]
        detections = est._predict_fixed(rotated, crop_refine=crop)
        for det in detections:
            view = next(v for v in views if v.camera_name == det.camera_name)
            xy = inverse_quarter_turn(det.keypoints_px, turns, *view.image_bgr.shape[:2])
            alternatives[det.camera_name].append({'turns': turns, 'crop': crop, 'xy': xy.tolist(),
                'conf': det.keypoint_confidences.tolist(), 'box_conf': det.box_confidence})
    fits = []
    for names in itertools.combinations(alternatives, 2):
        for rows in itertools.product(*(alternatives[n] for n in names)):
            ds = [PlugKeypointDetection(n, np.array(r['xy']), np.array(r['conf']), r['box_conf']) for n, r in zip(names, rows)]
            try:
                p, R, shape, reproj, count, confidence = fuse_multiview_keypoints(views, ds, local_keypoints_m=est.local_keypoints_m)
                fits.append({'cameras': names, 'conditions': [{k: r[k] for k in ['turns', 'crop']} for r in rows],
                             'position': p.tolist(), 'rotation': R.tolist(), 'shape_mm': shape*1000,
                             'reprojection_px': reproj, 'passes_geometry': shape <= .0035 and reproj <= 6})
            except ValueError:
                pass
    result = {'connector': args.connector, 'GT_used_in_fitting': False, 'wall_s': time.monotonic()-start,
              'alternatives': alternatives, 'fits': sorted(fits, key=lambda r: r['reprojection_px'])}
    (args.out / 'predictions_without_truth.json').write_text(json.dumps(result, indent=2))
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler('z', -np.pi).as_matrix()
    base[:3, 3] = [-.2, .2, 0]
    truth = np.linalg.inv(base) @ np.array(audit['views'][0]['tip_world'])
    for row in result['fits']:
        row['diagnostic_tip_error_mm'] = float(np.linalg.norm(np.array(row['position'])-truth[:3, 3])*1000)
        row['diagnostic_angle_error_deg'] = float(np.rad2deg(Rotation.from_matrix(truth[:3, :3].T @ np.array(row['rotation'])).magnitude()))
    (args.out / 'diagnostic_scores.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'count': len(fits), 'geometry_pass_count': sum(r['passes_geometry'] for r in fits),
                      'best_reprojection': result['fits'][:5], 'wall_s': result['wall_s']}, indent=2))


if __name__ == '__main__':
    main()
