"""Replay SC side-reference rejection; truth is an offline counterfactual only."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'aic_model'), str(ROOT/'flowstate/aic_perception')]
from aic_model.insertion.sc_visual_alignment import (
    measure_sc_blue_side_signature, _rectify_sc_port_view,
    _recovery_side_band_slices,
    detect_sc_duplex_opening, ray_to_plane, fuse_sc_opening_hits,
)
from aic_perception.gripper_masks import GripperMaskBank


def project(P, points):
    q = (P @ np.c_[points, np.ones(len(points))].T).T
    return q[:, :2] / q[:, 2:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--capture-phase', default='perception_before')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    archive = ROOT/'.artifacts/lastinch_runs'/args.run
    folder = archive/args.run/args.capture_phase
    views = json.loads((folder/'camera_geometry.json').read_text())
    log = (ROOT/'.artifacts/container'/f'{args.run}_policy.log').read_text()
    first = next(line for line in log.splitlines() if 'SC_PERCEPT_BEST' in line)
    detections = ast.literal_eval(first.split(' cams=', 1)[1])
    matrices = {v['camera']: np.array(v['K']) @ np.linalg.inv(v['T_base_from_camera'])[:3] for v in views}
    points = []
    for k in range(4):
        A = []
        for d in detections:
            P = matrices[d['cam']]; u, v = d['kps'][k]
            A.extend([u*P[2]-P[0], v*P[2]-P[1]])
        _, _, vt = np.linalg.svd(A); points.append(vt[-1, :3]/vt[-1, 3])
    points = np.array(points)
    x = (points[0]+points[3]-points[1]-points[2])/2; x[2] = 0; x /= np.linalg.norm(x)
    z = np.array([0., 0., -1.]); y = np.cross(z, x)
    learned = np.eye(4); learned[:3, :3] = np.column_stack([x, y, z])
    consensus = re.search(r'perception consensus:.*?port=(\[.*?\])', log)
    learned[:3, 3] = ast.literal_eval(consensus[1] if consensus else re.search(r' X=(\[.*?\])', first)[1])
    # Evaluator-only comparison: never used to select/repair a live estimate.
    scene = json.loads((archive/'scene.json').read_text())
    base = np.eye(4); base[:3, :3] = Rotation.from_euler('z', -np.pi).as_matrix(); base[:3, 3] = [-.2, .2, 0]
    truth = np.linalg.inv(base) @ np.array(scene['reset']['mouth_world'])
    quads = np.array([[[cx+.00971/2, .00785/2, 0], [cx-.00971/2, .00785/2, 0],
                       [cx-.00971/2, -.00785/2, 0], [cx+.00971/2, -.00785/2, 0]] for cx in [.01270/2, -.01270/2]])
    bank = GripperMaskBank(); results = []; panels = []; refinements = []
    for name, T in [('learned_rounded_log_reconstruction', learned), ('truth_diagnostic_only', truth)]:
        hits = []; detection_rows = []
        for view in views:
            camera = view['camera']; image = cv2.imread(str(folder/f'{camera}_raw.png'))
            world = (T[:3, :3] @ quads.reshape(-1, 3).T).T + T[:3, 3]
            uv = project(matrices[camera], world).reshape(2, 4, 2)
            ignored = bank.ignored_pixels(camera, image.shape)
            d = {}; detected = detect_sc_duplex_opening(image, uv, ignored, diagnostics=d, search_scale=1.65)
            row = {'camera': camera, 'reason': d.get('reason'), 'detected': detected is not None}
            if detected is not None:
                hit = ray_to_plane(detected.center_uv, np.array(view['K']), np.linalg.inv(view['T_base_from_camera']), plane_point=T[:3, 3], plane_normal=T[:3, 2])
                if hit is not None: hits.append({'camera': camera, 'plane_point': hit})
                row.update(center_uv=detected.center_uv.tolist(), bores=detected.detected_bores)
            detection_rows.append(row)
            for mask_name, mask in [('masked', ignored), ('unmasked', None)]:
                diagnostics = {}; signature = measure_sc_blue_side_signature(image, uv, mask, band_half_width_px=12, diagnostics=diagnostics)
                row = {'pose': name, 'camera': camera, 'mask': mask_name,
                       'reason': diagnostics['reason'], 'fractions': None if signature is None else signature.blue_fractions.tolist(),
                       'passes_existing_reference_gate': bool(signature is not None and np.all(signature.blue_fractions >= .5))}
                results.append(row)
            rect, valid, bounds = _rectify_sc_port_view(image, uv, ignored)
            rect[~valid] = [100, 0, 100]
            for i, (ys, xs) in enumerate(_recovery_side_band_slices(bounds, 12)):
                cv2.rectangle(rect, (xs.start, ys.start), (xs.stop, ys.stop), (0, 255, 255), 1)
                cv2.putText(rect, str(i), (xs.start, ys.start+12), 0, .4, (255,255,255), 1)
            canvas = cv2.copyMakeBorder(rect, 35, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(canvas, name.split('_')[0]+' '+camera, (5, 20), 0, .5, (255,255,255), 1)
            panels.append(canvas)
            cv2.imwrite(str(args.out/f'{name}_{camera}.png'), canvas)
        fused = fuse_sc_opening_hits(hits, origin_port_pos=T[:3, 3], Rp=T[:3, :3], max_view_disagreement_m=.001, max_total_offset_m=.003, allow_single_view=False)
        refinements.append({'pose': name, 'detections': detection_rows, 'accepted': fused is not None,
                            'position': None if fused is None else fused.point_world.tolist(),
                            'diagnostic_error_mm': None if fused is None else float(np.linalg.norm(fused.point_world-truth[:3, 3])*1000)})
    cv2.imwrite(str(args.out/'reference_comparison.png'), np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])]))
    payload = {'run': args.run, 'learned_pose_precision': 'rounded log, first candidate orientation and consensus position; approximate replay',
               'learned_pose_base': learned.tolist(), 'capture_folder': str(folder),
               'ground_truth_used_only_for_diagnostic_comparison': True, 'results': results, 'opening_refinement': refinements}
    (args.out/'reference_audit.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(refinements, indent=2))


if __name__ == '__main__': main()
