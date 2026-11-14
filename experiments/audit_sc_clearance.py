"""Offline source-box clearance by insertion depth; never a policy input.

The 20 x 6.4 mm learned keypoint cuboid is not the collision envelope.
Report the translation at first box overlap, retaining the limiting pair.
Cylindrical ferrules are excluded, so this is a box-only diagnostic.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.lastinch_geometry import source_model, sdf_frames
from experiments.audit_sc_collision_fit import boxes, penetration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    port, plug = source_model('SC Port'), source_model('SC Plug')
    pf, sf = sdf_frames(port), sdf_frames(plug)
    entrance = pf['sc_port_base_link_entrance']
    target = boxes(port.find("link[@name='sc_port_link']"), pf['sc_port_link'])

    def worst(depth, xy):
        tip = entrance.copy()
        tip[:3, 3] += entrance[:3, :3] @ np.r_[xy, depth]
        transform = tip @ np.linalg.inv(sf['sc_tip_link'])
        moving = boxes(plug.find("link[@name='sc_plug_link']"), transform)
        value, pair = max((penetration(a, b), (a[0], b[0])) for a in moving for b in target)
        return float(value), pair

    rows = []
    for depth in [0, .003, .005, .010, .012, .015, .01564]:
        baseline, pair = worst(depth, [0., 0.])
        row = {'depth_mm': depth * 1000, 'nominal_max_box_overlap_mm': baseline * 1000,
               'nominal_limiting_pair': pair, 'first_overlap_translation_mm': {}}
        for axis in range(2):
            for sign in [-1, 1]:
                # Existing shoulder overlap at the seated depth is reported,
                # not hidden by subtracting it or raising a collision gate.
                if baseline > 1e-7:
                    result = {'translation_mm': None, 'reason': 'nominal_box_overlap'}
                else:
                    lo, hi = 0., .004
                    for _ in range(28):
                        mid = (lo + hi) / 2
                        xy = np.zeros(2); xy[axis] = sign * mid
                        overlap, _ = worst(depth, xy)
                        if overlap > 0.: hi = mid
                        else: lo = mid
                    xy = np.zeros(2); xy[axis] = sign * hi
                    _, limiting = worst(depth, xy)
                    result = {'translation_mm': hi * 1000, 'limiting_pair': limiting}
                row['first_overlap_translation_mm'][f'port_{"xy"[axis]}_{sign:+}'] = result
        rows.append(row)
    payload = {'purpose': __doc__, 'asset_revision': '7148df9ab62b4b550b101189098f05bc0dba8444',
               'diagnostic_only': True, 'cylinders_included': False, 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    for row in rows:
        print(json.dumps({'depth_mm': row['depth_mm'], 'nominal_overlap_mm': row['nominal_max_box_overlap_mm'],
                          'clearances_mm': {k: v['translation_mm'] for k, v in row['first_overlap_translation_mm'].items()}}))


if __name__ == '__main__':
    main()
