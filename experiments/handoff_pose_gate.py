"""Evaluator-only check of the actual prescribed handoff, not a policy input."""
import numpy as np
from scipy.spatial.transform import Rotation


def check_tip_window(poses, desired, end):
    rows = [(t, np.asarray(p)) for t, p in poses if end - .5 <= t <= end + .025]
    covered = bool(rows and rows[0][0] <= end - .47 and rows[-1][0] >= end - .025
                   and all(0 < b[0] - a[0] <= .03 for a, b in zip(rows, rows[1:])))
    desired = np.asarray(desired)
    errors = [float(np.linalg.norm(p[:3, 3] - desired[:3, 3]) * 1000) for _, p in rows]
    angles = [float(Rotation.from_matrix(desired[:3, :3].T @ p[:3, :3]).magnitude()
                    * 180 / np.pi) for _, p in rows]
    return {'passed': bool(covered and max(errors) <= .25 and max(angles) <= .25),
            'covered': covered, 'max_position_error_mm': max(errors, default=None),
            'max_angle_error_deg': max(angles, default=None), 'samples': len(rows),
            'position_limit_mm': .25, 'angle_limit_deg': .25, 'window_s': .5}
