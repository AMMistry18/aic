"""Choose a learned-keypoint refinement using observed geometry, never truth.

This is an experimental extension, not a calibrated pose-uncertainty model.
Passing these checks establishes self-consistency, not insertion accuracy.
"""
from __future__ import annotations

import numpy as np

from .sfp_plug_pose import fuse_multiview_keypoints


def select_geometric_refinement(views, alternatives, local_keypoints_m, *,
                                max_reprojection_error_px, max_keypoint_rmse_m,
                                max_pose_disagreement_m, min_keypoint_confidence=.15):
    """Return detections and diagnostics; reject incompatible valid hypotheses.

    Alternatives are whole multiview detection sets (e.g. full frame and crop),
    rather than cherry-picked individual keypoints. Gates are supplied by the
    caller. The disagreement bound is an additional conservative gate.
    """
    if not np.isfinite(max_pose_disagreement_m) or max_pose_disagreement_m <= 0:
        raise ValueError('A finite positive pose disagreement bound is required')
    records=[]
    valid=[]
    points=np.asarray(local_keypoints_m,dtype=float)
    for label,detections in alternatives:
        row={'condition':label,'accepted_geometry':False}
        try:
            p,R,shape,reprojection,count,confidence=fuse_multiview_keypoints(
                views,detections,local_keypoints_m=points,min_keypoint_confidence=min_keypoint_confidence)
            good=bool(np.isfinite(shape) and np.isfinite(reprojection)
                      and shape<=max_keypoint_rmse_m and reprojection<=max_reprojection_error_px)
            row.update(accepted_geometry=good,shape_rmse_m=float(shape),reprojection_error_px=float(reprojection),
                       triangulated_count=count,mean_keypoint_confidence=confidence)
            if good:
                valid.append((row,detections,(R@points.T).T+p))
        except ValueError as exc:
            row['failure']=str(exc)
        records.append(row)
    diagnostics={'conditions':records,'selected':None,'GT_used':False}
    if not valid:
        diagnostics['reason']='all_refinements_failed_original_geometry_gates'
        return [],diagnostics
    spread=max([0.]+[float(np.max(np.linalg.norm(a[2]-b[2],axis=1)))
                     for i,a in enumerate(valid) for b in valid[i+1:]])
    diagnostics['maximum_valid_hypothesis_keypoint_disagreement_m']=spread
    if spread>max_pose_disagreement_m:
        diagnostics['reason']='valid_refinements_disagree'
        return [],diagnostics
    winner=min(valid,key=lambda x:(x[0]['reprojection_error_px'],-x[0]['triangulated_count'],x[0]['shape_rmse_m']))
    diagnostics.update(selected=winner[0]['condition'],reason='consistent_geometric_refinement')
    return list(winner[1]),diagnostics
