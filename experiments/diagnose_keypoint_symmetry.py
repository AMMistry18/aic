"""Offline diagnostic of cuboid correspondence ambiguity; never a policy fix."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sfp_plug_pose import PlugPoseView, PlugKeypointDetection, fuse_multiview_keypoints
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as points


def main():
    run=ROOT/'.artifacts/kilted_handoff_v1_before_ipc'
    info=json.loads((run/'policy_images/camera_geometry.json').read_text())
    raw=json.loads((run/'live_policy_probe_primed.json').read_text())
    views=[PlugPoseView(r['name'],cv2.imread(str(run/'policy_images'/f"{r['name']}_raw.png")),
        np.array(r['K']),np.array(r['T_base_from_camera']),r['stamp_s'],r['frame_id']) for r in info]
    detections=[PlugKeypointDetection(r['camera'],np.array(r['keypoints_px']),np.ones(8),r['confidence'])
                for r in raw['plug_detections']]
    detections.sort(key=lambda d:-d.box_confidence)
    centered=points-points.mean(axis=0)
    permutations=[]
    for signs in itertools.product((-1,1),repeat=3):
        if np.prod(signs)!=1:continue
        changed=centered*np.array(signs)
        index=np.argmin(np.linalg.norm(changed[:,None]-centered[None,:],axis=2),axis=1)
        permutations.append((signs,index))
    rows=[]
    for signs,perm in permutations:
        ds=[detections[0],*[replace(d,keypoints_px=d.keypoints_px[perm]) for d in detections[1:]]]
        try:
            p,R,rmse,reproj,*_=fuse_multiview_keypoints(views,ds,local_keypoints_m=points)
            rows.append({'other_view_cuboid_rotation':signs,'reprojection_px':reproj,
                         'shape_rmse_mm':rmse*1000,'position_base':p.tolist()})
        except Exception as exc:rows.append({'other_view_cuboid_rotation':signs,'error':str(exc)})
    rows.sort(key=lambda r:r.get('reprojection_px',float('inf')))
    result={'purpose':'single_stored_frame_correspondence_diagnostic_not_validation',
            'keypoint_confidence_note':'all usable confidences replaced by one; ranking diagnostic only',
            'anchor_camera':detections[0].camera_name,'candidates':rows}
    out=ROOT/'.artifacts/container/correspondence_diagnostic.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
