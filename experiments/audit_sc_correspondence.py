"""Read-only SC label-correspondence/geometry hypotheses, never policy tuning."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sfp_plug_pose import PlugPoseView,PlugKeypointDetection,fuse_multiview_keypoints
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as SC
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M as SFP

def main():
    folder=ROOT/'.artifacts/kilted_handoff_sc_development_20261024/projection_audit'
    raw=json.loads((folder/'audit.json').read_text());rows=[r for r in raw['views'] if 'predicted_px' in r]
    views=[PlugPoseView(r['camera'],cv2.imread(str(folder/f"{r['camera']}_raw.png")),np.array(r['K']),
        np.linalg.inv(r['T_camera_from_base_exact']),r['stamp_s'],r['camera']) for r in rows]
    ds=[PlugKeypointDetection(r['camera'],np.array(r['predicted_px']),np.array(r['confidence']),r['box_confidence']) for r in rows]
    ds.sort(key=lambda d:-d.box_confidence)
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    truth=np.linalg.inv(base)@rows[0]['tip_world']
    cube=np.sign(SC-SC.mean(axis=0));records=[]
    for axes in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            changed=cube[:,axes]*signs
            perm=np.argmin(np.linalg.norm(changed[:,None]-cube[None,:],axis=2),axis=1)
            candidates=[ds[0],replace(ds[1],keypoints_px=ds[1].keypoints_px[perm],keypoint_confidences=ds[1].keypoint_confidences[perm])]
            try:
                p,R,rmse,reproj,*_=fuse_multiview_keypoints(views,candidates,local_keypoints_m=SC)
                records.append({'axes':axes,'signs':signs,'reprojection_px':reproj,'shape_rmse_mm':rmse*1000,
                    'GT_position_error_mm':float(np.linalg.norm(p-truth[:3,3])*1000),
                    'GT_rotation_error_deg':float(Rotation.from_matrix(truth[:3,:3].T@R).magnitude()*180/np.pi)})
            except ValueError:pass
    geometry=[]
    for label,points in [('SC_contract',SC),('SFP_geometry_hypothesis_not_valid_SC_contract',SFP)]:
        p,R,rmse,reproj,*_=fuse_multiview_keypoints(views,ds,local_keypoints_m=points)
        geometry.append({'hypothesis':label,'reprojection_px':reproj,'shape_rmse_mm':rmse*1000,
            'GT_position_error_mm_if_claimed_SC_tip':float(np.linalg.norm(p-truth[:3,3])*1000)})
    records.sort(key=lambda x:x['reprojection_px'])
    result={'purpose':'diagnostic_only_no_policy_change','anchor_camera':ds[0].camera_name,'permutations':records,'geometries':geometry}
    out=ROOT/'.artifacts/sc_correspondence_audit.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'best_permutations':records[:5],'geometries':geometry},indent=2))

if __name__=='__main__':main()
