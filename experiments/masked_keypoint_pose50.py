"""Test robot-self-occlusion exclusion with original learned geometry gates."""
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'aic_model'),str(ROOT)]
from aic_model.insertion.sfp_plug_pose import PlugPoseView,PlugKeypointDetection,fuse_multiview_keypoints
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
from aic_model.insertion.sc_body_refinement import robot_tool_mask
from experiments.lastinch_geometry import robot_fk
from experiments.angular_pose_development46 import fit_joint,errors


def main():
    cv2.setNumThreads(1);out=ROOT/'experiments/angular_development50';out.mkdir(exist_ok=True)
    if (out/'masked_keypoint_results.json').exists():raise FileExistsError('Preserve first visibility test')
    cache=json.loads((ROOT/'experiments/parallel_dev46/angular_inputs.json').read_text());rows=[]
    robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot()
    with threadpool_limits(limits=1):
        for row in cache['rows']:
            local=SC_PLUG_LOCAL_KEYPOINTS_M if row['connector']=='sc' else SFP_PLUG_LOCAL_KEYPOINTS_M
            tcp=robot_fk(robot,row['own_robot_q'])['gripper/tcp'];views=[];detections=copy.deepcopy(row['detections']);exclusions={}
            for v in row['inputs']:
                image=cv2.imread(str(ROOT/v['image']));K=np.array(v['K']);T=np.linalg.inv(v['T_world_from_camera'])
                mask=cv2.dilate(robot_tool_mask(K,T,tcp,image.shape),np.ones((5,5),np.uint8))
                d=next(d for d in detections if d['camera_name']==v['camera']);xy=np.rint(d['keypoints_px']).astype(int)
                h,w=mask.shape;valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                hidden=~valid;hidden[valid]=mask[xy[valid,1],xy[valid,0]]!=0
                conf=np.array(d['keypoint_confidences']);conf[hidden]=0.;d['keypoint_confidences']=conf.tolist()
                exclusions[v['camera']]=int(hidden.sum())
                views.append(PlugPoseView(v['camera'],image,K,np.array(v['T_world_from_camera']),v['stamp_s'],v['camera']))
            result=dict(connector=row['connector'],case_id=row['case_id'],excluded_keypoints_per_camera=exclusions)
            try:
                p,R,rmse,reproj,count,confidence=fuse_multiview_keypoints(views,[PlugKeypointDetection(**d) for d in detections],local_keypoints_m=local)
                candidate=fit_joint(row['inputs'],detections,local,p,R)
                result.update(geometry_only=dict(position=p.tolist(),rotation=R.tolist(),rmse_m=rmse,reprojection_px=reproj,triangulated=count),joint=candidate)
                # All diagnostic truth is accessed after both estimates finish.
                truth=np.array(row['scoring_only']['tip_world']);result['scoring_only']=errors(p,R,truth)
                if candidate:result['joint_scoring_only']=errors(np.array(candidate['position']),np.array(candidate['rotation']),truth)
            except (ValueError,RuntimeError) as exc:result['reason']=str(exc)
            rows.append(result);(out/'masked_keypoint_results.json').write_text(json.dumps(dict(
                scope='Static old DEVELOPMENT images only; original timestamp veto and six-keypoint minimum preserved',
                final_cases_used=0,original_learned_weights_unchanged=True,rows=rows),indent=2))
            print(json.dumps({k:v for k,v in result.items() if k not in ['geometry_only','joint']}),flush=True)


if __name__=='__main__':main()
