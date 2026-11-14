"""Depth-ordered own-robot occlusion diagnostic on old static DEV images.

The mask uses FK, robot CAD and the original learned pose only. Object truth is
accessed after fitting, for scoring. The production timestamp veto is retained;
these geometry fits cannot be deployed as synchronized runtime measurements.
"""
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
from aic_model.insertion.sc_body_refinement import template
from experiments.lastinch_geometry import robot_fk
from experiments.angular_pose_development46 import fit_joint,errors


def nearest_robot_depth(triangles,ray):
    """Double-sided Moller-Trumbore ray intersections; ray camera Z is one."""
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    h=np.cross(np.broadcast_to(ray,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
    valid=abs(det)>1e-12;inverse=np.zeros(len(det));inverse[valid]=1/det[valid]
    s=-a;u=np.einsum('ij,ij->i',s,h)*inverse;q=np.cross(s,e1)
    v=(q@ray)*inverse;depth=np.einsum('ij,ij->i',e2,q)*inverse
    valid&=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(depth>.001)
    return float(depth[valid].min()) if valid.any() else None


def main():
    out=ROOT/'experiments/angular_development52';out.mkdir(exist_ok=True)
    target=out/'depth_ordered_results.json'
    if target.exists():raise FileExistsError('Preserve first diagnostic')
    # Independent analytic intersection sanity checks before using the mesh.
    plane=np.array([[[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]]])
    assert abs(nearest_robot_depth(plane,np.array([0.,0.,1.]))-2)<1e-12
    assert nearest_robot_depth(plane,np.array([2.,0.,1.])) is None
    robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot()
    _,triangles=template();cache=json.loads((ROOT/'experiments/parallel_dev46/angular_inputs.json').read_text());rows=[]
    cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        for row in cache['rows']:
            local=SC_PLUG_LOCAL_KEYPOINTS_M if row['connector']=='sc' else SFP_PLUG_LOCAL_KEYPOINTS_M
            tcp=robot_fk(robot,row['own_robot_q'])['gripper/tcp'];views=[]
            for v in row['inputs']:
                views.append(PlugPoseView(v['camera'],cv2.imread(str(ROOT/v['image'])),np.array(v['K']),
                    np.array(v['T_world_from_camera']),v['stamp_s'],v['camera']))
            detections=copy.deepcopy(row['detections']);result=dict(connector=row['connector'],case_id=row['case_id'])
            try:
                original=fuse_multiview_keypoints(views,[PlugKeypointDetection(**d) for d in detections],local_keypoints_m=local)
                p,R=original[:2];world=(R@local.T).T+p;depth_records={}
                for v in row['inputs']:
                    T=np.linalg.inv(v['T_world_from_camera']);tool=T@tcp
                    camera_triangles=((tool[:3,:3]@triangles.reshape(-1,3).T).T+tool[:3,3]).reshape(-1,3,3)
                    expected_depth=((T[:3,:3]@world.T).T+T[:3,3])[:,2]
                    d=next(x for x in detections if x['camera_name']==v['camera'])
                    rays=(np.linalg.inv(v['K'])@np.c_[np.array(d['keypoints_px'])+.5,np.ones(len(local))].T).T
                    records=[]
                    for index,ray in enumerate(rays):
                        depth=nearest_robot_depth(camera_triangles,ray)
                        # Only a surface clearly in front by >=1mm is excluded;
                        # that reserve accommodates uncertainty in the proposal.
                        hidden=bool(depth is not None and depth<expected_depth[index]-.001)
                        if hidden:d['keypoint_confidences'][index]=0.
                        records.append(dict(index=index,nearest_robot_depth_m=depth,
                            learned_keypoint_depth_m=float(expected_depth[index]),excluded=hidden))
                    depth_records[v['camera']]=records
                result['depth_records']=depth_records
                estimate=fuse_multiview_keypoints(views,[PlugKeypointDetection(**d) for d in detections],local_keypoints_m=local)
                p,R,rmse,reproj,count,_=estimate;candidate=fit_joint(row['inputs'],detections,local,p,R)
                result.update(triangulated=count,reprojection_px=reproj,rmse_m=rmse,joint=candidate)
                truth=np.array(row['scoring_only']['tip_world']);result['scoring_only']=errors(p,R,truth)
                if candidate:result['joint_scoring_only']=errors(np.array(candidate['position']),np.array(candidate['rotation']),truth)
            except (ValueError,RuntimeError) as exc:result['reason']=str(exc)
            rows.append(result)
            target.write_text(json.dumps(dict(scope='Static old DEVELOPMENT geometry only, no synchronized pose or insertion claim',
                final_cases_used=0,policy_weights_changed=False,original_timestamp_veto_preserved=True,
                production_policy_modified=False,rows=rows),indent=2),encoding='utf-8')
            print(json.dumps(dict(connector=result['connector'],case_id=result['case_id'],reason=result.get('reason'),
                excluded={k:sum(x['excluded'] for x in v) for k,v in result.get('depth_records',{}).items()},
                joint_scoring_only=result.get('joint_scoring_only'))),flush=True)


if __name__=='__main__':main()
