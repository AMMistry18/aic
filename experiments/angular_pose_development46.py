"""Short direct joint SE(3) reprojection tests on cached development detections.

Original weighted triangulation/Kabsch is compared with confidence-weighted
joint image fitting. Truth is a scoring argument, never an optimizer input.
"""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
from aic_model.insertion.sfp_plug_pose import PlugPoseView,PlugKeypointDetection,fuse_multiview_keypoints


def project(P,points):
    q=(P@np.c_[points,np.ones(len(points))].T).T
    return q[:,:2]/q[:,2:]


def fit_joint(inputs,detections,local,position,rotation):
    cameras={v['camera']:v for v in inputs};prepared=[]
    for detection in detections:
        camera=cameras[detection['camera_name']]
        good=np.asarray(detection['keypoint_confidences'])>=.15
        if sum(good)<4:continue
        P=np.asarray(camera['K'])@np.linalg.inv(np.asarray(camera['T_world_from_camera']))[:3]
        prepared.append((P,np.asarray(detection['keypoints_px'])[good],
                         np.sqrt(np.asarray(detection['keypoint_confidences'])[good,None]),local[good]))
    if len(prepared)<2:return None
    def pose(x):return position+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@rotation
    def residual(x,selected):
        p,R=pose(x)
        return np.concatenate([((project(P,(R@points.T).T+p)-xy)*weight).ravel() for P,xy,weight,points in selected])
    starts=[np.zeros(6)]
    for axis in range(3):
        for sign in [-1,1]:
            x=np.zeros(6);x[3+axis]=sign;starts.append(x)
    options=[least_squares(residual,x,args=(prepared,),bounds=([-5.]*6,[5.]*6),loss='soft_l1',
                           f_scale=1.,diff_step=1e-4,max_nfev=100) for x in starts]
    best=min(options,key=lambda x:x.cost);p,R=pose(best.x)
    disagreements=[]
    for omitted in range(len(prepared)):
        selected=[v for i,v in enumerate(prepared) if i!=omitted]
        if len(selected)<2:continue
        other=least_squares(residual,best.x,args=(selected,),bounds=([-5.]*6,[5.]*6),loss='soft_l1',
                            f_scale=1.,diff_step=1e-4,max_nfev=100)
        op,oR=pose(other.x)
        disagreements.append(dict(position_m=float(np.linalg.norm(op-p)),angle_deg=float(np.rad2deg(Rotation.from_matrix(oR@R.T).magnitude()))))
    reprojection=[np.linalg.norm(project(P,(R@points.T).T+p)-xy,axis=1) for P,xy,_,points in prepared]
    return dict(position=p.tolist(),rotation=R.tolist(),optimizer_success=bool(best.success),
                mean_reprojection_error_px=float(np.mean(np.concatenate(reprojection))),
                fitted_delta_mm_deg=best.x.tolist(),leave_one_view_out=disagreements)


def errors(position,rotation,truth):
    return dict(position_mm=float(np.linalg.norm(position-truth[:3,3])*1000),
                angle_deg=float(np.rad2deg(Rotation.from_matrix(rotation@truth[:3,:3].T).magnitude())))


def main():
    cv2.setNumThreads(1)
    out=ROOT/'experiments/parallel_dev46';data=json.loads((out/'angular_inputs.json').read_text());rows=[]
    with threadpool_limits(limits=1):
        for row in data['rows']:
            result=dict(connector=row['connector'],case_id=row['case_id'],original_failure=row['failure_reason'],
                        original_crop_pipeline_wall_s=row['full_original_crop_multiview_wall_s'])
            baseline=row['original_estimate']
            if baseline is None:
                # Static rendered bodies/cameras never move, but the collector
                # stored each camera's first received exposure asynchronously.
                # Keep the production timestamp rejection above. Test only the
                # pure geometry primitive, without changing any runtime gate.
                views=[PlugPoseView(v['camera'],cv2.imread(str(ROOT/v['image'])),np.array(v['K']),
                                   np.array(v['T_world_from_camera']),v['stamp_s'],v['camera']) for v in row['inputs']]
                detections=[PlugKeypointDetection(**d) for d in row['detections']]
                local=SC_PLUG_LOCAL_KEYPOINTS_M if row['connector']=='sc' else SFP_PLUG_LOCAL_KEYPOINTS_M
                try:
                    p,R,rmse,reprojection,count,confidence=fuse_multiview_keypoints(views,detections,local_keypoints_m=local)
                    baseline=dict(position_world=p,rotation_world_from_plug=R)
                    result['static_geometry_diagnostic']=dict(production_clock_rejection_preserved=True,
                        rmse_m=rmse,reprojection_px=reprojection,production_pose_accepted=False)
                except ValueError as exc:result['static_geometry_failure']=str(exc)
            if baseline is not None:
                position=np.asarray(baseline['position_world']);rotation=np.asarray(baseline['rotation_world_from_plug'])
                local=SC_PLUG_LOCAL_KEYPOINTS_M if row['connector']=='sc' else SFP_PLUG_LOCAL_KEYPOINTS_M
                begin=time.perf_counter();candidate=fit_joint(row['inputs'],row['detections'],local,position,rotation)
                result['joint_fit_wall_s']=time.perf_counter()-begin;result['joint_fit']=candidate
                # Truth is accessed only after all candidate fitting finishes.
                truth=np.asarray(row['scoring_only']['tip_world'])
                result['original_errors']=errors(position,rotation,truth)
                if candidate:result['joint_errors']=errors(np.array(candidate['position']),np.array(candidate['rotation']),truth)
            rows.append(result)
            (out/'angular_reprojection_results.json').write_text(json.dumps(dict(
                scope='Development perception diagnostics only; no angular-motion or insertion success claimed',
                final_cases_read=0,truth_used_for_optimization=False,rows=rows),indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='joint_fit'}),flush=True)


if __name__=='__main__':main()
