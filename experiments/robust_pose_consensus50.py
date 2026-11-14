"""Six-point CAD consensus using original learned keypoints, static DEV only."""
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'aic_model'),str(ROOT)]
from aic_model.insertion.sfp_plug_pose import triangulate_dlt,fit_rigid_transform
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
from experiments.angular_pose_development46 import errors


def project(P,world):
    q=(P@np.c_[world,np.ones(len(world))].T).T;return q[:,:2]/q[:,2:]


def fit(inputs,detections,local):
    byname={v['camera']:v for v in inputs};prepared=[]
    for d in detections:
        v=byname[d['camera_name']];P=np.array(v['K'])@np.linalg.inv(v['T_world_from_camera'])[:3]
        prepared.append((P,np.array(d['keypoints_px']),np.array(d['keypoint_confidences'])))
    def solve(selected):
        target=[]
        for keypoint in range(len(local)):
            good=[(P,xy[keypoint],confidence[keypoint]) for P,xy,confidence in selected if confidence[keypoint]>=.15]
            if len(good)<2:raise ValueError('Insufficient observations of a keypoint')
            target.append(triangulate_dlt([g[1] for g in good],[g[0] for g in good],[g[2] for g in good]))
        target=np.array(target);candidates=[]
        for subset in itertools.combinations(range(len(local)),6):
            ids=np.array(subset);R,p,_=fit_rigid_transform(local[ids],target[ids])
            world=(R@local.T).T+p
            # Robust ranking is image-only and independent of scoring truth.
            cost=np.mean(np.concatenate([np.log1p(np.sum((project(P,world)-xy)**2,axis=1)/4) for P,xy,_ in selected]))
            candidates.append((float(cost),p,R))
        _,position,rotation=min(candidates,key=lambda x:x[0])
        def pose(x):return position+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@rotation
        def residual(x):
            p,R=pose(x);world=(R@local.T).T+p
            return np.concatenate([((project(P,world)-xy)*np.sqrt(conf[:,None])).ravel() for P,xy,conf in selected])
        result=least_squares(residual,np.zeros(6),bounds=([-3.]*3+[-6.]*3,[3.]*3+[6.]*3),
                             loss='cauchy',f_scale=1.,diff_step=1e-4,max_nfev=150)
        p,R=pose(result.x);world=(R@local.T).T+p
        reprojection=float(np.mean(np.concatenate([np.linalg.norm(project(P,world)-xy,axis=1) for P,xy,_ in selected])))
        rmse=float(np.sqrt(np.mean(np.sum((world-target)**2,axis=1))))
        return p,R,dict(optimizer_success=bool(result.success),reprojection_error_px=reprojection,keypoint_rmse_m=rmse,
                        original_geometry_gates_pass=bool(result.success and reprojection<=6 and rmse<=.0035))
    p,R,diagnostic=solve(prepared);world=(R@local.T).T+p;loo=[]
    for omit in range(len(prepared)):
        op,oR,_=solve([v for i,v in enumerate(prepared) if i!=omit]);alternate=(oR@local.T).T+op
        loo.append(dict(model_displacement_m=float(np.linalg.norm(alternate-world,axis=1).max()),
            angle_deg=float(np.rad2deg(Rotation.from_matrix(oR@R.T).magnitude()))))
    return dict(position=p.tolist(),rotation=R.tolist(),diagnostic=diagnostic,leave_one_view_out=loo,truth_used_for_fitting=False)


def main():
    out=ROOT/'experiments/angular_development50';out.mkdir(exist_ok=True)
    if (out/'robust_consensus_results.json').exists():raise FileExistsError('Preserve candidate version')
    data=json.loads((ROOT/'experiments/parallel_dev46/angular_inputs.json').read_text());rows=[]
    with threadpool_limits(limits=1):
        for row in data['rows']:
            local=SC_PLUG_LOCAL_KEYPOINTS_M if row['connector']=='sc' else SFP_PLUG_LOCAL_KEYPOINTS_M
            result=dict(connector=row['connector'],case_id=row['case_id']);begin=time.perf_counter()
            try:
                candidate=fit(row['inputs'],row['detections'],local);result['fit']=candidate
                result['scoring_only']=errors(np.array(candidate['position']),np.array(candidate['rotation']),np.array(row['scoring_only']['tip_world']))
                limit=.00025 if row['connector']=='sc' else .0001
                result['view_consistency_passed']=max(v['model_displacement_m'] for v in candidate['leave_one_view_out'])<=limit
            except (ValueError,RuntimeError) as exc:result['reason']=str(exc)
            result['wall_s']=time.perf_counter()-begin;rows.append(result)
            (out/'robust_consensus_results.json').write_text(json.dumps(dict(scope='Static old training/development geometry only, no runtime acceptance or insertion claim',
                final_cases_used=0,original_timestamp_veto_preserved=True,rows=rows),indent=2));print(json.dumps({k:v for k,v in result.items() if k!='fit'}),flush=True)


if __name__=='__main__':main()
