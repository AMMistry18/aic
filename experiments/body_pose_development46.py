"""Direct six-DOF SC body-outline fitting; development RGB diagnostics only."""
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'aic_model'),str(ROOT)]
from aic_model.insertion.sc_body_refinement import template,observations,robot_tool_mask
from aic_model.insertion.sc_rim_refinement import project
from experiments.lastinch_geometry import robot_fk
from experiments.angular_pose_development46 import errors


def fit_pose(views,position,rotation):
    points,_=template();prepared=observations(views,points,position,rotation)
    def pose(x):return position+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@rotation
    def residual(x,selected):
        p,R=pose(x);world=(R@points.T).T+p
        return np.concatenate([np.max((project(P,world)-.5)@D.T,axis=0)-support for P,D,support in selected])
    starts=[np.zeros(6)]
    for axis in range(3):
        for sign in [-1,1]:
            x=np.zeros(6);x[3+axis]=sign;starts.append(x)
    bounds=([-3.]*3+[-5.]*3,[3.]*3+[5.]*3)
    fits=[least_squares(residual,x,args=(prepared,),bounds=bounds,loss='soft_l1',f_scale=1.,diff_step=1e-4,max_nfev=120) for x in starts]
    best=min(fits,key=lambda x:x.cost);p,R=pose(best.x);model=(R@points.T).T+p
    disagreements=[]
    for omit in range(3):
        other=least_squares(residual,best.x,args=([v for i,v in enumerate(prepared) if i!=omit],),
                            bounds=bounds,loss='soft_l1',f_scale=1.,diff_step=1e-4,max_nfev=120)
        op,oR=pose(other.x);alternate=(oR@points.T).T+op
        disagreements.append(dict(maximum_model_displacement_m=float(np.max(np.linalg.norm(alternate-model,axis=1))),
                                  angle_deg=float(np.rad2deg(Rotation.from_matrix(oR@R.T).magnitude()))))
    error=float(np.mean(np.abs(residual(best.x,prepared))))
    accepted=bool(best.success and error<=1. and max(d['maximum_model_displacement_m'] for d in disagreements)<=.00025)
    return dict(position=p.tolist(),rotation=R.tolist(),mean_support_error_px=error,accepted=accepted,
                optimizer_success=bool(best.success),delta_mm_deg=best.x.tolist(),leave_one_view_out=disagreements,
                ground_truth_inputs=False)


def main():
    cv2.setNumThreads(1)
    out=ROOT/'experiments/parallel_dev46';cache=json.loads((out/'angular_inputs.json').read_text())
    previous=json.loads((out/'angular_reprojection_results.json').read_text())
    poses={r['case_id']:r.get('joint_fit') for r in previous['rows']};rows=[]
    robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot()
    with threadpool_limits(limits=1):
        for row in cache['rows']:
            if row['connector']!='sc':continue
            seed=poses[row['case_id']];result=dict(case_id=row['case_id'])
            if seed is None:result['reason']='no_visual_seed'
            else:
                tcp=robot_fk(robot,row['own_robot_q'])['gripper/tcp'];views=[]
                for item in row['inputs']:
                    image=cv2.imread(str(ROOT/item['image']));K=np.array(item['K']);T=np.linalg.inv(item['T_world_from_camera'])
                    views.append(dict(camera=item['camera'],image=image,P=K@T[:3],
                                      tool_mask=robot_tool_mask(K,T,tcp,image.shape)))
                try:
                    begin=time.perf_counter();fit=fit_pose(views,np.array(seed['position']),np.array(seed['rotation']))
                    result.update(fit=fit,wall_s=time.perf_counter()-begin)
                    # Diagnostic truth is accessed only after fitting has returned.
                    result['scoring_only']=errors(np.array(fit['position']),np.array(fit['rotation']),np.array(row['scoring_only']['tip_world']))
                except (ValueError,RuntimeError) as exc:result['reason']=str(exc)
            rows.append(result)
            (out/'body_pose_results.json').write_text(json.dumps(dict(scope='Static development image geometry only; no insertion/angular-motion qualification',
                final_cases_used=0,truth_used_for_optimization=False,rows=rows),indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='fit'}),flush=True)


if __name__=='__main__':main()
