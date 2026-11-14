"""Recompute visible SC boundary support after each six-DOF image fit.

Known robot geometry and learned initialization only. Diagnostic object truth
is read after all fits; the production synchronization veto stays unchanged.
"""
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

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'aic_model'),str(ROOT)]
from aic_model.insertion.sc_body_refinement import template,observations,robot_tool_mask
from aic_model.insertion.sc_rim_refinement import project
from experiments.lastinch_geometry import robot_fk
from experiments.angular_pose_development46 import errors


def fit(views,position,rotation):
    points,_=template();bounds=([-3.]*3+[-5.]*3,[3.]*3+[5.]*3)
    def pose(x):return position+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@rotation
    def residual(x,prepared):
        p,R=pose(x);world=(R@points.T).T+p
        return np.concatenate([np.max((project(P,world)-.5)@D.T,axis=0)-support for P,D,support in prepared])
    current=np.zeros(6);steps=[]
    for iteration in range(4):
        p,R=pose(current);prepared=observations(views,points,p,R)
        starts=[current]
        if iteration==0:
            for axis in range(3):
                for sign in [-1,1]:
                    x=current.copy();x[3+axis]=sign;starts.append(x)
        fits=[least_squares(residual,x,args=(prepared,),bounds=bounds,loss='soft_l1',f_scale=1.,
                            diff_step=1e-4,max_nfev=120) for x in starts]
        best=min(fits,key=lambda x:x.cost);delta=float(np.linalg.norm(best.x-current));current=best.x
        steps.append(dict(iteration=iteration,parameter_change_mm_deg=delta,
                          support_error_px=float(np.mean(abs(residual(current,prepared)))),
                          directions_per_view=[len(D) for _,D,_ in prepared]))
        if delta<1e-5:break
    p,R=pose(current);prepared=observations(views,points,p,R);world=(R@points.T).T+p
    error=float(np.mean(abs(residual(current,prepared))));disagreement=[]
    for omit in range(3):
        subset=[row for i,row in enumerate(prepared) if i!=omit]
        other=least_squares(residual,current,args=(subset,),bounds=bounds,loss='soft_l1',f_scale=1.,diff_step=1e-4,max_nfev=120)
        op,oR=pose(other.x);alternate=(oR@points.T).T+op
        disagreement.append(dict(model_displacement_m=float(np.linalg.norm(alternate-world,axis=1).max()),
            angle_deg=float(np.rad2deg(Rotation.from_matrix(oR@R.T).magnitude()))))
    accepted=bool(best.success and error<=1. and max(v['model_displacement_m'] for v in disagreement)<=.00025)
    return dict(position=p.tolist(),rotation=R.tolist(),mean_support_error_px=error,
                accepted=accepted,iterations=steps,leave_one_view_out=disagreement,
                ground_truth_used_for_fitting=False)


def main():
    cv2.setNumThreads(1);out=ROOT/'experiments/angular_development47';out.mkdir(exist_ok=True)
    if (out/'iterative_body_results.json').exists():raise FileExistsError('Preserve the first experiment')
    inputs=json.loads((ROOT/'experiments/parallel_dev46/angular_inputs.json').read_text())['rows']
    seeds={r['case_id']:r.get('joint_fit') for r in json.loads((ROOT/'experiments/parallel_dev46/angular_reprojection_results.json').read_text())['rows']}
    robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot();rows=[]
    with threadpool_limits(limits=1):
        for row in inputs:
            if row['connector']!='sc':continue
            seed=seeds[row['case_id']];tcp=robot_fk(robot,row['own_robot_q'])['gripper/tcp'];views=[]
            for v in row['inputs']:
                image=cv2.imread(str(ROOT/v['image']));K=np.array(v['K']);T=np.linalg.inv(v['T_world_from_camera'])
                views.append(dict(camera=v['camera'],image=image,P=K@T[:3],tool_mask=robot_tool_mask(K,T,tcp,image.shape)))
            result=dict(case_id=row['case_id']);begin=time.perf_counter()
            try:
                candidate=fit(views,np.array(seed['position']),np.array(seed['rotation']));result['fit']=candidate
                result['scoring_only']=errors(np.array(candidate['position']),np.array(candidate['rotation']),np.array(row['scoring_only']['tip_world']))
            except (ValueError,RuntimeError) as exc:result['reason']=str(exc)
            result['wall_s']=time.perf_counter()-begin;rows.append(result)
            (out/'iterative_body_results.json').write_text(json.dumps(dict(scope='Static development CAD/RGB fitting only; no runtime pose or insertion acceptance',
                final_cases_used=0,production_stamp_veto_preserved=True,rows=rows),indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='fit'}),flush=True)


if __name__=='__main__':main()
