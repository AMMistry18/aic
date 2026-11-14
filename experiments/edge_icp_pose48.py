"""Subpixel point-to-normal CAD contour fitting on static DEVELOPMENT RGB.

Uses learned pose initialization and own robot mask; no truth in correspondence
search/optimization. Truth is read after fitting only. All static timestamp
vetoes remain; no runtime angular correction is qualified by this script.
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
from aic_model.insertion.sc_body_refinement import template as sc_template,robot_tool_mask
from aic_model.insertion.sc_rim_refinement import project
from experiments.sfp_shell_pose47 import template as sfp_template
from experiments.lastinch_geometry import robot_fk
from experiments.angular_pose_development46 import errors


def correspondences(view,points,position,rotation):
    uv=project(view['P'],(rotation@points.T).T+position)-.5
    ids=cv2.convexHull(uv.astype(np.float32),returnPoints=False).ravel();local=[];normals=[]
    for i,j in zip(ids,np.roll(ids,-1)):
        direction=uv[j]-uv[i];length=np.linalg.norm(direction)
        if length<3:continue
        normal=np.array([direction[1],-direction[0]])/length
        n=max(3,min(30,int(length/3)))
        for fraction in np.linspace(.1,.9,n):
            local.append(points[i]*(1-fraction)+points[j]*fraction);normals.append(normal)
    local=np.array(local);normals=np.array(normals)
    if not len(local):raise ValueError('no_cad_contour')
    predicted=project(view['P'],(rotation@local.T).T+position)-.5
    s=np.linspace(-6,6,121);queries=predicted[:,None,:]+normals[:,None,:]*s[None,:,None]
    gray=view['gray'];h,w=gray.shape
    inside=np.all((queries[:,:,0]>=1)&(queries[:,:,0]<w-1)&(queries[:,:,1]>=1)&(queries[:,:,1]<h-1),axis=1)
    xy=np.rint(predicted).astype(int);safe=np.clip(xy,[0,0],[w-1,h-1]);inside&=view['mask'][safe[:,1],safe[:,0]]==0
    profiles=cv2.remap(gray,queries[:,:,0].astype(np.float32),queries[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
    gradient=abs(np.gradient(profiles,s,axis=1));gradient[:,:10]=0;gradient[:,-10:]=0
    score=gradient*np.exp(-s[None,:]**2/(2*4**2));peaks=np.argmax(score,axis=1)
    selected=[];targets=[];weights=[]
    for i,j in enumerate(peaks):
        if not inside[i] or gradient[i,j]<15:continue
        left=float(np.median(profiles[i,j-10:j-5]));right=float(np.median(profiles[i,j+5:j+10]))
        contrast=right-left
        if abs(contrast)<30:continue
        alpha=(profiles[i]-left)/contrast
        crossings=np.flatnonzero((alpha[:-1]<=.5)&(alpha[1:]>.5))
        if not len(crossings):continue
        k=min(crossings,key=lambda q:abs(s[q]-s[j]));offset=s[k]+(.5-alpha[k])*(s[k+1]-s[k])/(alpha[k+1]-alpha[k])
        if abs(offset)>5.5:continue
        target=predicted[i]+normals[i]*offset;xy=np.rint(target).astype(int)
        if view['mask'][xy[1],xy[0]]:continue
        selected.append(i);targets.append(float(normals[i]@target));weights.append(np.sqrt(min(1.,abs(contrast)/100)))
    if len(selected)<12:raise ValueError(f"insufficient_visible_contour:{view['camera']}:{len(selected)}")
    return view['P'],local[selected],normals[selected],np.array(targets),np.array(weights)


def fit(views,points,position,rotation):
    bounds=([-3.]*3+[-6.]*3,[3.]*3+[6.]*3)
    def pose(x):return position+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@rotation
    def residual(x,prepared):
        p,R=pose(x);rows=[]
        for P,local,n,target,weight in prepared:
            uv=project(P,(R@local.T).T+p)-.5;rows.append((np.sum(uv*n,axis=1)-target)*weight)
        return np.concatenate(rows)
    current=np.zeros(6);iterations=[]
    for iteration in range(5):
        p,R=pose(current);prepared=[correspondences(v,points,p,R) for v in views]
        best=least_squares(residual,current,args=(prepared,),bounds=bounds,loss='soft_l1',f_scale=.5,diff_step=1e-4,max_nfev=120)
        change=float(np.linalg.norm(current-best.x));current=best.x
        iterations.append(dict(iteration=iteration,mean_normal_error_px=float(np.mean(abs(residual(current,prepared)))),
            points_per_view=[len(p[1]) for p in prepared],parameter_change_mm_deg=change))
        if change<1e-4:break
    p,R=pose(current);world=(R@points.T).T+p;disagreements=[]
    for omit in range(3):
        other=least_squares(residual,current,args=([v for i,v in enumerate(prepared) if i!=omit],),
            bounds=bounds,loss='soft_l1',f_scale=.5,diff_step=1e-4,max_nfev=120)
        op,oR=pose(other.x);alternative=(oR@points.T).T+op
        disagreements.append(float(np.linalg.norm(alternative-world,axis=1).max()))
    return dict(position=p.tolist(),rotation=R.tolist(),iterations=iterations,
        leave_one_view_out_displacement_m=disagreements,ground_truth_used=False)


def main():
    cv2.setNumThreads(1);out=ROOT/'experiments/angular_development48';out.mkdir(exist_ok=True)
    if (out/'contour_results.json').exists():raise FileExistsError('Preserve candidate version')
    points=dict(sc=sc_template()[0],sfp=sfp_template('metallic')[0])
    cache=json.loads((ROOT/'experiments/parallel_dev46/angular_inputs.json').read_text())['rows']
    seeds={r['case_id']:r['joint_fit'] for r in json.loads((ROOT/'experiments/parallel_dev46/angular_reprojection_results.json').read_text())['rows']}
    robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot();rows=[]
    with threadpool_limits(limits=1):
        for row in cache:
            seed=seeds[row['case_id']];tcp=robot_fk(robot,row['own_robot_q'])['gripper/tcp'];views=[]
            for v in row['inputs']:
                image=cv2.imread(str(ROOT/v['image']));K=np.array(v['K']);T=np.linalg.inv(v['T_world_from_camera'])
                mask=cv2.dilate(robot_tool_mask(K,T,tcp,image.shape),np.ones((5,5),np.uint8))
                views.append(dict(camera=v['camera'],gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32),P=K@T[:3],mask=mask))
            result=dict(connector=row['connector'],case_id=row['case_id']);begin=time.perf_counter()
            try:
                candidate=fit(views,points[row['connector']],np.array(seed['position']),np.array(seed['rotation']));result['fit']=candidate
                result['scoring_only']=errors(np.array(candidate['position']),np.array(candidate['rotation']),np.array(row['scoring_only']['tip_world']))
                limit=.00025 if row['connector']=='sc' else .0001
                result['view_consistency_passed']=max(candidate['leave_one_view_out_displacement_m'])<=limit
            except (ValueError,RuntimeError) as exc:result['reason']=str(exc)
            result['wall_s']=time.perf_counter()-begin;rows.append(result)
            (out/'contour_results.json').write_text(json.dumps(dict(scope='Static development contour diagnostics, no runtime acceptance/insertion claim',
                final_cases_used=0,production_timestamp_vetoes_preserved=True,rows=rows),indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='fit'}),flush=True)


if __name__=='__main__':main()
