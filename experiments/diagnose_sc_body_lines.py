"""Offline multiview straight-body-edge orientation diagnostic (no commands)."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion import sc_body_refinement as body
from experiments.lastinch_geometry import HOME,robot_fk


def fit(views,p,R):
    pts,_=body.template();prepared=[];centroid=pts.mean(0)
    for v in views:
        image=v['image'];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);white=((hsv[:,:,1]<60)&(hsv[:,:,2]>80)).astype(np.uint8)
        occluded=cv2.dilate(v['tool_mask'],np.ones((3,3),np.uint8))>0
        orange=cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255]))>0
        occluded|=cv2.dilate(orange.astype(np.uint8),np.ones((5,5),np.uint8))>0
        hull=body.project(v['P'],(R@pts.T).T+p)-.5;roi=np.zeros(image.shape[:2],np.uint8)
        cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(hull).astype(np.int32)),255);roi=cv2.dilate(roi,np.ones((9,9),np.uint8))>0
        nearwhite=cv2.dilate(white,np.ones((3,3),np.uint8))>0
        lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))[0]
        axes=[]
        for j in range(3):
            endpoints=np.array([centroid-np.eye(3)[j]*.005,centroid+np.eye(3)[j]*.005]);uv=body.project(v['P'],(R@endpoints.T).T+p)-.5;d=uv[1]-uv[0];axes.append(d/np.linalg.norm(d))
        rows=[]
        for line in ([] if lines is None else lines.reshape(-1,4)):
            uv=line.reshape(2,2);d=uv[1]-uv[0];length=np.linalg.norm(d)
            if length<8:continue
            sample=uv[0]+np.linspace(.1,.9,9)[:,None]*d;xy=np.rint(sample).astype(int);h,w=white.shape
            if np.any(xy<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]>=h):continue
            yy,xx=xy[:,1],xy[:,0]
            if not np.all(roi[yy,xx]&nearwhite[yy,xx]&~occluded[yy,xx]):continue
            angles=np.rad2deg(np.arccos(np.clip(np.abs(np.array(axes)@(d/length)),0,1)));order=np.argsort(angles)
            if angles[order[0]]>8 or angles[order[1]]-angles[order[0]]<3:continue
            # Image raster endpoints -> optical pixel-centre projection coordinates.
            l=np.cross(np.r_[uv[0]+.5,1],np.r_[uv[1]+.5,1]);T=np.linalg.solve(v['K'],v['P']);normal=T[:3,:3].T@v['K'].T@l;normal/=np.linalg.norm(normal)
            rows.append((int(order[0]),normal,length,uv.tolist()))
        prepared.append((v['camera'],rows))
    def optimize(selected):
        def residual(x):
            candidate=Rotation.from_rotvec(np.deg2rad(x)).as_matrix()@R
            return np.array([float(n@candidate[:,axis])*np.sqrt(length) for _,rows in selected for axis,n,length,_ in rows])
        if sum(len(rs) for _,rs in selected)<6:return None
        opt=least_squares(residual,np.zeros(3),bounds=([-5]*3,[5]*3),loss='soft_l1',f_scale=.02,diff_step=1e-4,max_nfev=200)
        return Rotation.from_rotvec(np.deg2rad(opt.x)).as_matrix()@R
    result=optimize(prepared)
    if result is None:return {'reason':'insufficient_lines'}
    loo=[]
    for omit in range(3):
        pair=optimize([v for i,v in enumerate(prepared) if i!=omit]);loo.append(None if pair is None else float(np.rad2deg(Rotation.from_matrix(result.T@pair).magnitude())))
    return {'rotation':result.tolist(),'delta_angle_deg':float(np.rad2deg(Rotation.from_matrix(R.T@result).magnitude())),
            'leave_one_view_out_deg':loo,'lines':{name:[dict(axis=a,normal=n.tolist(),length_px=float(l),uv=uv) for a,n,l,uv in rows] for name,rows in prepared}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--cad-associations',action='store_true');parser.add_argument('--full-silhouette',action='store_true');a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME);datasets=[]
    n='sc_supported_track30v2_original_perception_rim_position_body_track_handoff';folder=ROOT/'.artifacts/lastinch_runs'/n/n/'body_inputs/000';d=json.loads((folder/'inputs.json').read_text());views=[]
    for v in d['views']:
        image=cv2.imread(str(folder/(v['camera']+'.png')));views.append(dict(camera=v['camera'],image=image,K=np.array(v['K']),P=np.array(v['P']),tool_mask=cv2.imread(str(folder/(v['camera']+'_tool_mask.png')),0)))
    datasets.append(dict(run=n,stamp_s=max(v['stamp_s'] for v in d['views']),position=np.array(d['initial_position']),rotation=np.array(d['initial_rotation']),views=views))
    n='dev_3000_cable30v1_original_perception_rim_position_body_track_handoff';r=ROOT/'.artifacts/lastinch_runs'/n/n;geom=json.loads((r/'perception_after/camera_geometry.json').read_text());pred=json.loads((ROOT/'.artifacts/reference30/cable30_after/replay/replay.json').read_text())['conditions'][2];views=[]
    for v in geom:
        c=v['camera'];K=np.array(v['K']);T=np.linalg.inv(v['T_base_from_camera']);C=np.linalg.inv(fk[c+'/optical'])@fk['gripper/tcp'];B=np.linalg.inv(T)@C;image=cv2.imread(str(r/'perception_after'/(c+'_raw.png')))
        views.append(dict(camera=c,image=image,K=K,P=K@T[:3],tool_mask=body.robot_tool_mask(K,T,B,image.shape)))
    datasets.append(dict(run=n,stamp_s=max(v['stamp_s'] for v in geom),position=np.array(pred['position']),rotation=np.array(pred['rotation']),views=views))
    data=ROOT/'.artifacts/perception_data29/qualified';predictions=json.loads((ROOT/'.artifacts/perception_data29/evaluation/mouth_transfer_original_plug_multiview_val_without_truth.json').read_text())
    for pred in predictions:
        case=pred['case_id'];capture=json.loads((data/'cases'/case/'training_capture.json').read_text());views=[]
        for c,v in capture['cameras'].items():
            K=np.array(v['K']);T=np.linalg.inv(v['T_world_from_camera']);C=np.linalg.inv(fk[c+'/optical'])@fk['gripper/tcp'];B=np.linalg.inv(T)@C;image=cv2.imread(str(data/'dataset/sc_plug/images/val'/(case+'_'+c+'.png')))
            views.append(dict(camera=c,image=image,K=K,P=K@T[:3],tool_mask=body.robot_tool_mask(K,T,B,image.shape)))
        datasets.append(dict(case_id=case,position=np.array(pred['plug_position']),rotation=np.array(pred['plug_rotation']),views=views))
    rows=[]
    fitter=fit
    if a.cad_associations:
        from sc_cad_line_diagnostic import fit as fitter
    if a.full_silhouette:
        from sc_silhouette_pose_diagnostic import fit as fitter
    for d in datasets:
        result=fitter(d['views'],d['position'],d['rotation']);result.update({k:v for k,v in d.items() if k in ['run','case_id','stamp_s']});rows.append(result)
    (a.out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0]
    for r in rows:
        if 'rotation' not in r:continue
        if 'run' in r:
            trace=json.load(gzip.open(ROOT/'.artifacts/lastinch_runs'/r['run']/r['run']/'verification_trace.json.gz','rt'));actual=min(trace,key=lambda x:abs(x['sim_time_s']-r['stamp_s']));truth=np.linalg.inv(B)@actual['tip_world']
        else:truth=np.array(json.loads((data/'cases'/r['case_id']/'scene.json').read_text())['reset']['tip_world'])
        r['diagnostic_angle_deg']=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@r['rotation']).magnitude()))
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k not in ['rotation','lines']} for r in rows],indent=2))


if __name__=='__main__':main()
