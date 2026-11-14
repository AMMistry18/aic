"""Offline free-rotation body-fit diagnostic; never commands a robot.

All fits are persisted before diagnostic truth scoring. This investigates the
learned-orientation veto, not a deployed replacement or certified angular bound.
"""
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


def fit(views,position,rotation):
    points,_=body.template()
    try:prepared=body.observations(views,points,position,rotation)
    except ValueError as e:return {'reason':str(e)}
    def pose(x):return position+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@rotation
    def optimize(selected):
        def residual(x):
            p,R=pose(x);world=(R@points.T).T+p
            return np.concatenate([np.max((body.project(P,world)-.5)@D.T,axis=0)-s for P,D,s in selected])
        candidates=[least_squares(residual,[x,y,0,0,0,0],bounds=([-3]*3+[-5]*3,[3]*3+[5]*3),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=200)
                    for x,y in [(0,0),(-.5,0),(.5,0),(0,-.5),(0,.5)]]
        best=min(candidates,key=lambda x:x.cost);p,R=pose(best.x)
        return p,R,{'delta_mm_deg':best.x.tolist(),'mean_support_error_px':float(np.mean(abs(residual(best.x)))),'optimizer_success':bool(best.success)}
    p,R,d=optimize(prepared);pairs=[]
    for omit in range(3):
        pp,RR,dd=optimize([v for i,v in enumerate(prepared) if i!=omit])
        pairs.append(dict(dd,position_disagreement_mm=float(np.linalg.norm(pp-p)*1000),rotation_disagreement_deg=float(np.rad2deg(Rotation.from_matrix(R.T@RR).magnitude()))))
    return dict(d,position=p.tolist(),rotation=R.tolist(),prior_mismatch_deg=float(np.rad2deg(Rotation.from_matrix(rotation.T@R).magnitude())),leave_one_view_out=pairs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME)
    rows=[]
    # Recorded positive nominal images, exact exposure-time robot self masks.
    n='sc_supported_track30v2_original_perception_rim_position_body_track_handoff'
    folder=ROOT/'.artifacts/lastinch_runs'/n/n/'body_inputs/000';data=json.loads((folder/'inputs.json').read_text());views=[]
    for v in data['views']:
        image=cv2.imread(str(folder/(v['camera']+'.png')))
        views.append(dict(camera=v['camera'],image=image,P=np.array(v['P']),tool_mask=cv2.imread(str(folder/(v['camera']+'_tool_mask.png')),0)))
    rows.append(dict(run=n,camera_stamp_s=[v['stamp_s'] for v in data['views']],**fit(views,np.array(data['initial_position']),np.array(data['initial_rotation']))))
    # New varied-case final observation. Only camera/robot calibration enters fits.
    n='dev_3000_cable30v1_original_perception_rim_position_body_track_handoff';r=ROOT/'.artifacts/lastinch_runs'/n/n
    geometry=json.loads((r/'perception_after/camera_geometry.json').read_text());policy=json.loads((r/'policy.json').read_text())
    replay=json.loads((ROOT/'.artifacts/reference30/cable30_after/replay/replay.json').read_text())
    initial=np.array(replay['conditions'][2]['position']);prior=np.array(policy['sc_rim_diagnostics'][-1]['rotation']);views=[]
    for v in geometry:
        camera=v['camera'];K=np.array(v['K']);T=np.linalg.inv(v['T_base_from_camera'])
        camera_from_tcp=np.linalg.inv(fk[camera+'/optical'])@fk['gripper/tcp'];B=np.linalg.inv(T)@camera_from_tcp
        image=cv2.imread(str(r/'perception_after'/(camera+'_raw.png')))
        views.append(dict(camera=camera,image=image,P=K@T[:3],tool_mask=body.robot_tool_mask(K,T,B,image.shape)))
    rows.append(dict(run=n,camera_stamp_s=[v['stamp_s'] for v in geometry],**fit(views,initial,prior)))
    # Previously used static development negative controls with angular perturbation.
    preds=json.loads((ROOT/'.artifacts/perception_data29/evaluation/mouth_transfer_original_plug_multiview_val_without_truth.json').read_text())
    priors={r['case_id']:r for r in json.loads((ROOT/'.artifacts/reference30/body_prior_violation/fits_without_truth.json').read_text())}
    for pred in preds:
        case=pred['case_id'];root=ROOT/'.artifacts/perception_data29/qualified';capture=json.loads((root/'cases'/case/'training_capture.json').read_text());views=[]
        for camera,v in capture['cameras'].items():
            K=np.array(v['K']);T=np.linalg.inv(v['T_world_from_camera']);C=np.linalg.inv(fk[camera+'/optical'])@fk['gripper/tcp'];B=np.linalg.inv(T)@C
            image=cv2.imread(str(root/'dataset/sc_plug/images/val'/(case+'_'+camera+'.png')))
            views.append(dict(camera=camera,image=image,P=K@T[:3],tool_mask=body.robot_tool_mask(K,T,B,image.shape)))
        rows.append(dict(case_id=case,**fit(views,np.array(pred['plug_position']),np.array(priors[case]['prior_rotation']))))
    (args.out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0]
    for row in rows:
        if 'position' not in row:continue
        if 'run' in row:
            trace=json.load(gzip.open(ROOT/'.artifacts/lastinch_runs'/row['run']/row['run']/'verification_trace.json.gz','rt'))
            actual=min(trace,key=lambda v:abs(v['sim_time_s']-max(row['camera_stamp_s'])))
            truth=np.linalg.inv(B)@np.array(actual['tip_world'])
        else:truth=np.array(json.loads((ROOT/'.artifacts/perception_data29/qualified/cases'/row['case_id']/'scene.json').read_text())['reset']['tip_world'])
        row.update(diagnostic_position_mm=float(np.linalg.norm(np.array(row['position'])-truth[:3,3])*1000),diagnostic_angle_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@np.array(row['rotation'])).magnitude())))
    (args.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ['rotation','position','leave_one_view_out','camera_stamp_s']} for r in rows],indent=2))


if __name__=='__main__':main()
