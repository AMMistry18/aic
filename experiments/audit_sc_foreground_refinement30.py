"""Offline qualification of context-suppressed refinement; truth scored last."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import PlugPoseView,fuse_multiview_keypoints
from experiments.lastinch_geometry import HOME,robot_fk
from experiments.sc_foreground_refinement30 import detect
from experiments.summarize_visual_screen30 import NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rigid-anchor',action='store_true');p.add_argument('--left-right-only',action='store_true');p.add_argument('--pixel-phase-ensemble',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve existing audit')
    a.out.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    weight=ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt';est=ScPlugPoseEstimator(weight,device='cuda:0',crop_refine=False);refiner=YOLO(str(a.weights))
    fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME);C={c:np.linalg.inv(fk[c+'/optical'])@fk['gripper/tcp'] for c in ['left_camera','center_camera','right_camera']}
    data=ROOT/'.artifacts/perception_data29/qualified';manifest=json.loads((data/'manifest.json').read_text());datasets=[]
    for case in manifest['cases']:
        if case['split']!='val':continue
        name=case['case_id'];capture=json.loads((data/'cases'/name/'training_capture.json').read_text());views=[]
        for c,v in capture['cameras'].items():
            path=data/'dataset/sc_plug/images/val'/(name+'_'+c+'.png');image=cv2.imread(str(path));views.append(PlugPoseView(c,image,np.array(v['K']),np.array(v['T_world_from_camera']),v['image_stamp_s'],c))
        datasets.append(dict(case_id=name,split='static_development',views=views))
    for n in NAMES:
        root=ROOT/'.artifacts/lastinch_runs'/n/n
        for capture in ['perception_before','perception_after']:
            if 'dev_3006_' in n and capture=='perception_after':continue
            geometry=json.loads((root/capture/'camera_geometry.json').read_text());views=[]
            for v in geometry:
                c=v['camera'];image=cv2.imread(str(root/capture/(c+'_raw.png')));views.append(PlugPoseView(c,image,np.array(v['K']),np.array(v['T_base_from_camera']),v['stamp_s'],c))
            datasets.append(dict(run=n,capture=capture,split='recorded_live_development',views=views))
    rows=[];anchors_by_run={}
    for item in datasets:
        row={k:v for k,v in item.items() if k!='views'};row.update(simulator_truth_for_inference=False,camera_stamps_s=[v.stamp_s for v in item['views']])
        try:
            anchor=anchors_by_run.get(item.get('run')) if a.rigid_anchor else None
            ds,diagnostic=detect(est,refiner,item['views'],C,anchors=anchor,pixel_phase_ensemble=a.pixel_phase_ensemble)
            if a.rigid_anchor and item.get('run') and item.get('capture')=='perception_before':
                anchors_by_run[item['run']]={d['camera']:d['refinement_proposal_box'] for d in diagnostic}
            selected_views=[v for v in item['views'] if not a.left_right_only or v.camera_name!='center_camera']
            selected_detections=[d for d in ds if not a.left_right_only or d.camera_name!='center_camera']
            pos,R,shape,error,count,conf=fuse_multiview_keypoints(selected_views,selected_detections,local_keypoints_m=est.local_keypoints_m)
            row.update(position=pos.tolist(),rotation=R.tolist(),shape_mm=shape*1000,reprojection_px=error,original_geometry_gates_passed=shape<=.0035 and error<=6,preprocessing=diagnostic,
                detections=[dict(camera=d.camera_name,keypoints_px=d.keypoints_px.tolist(),confidence=d.keypoint_confidences.tolist()) for d in ds])
            own_tcp=item['views'][0].T_world_from_camera@C[item['views'][0].camera_name];pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=pos;row['estimated_grasp_tcp']=(np.linalg.inv(own_tcp)@pose).tolist()
        except ValueError as e:row['failure']=str(e)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k in ['run','capture','case_id','shape_mm','reprojection_px','original_geometry_gates_passed','failure']}),flush=True)
        (a.out/'fits_without_truth.json').write_text(json.dumps(dict(original_weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),refiner_weight_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
    invariance=[]
    for name in NAMES:
        pair=[r for r in rows if r.get('run')==name]
        if len(pair)!=2:continue
        if any('estimated_grasp_tcp' not in r for r in pair):invariance.append(dict(run=name,passed=False,reason='missing_estimate'));continue
        G0,G1=[np.array(r['estimated_grasp_tcp']) for r in pair];translation=float(np.linalg.norm(G0[:3,3]-G1[:3,3])*1000);angle=float(np.rad2deg(Rotation.from_matrix(G0[:3,:3].T@G1[:3,:3]).magnitude()))
        invariance.append(dict(run=name,translation_change_mm=translation,rotation_change_deg=angle,passed=translation<=.25 and angle<=.25))
    (a.out/'image_only_invariance.json').write_text(json.dumps(invariance,indent=2)+'\n')
    B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0]
    for r in rows:
        if 'position' not in r:continue
        if 'case_id' in r:truth=np.array(json.loads((data/'cases'/r['case_id']/'scene.json').read_text())['reset']['tip_world'])
        else:
            trace=json.load(gzip.open(ROOT/'.artifacts/lastinch_runs'/r['run']/r['run']/'verification_trace.json.gz','rt'));near=min(trace,key=lambda x:abs(x['sim_time_s']-max(r['camera_stamps_s'])));truth=np.linalg.inv(B)@near['tip_world']
        r.update(position_error_mm=float(np.linalg.norm(np.array(r['position'])-truth[:3,3])*1000),angle_error_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@r['rotation']).magnitude())))
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    static=[r for r in rows if r['split']=='static_development'];live=[r for r in rows if r['split']=='recorded_live_development']
    baseline=[r for r in json.loads((ROOT/'.artifacts/reference30/crop_resolution30/diagnostic_scores.json').read_text()) if r['split']=='val' and r['crop_scale']==6.]
    summary=dict(rigid_crop_anchor=a.rigid_anchor,left_right_only=a.left_right_only,pixel_phase_ensemble=a.pixel_phase_ensemble,static_scenes=len(static),static_gated=sum(r.get('original_geometry_gates_passed',False) for r in static),
        median_static_position_mm=float(np.median([r.get('position_error_mm',1000) for r in static])),
        median_static_angle_deg=float(np.median([r.get('angle_error_deg',180) for r in static])),
        max_static_angle_deg=max(r.get('angle_error_deg',180) for r in static),
        median_recorded_live_angle_deg=float(np.median([r.get('angle_error_deg',180) for r in live])),
        invariance_passed=sum(r['passed'] for r in invariance),invariance_pairs=len(invariance),
        protocol_sha256=hashlib.sha256((ROOT/'experiments/configs/foreground_refinement30.json').read_bytes()).hexdigest(),not_physical_insertion=True,production_enabled=False,final_cases_used=False)
    summary['candidate_qualifies_for_development_physical_trial']=bool(summary['static_gated']==6 and summary['median_static_angle_deg']<np.median([r['angle_error_deg'] for r in baseline]) and summary['max_static_angle_deg']<=max(r['angle_error_deg'] for r in baseline) and len(invariance)==4 and all(r['passed'] for r in invariance))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
