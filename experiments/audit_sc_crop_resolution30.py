"""Offline original-weight crop audit; training-only choice, no live mutation.

Saves image-only predictions before reading scoring truth. The crop scale is
chosen using training scenes only. Static and recorded-live development scores
then assess that frozen choice; none are held-out final insertion trials.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import PlugPoseView,fuse_multiview_keypoints
from experiments.summarize_visual_screen30 import NAMES

SCALES=(None,1.5,2.,3.,4.,6.,8.)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise RuntimeError('Choose a new output; preserve prior evidence')
    a.out.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    weight=ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt'
    est=ScPlugPoseEstimator(weight,device='cuda:0',crop_refine=True,crop_pad_scale=6,refinement_strategy='fixed')
    data=ROOT/'.artifacts/perception_data29/qualified';manifest=json.loads((data/'manifest.json').read_text());datasets=[]
    for case in manifest['cases']:
        name=case['case_id'];split=case['split']
        if split not in ('train','val'):raise ValueError('No final/test data accepted')
        capture=json.loads((data/'cases'/name/'training_capture.json').read_text());views=[]
        for c,v in capture['cameras'].items():
            image=cv2.imread(str(data/'dataset/sc_plug/images'/split/(name+'_'+c+'.png')))
            views.append(PlugPoseView(c,image,np.array(v['K']),np.array(v['T_world_from_camera']),v['image_stamp_s'],c))
        datasets.append(dict(case_id=name,split=split,views=views))
    for n in NAMES:
        root=ROOT/'.artifacts/lastinch_runs'/n/n
        for capture in ['perception_before','perception_after']:
            if 'dev_3006_' in n and capture=='perception_after':continue
            geometry=json.loads((root/capture/'camera_geometry.json').read_text());views=[]
            for v in geometry:
                c=v['camera'];image=cv2.imread(str(root/capture/(c+'_raw.png')))
                views.append(PlugPoseView(c,image,np.array(v['K']),np.array(v['T_base_from_camera']),v['stamp_s'],c))
            datasets.append(dict(run=n,capture=capture,split='recorded_live_development',views=views))
    rows=[]
    for item in datasets:
        for scale in SCALES:
            est.crop_refine=scale is not None;est.crop_pad_scale=scale or 6
            row={k:v for k,v in item.items() if k!='views'}
            row.update(crop_scale=scale,simulator_inputs_for_prediction=False,camera_stamps_s=[v.stamp_s for v in item['views']])
            try:
                ds=est.detect_views(item['views'])
                row['detections']=[dict(camera=d.camera_name,keypoints=d.keypoints_px.tolist(),confidence=d.keypoint_confidences.tolist()) for d in ds]
                pos,R,shape,reproj,count,conf=fuse_multiview_keypoints(item['views'],ds,local_keypoints_m=est.local_keypoints_m)
                row.update(position=pos.tolist(),rotation=R.tolist(),shape_mm=shape*1000,reprojection_px=reproj,original_gates_passed=shape<=.0035 and reproj<=6)
            except ValueError as e:row['reason']=str(e)
            rows.append(row)
        (a.out/'fits_without_truth.json').write_text(json.dumps(dict(weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),scales=SCALES,rows=rows),indent=2)+'\n')
        print(json.dumps(dict(scene=item.get('case_id',item.get('run')),capture=item.get('capture'),completed=True)),flush=True)
    B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0]
    # Supervised training-only selection is saved before any development truth.
    def score(r):
        if 'position' not in r:return
        if 'case_id' in r:truth=np.array(json.loads((data/'cases'/r['case_id']/'scene.json').read_text())['reset']['tip_world'])
        else:
            trace=json.load(gzip.open(ROOT/'.artifacts/lastinch_runs'/r['run']/r['run']/'verification_trace.json.gz','rt'))
            actual=min(trace,key=lambda x:abs(x['sim_time_s']-max(r['camera_stamps_s'])))
            truth=np.linalg.inv(B)@actual['tip_world']
        r.update(position_error_mm=float(np.linalg.norm(np.array(r['position'])-truth[:3,3])*1000),angle_error_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@r['rotation']).magnitude())))
    for r in rows:
        if r['split']=='train':score(r)
    train={str(s):dict(median_angle_deg=float(np.median([r.get('angle_error_deg',180) for r in rows if r['split']=='train' and r['crop_scale']==s])),gated_count=sum(r.get('original_gates_passed',False) for r in rows if r['split']=='train' and r['crop_scale']==s)) for s in SCALES}
    chosen=min(SCALES,key=lambda s:train[str(s)]['median_angle_deg'])
    choice=dict(training_scores=train,selected_scale=chosen,selection='training median rotation error; inference failures penalized 180 deg',final_100000_series_used=False,production_enabled=False)
    (a.out/'training_only_selection.json').write_text(json.dumps(choice,indent=2)+'\n')
    for r in rows:
        if r['split']!='train':score(r)
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary={str(s):{split:dict(median_angle_deg=float(np.median([r.get('angle_error_deg',180) for r in rows if r['split']==split and r['crop_scale']==s])),median_position_mm=float(np.median([r.get('position_error_mm',1000) for r in rows if r['split']==split and r['crop_scale']==s]))) for split in ['train','val','recorded_live_development']} for s in SCALES}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(choice=choice,summary=summary),indent=2))


if __name__=='__main__':main()
