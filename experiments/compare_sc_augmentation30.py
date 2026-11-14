"""Frozen original-network ensemble audit; fits saved before diagnostic truth."""
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
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model'),str(ROOT/'experiments')]
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import PlugPoseView,fuse_multiview_keypoints
from experiments.sc_keypoint_ensemble_diagnostic import collect,combine,AUGMENTATIONS
from experiments.summarize_visual_screen30 import NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    weight=ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt';est=ScPlugPoseEstimator(weight,device=a.device,crop_refine=True,crop_pad_scale=6,refinement_strategy='fixed');datasets=[]
    data=ROOT/'.artifacts/perception_data29/qualified';manifest=json.loads((data/'manifest.json').read_text())
    for case in manifest['cases']:
        if case['split']!='val':continue
        name=case['case_id'];capture=json.loads((data/'cases'/name/'training_capture.json').read_text());views=[]
        for c,v in capture['cameras'].items():
            image=cv2.imread(str(data/'dataset/sc_plug/images/val'/(name+'_'+c+'.png')));views.append(PlugPoseView(c,image,np.array(v['K']),np.array(v['T_world_from_camera']),v['image_stamp_s'],c))
        datasets.append(dict(case_id=name,views=views))
    for n in NAMES:
        root=ROOT/'.artifacts/lastinch_runs'/n/n
        for capture in ['perception_before','perception_after']:
            if 'dev_3006_' in n and capture=='perception_after':continue
            geometry=json.loads((root/capture/'camera_geometry.json').read_text());views=[]
            for v in geometry:
                c=v['camera'];image=cv2.imread(str(root/capture/(c+'_raw.png')));views.append(PlugPoseView(c,image,np.array(v['K']),np.array(v['T_base_from_camera']),v['stamp_s'],c))
            datasets.append(dict(run=n,capture=capture,views=views))
    rows=[]
    for item in datasets:
        groups=collect(est,item['views'])
        for mode,indices in [('original',[0]),('pixel_phase',[0,1,2,3,4]),('rotation_equivariance',[0,5,6]),('combined',list(range(7)))]:
            row={k:v for k,v in item.items() if k!='views'};row.update(mode=mode,GT_used_for_inference=False,camera_stamps_s=[v.stamp_s for v in item['views']])
            try:
                ds=groups[0] if mode=='original' else combine(groups,indices);pos,R,shape,error,count,conf=fuse_multiview_keypoints(item['views'],ds,local_keypoints_m=est.local_keypoints_m)
                row.update(position=pos.tolist(),rotation=R.tolist(),shape_mm=shape*1000,reprojection_px=error,original_gates_passed=shape<=.0035 and error<=6)
            except ValueError as e:row['reason']=str(e)
            rows.append(row)
        print(json.dumps(dict(scene=item.get('case_id',item.get('run')),capture=item.get('capture'),fitted=True)),flush=True)
        (a.out/'fits_without_truth.json').write_text(json.dumps(dict(augmentations=AUGMENTATIONS,weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
    B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0]
    for r in rows:
        if 'position' not in r:continue
        if 'case_id' in r:truth=np.array(json.loads((data/'cases'/r['case_id']/'scene.json').read_text())['reset']['tip_world'])
        else:
            trace=json.load(gzip.open(ROOT/'.artifacts/lastinch_runs'/r['run']/r['run']/'verification_trace.json.gz','rt'));actual=min(trace,key=lambda x:abs(x['sim_time_s']-max(r['camera_stamps_s'])));truth=np.linalg.inv(B)@actual['tip_world']
        r.update(diagnostic_position_mm=float(np.linalg.norm(r['position']-truth[:3,3])*1000),diagnostic_angle_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@r['rotation']).magnitude())))
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({m:{split:float(np.median([r['diagnostic_angle_deg'] for r in rows if r['mode']==m and ('case_id' in r)==(split=='static') and 'diagnostic_angle_deg' in r])) for split in ['static','recorded_live']} for m in ['original','pixel_phase','rotation_equivariance','combined']},indent=2))


if __name__=='__main__':main()
