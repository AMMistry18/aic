"""Static 3D plug-pose audit; all predictions saved before truth scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import PlugPoseView,fuse_multiview_keypoints


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    root=ROOT/'.artifacts/perception_data29/qualified';manifest=json.loads((root/'manifest.json').read_text());est=ScPlugPoseEstimator(a.weights,device='cpu',crop_refine=True,crop_pad_scale=6,refinement_strategy='fixed');rows=[]
    for case in manifest['cases']:
        if case['split']!='val':continue
        name=case['case_id'];capture=json.loads((root/'cases'/name/'training_capture.json').read_text());views=[]
        for c,v in capture['cameras'].items():
            image=cv2.imread(str(root/'dataset/sc_plug/images/val'/(name+'_'+c+'.png')));views.append(PlugPoseView(c,image,np.array(v['K']),np.array(v['T_world_from_camera']),v['image_stamp_s'],c))
        row=dict(case_id=name,GT_used_for_inference=False)
        try:
            ds=est.detect_views(views);pos,R,shape,reproj,count,conf=fuse_multiview_keypoints(views,ds,local_keypoints_m=est.local_keypoints_m)
            row.update(position=pos.tolist(),rotation=R.tolist(),shape_mm=shape*1000,reprojection_px=reproj,original_gates_passed=shape<=.0035 and reproj<=6)
        except ValueError as e:row['reason']=str(e)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['rotation','position']}),flush=True)
    (a.out/'fits_without_truth.json').write_text(json.dumps(dict(weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
    for r in rows:
        if 'position' not in r:continue
        truth=np.array(json.loads((root/'cases'/r['case_id']/'scene.json').read_text())['reset']['tip_world'])
        r.update(position_error_mm=float(np.linalg.norm(r['position']-truth[:3,3])*1000),angle_error_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@r['rotation']).magnitude())))
    summary=dict(static_development_scenes=len(rows),median_position_error_mm=float(np.median([r['position_error_mm'] for r in rows if 'position_error_mm' in r])),median_angle_error_deg=float(np.median([r['angle_error_deg'] for r in rows if 'angle_error_deg' in r])),not_physical_insertion=True,final_seeds_used=False,rows=rows)
    (a.out/'diagnostic_scores.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':main()
