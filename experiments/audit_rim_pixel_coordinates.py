"""Offline calibration diagnostic: compare true CAD front edges with RGB pixels.

This explicitly consumes evaluator truth. It neither changes camera calibration
nor supplies object poses to any production policy.
"""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.diagnose_sc_rim import edge_distance, project


def main():
    data=ROOT/'.artifacts/perception_data29/qualified'
    cases=json.loads((data/'manifest.json').read_text())['cases']
    points=np.load(ROOT/'experiments/configs/sc_front_rim.npz')['points_m'];rows=[]
    for case in cases:
        folder=data/'cases'/case['case_id'];capture=json.loads((folder/'training_capture.json').read_text())
        truth=np.array(json.loads((folder/'scene.json').read_text())['reset']['mouth_world'])
        world=(truth[:3,:3]@points.T).T+truth[:3,3]
        for name,v in capture['cameras'].items():
            img=cv2.imread(str(data/'dataset/sc_plug/images'/case['split']/f'{case["case_id"]}_{name}.png'))
            field,occluded=edge_distance(img)
            uv=project(np.array(v['K'])@np.linalg.inv(v['T_world_from_camera'])[:3],world)
            visible=map_coordinates(occluded,uv[:,::-1].T,order=0,mode='constant',cval=255)==0
            def residual(delta):
                return np.minimum(map_coordinates(field,(uv[visible]+delta)[:,::-1].T,order=1,mode='constant',cval=20),12)
            fit=least_squares(residual,[0.,0.],bounds=([-2,-2],[2,2]),loss='soft_l1',f_scale=1,diff_step=1e-4)
            rows.append({'case':case['case_id'],'split':case['split'],'camera':name,
                         'pixel_offset':fit.x.tolist(),'mean_error_px':float(np.mean(residual(fit.x)))})
    out=ROOT/'.artifacts/reference30/pixel_coordinate_audit.json'
    summary={}
    for split in ['train','val']:
        summary[split]={}
        for camera in sorted({r['camera'] for r in rows}):
            group=[r['pixel_offset'] for r in rows if r['split']==split and r['camera']==camera]
            summary[split][camera]={'median_offset_px':np.median(group,axis=0).tolist(),'range_px':np.ptp(group,axis=0).tolist()}
    out.write_text(json.dumps({'purpose':__doc__,'summary':summary,'rows':rows},indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
