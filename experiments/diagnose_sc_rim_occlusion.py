"""Replay recorded rim fits with a cable-occlusion negative control.

Fits and acceptance are saved before ground truth is opened for scoring.
No policy code or live simulator is modified by this diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion import sc_rim_refinement as rim


def occlusion_edges(image, include_cable):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blue=cv2.inRange(hsv,np.array([90,80,60]),np.array([130,255,255]))
    edge=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),50,100)
    edge &= cv2.dilate(blue,np.ones((3,3),np.uint8))
    white=(hsv[:,:,1]<60)&(hsv[:,:,2]>80)
    orange=cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255]))!=0
    occluded=cv2.dilate((white|(orange if include_cable else False)).astype(np.uint8)*255,np.ones((5,5),np.uint8))
    edge[occluded!=0]=0
    return cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float),occluded


def white_only_edges(image):
    return occlusion_edges(image, False)


def cable_aware_edges(image):
    return occlusion_edges(image, True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--limit',type=int,default=2);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();archive=ROOT/'.artifacts/lastinch_runs'/a.run;path=archive/a.run;a.out.mkdir(parents=True,exist_ok=True)
    original=rim.edge_distance;rows=[]
    for folder in sorted((path/'rim_inputs').iterdir())[:a.limit]:
        inputs=json.loads((folder/'inputs.json').read_text())
        views=[{'camera':v['camera'],'P':np.array(v['P']),'image':cv2.imread(str(folder/f'{v["camera"]}.png'))} for v in inputs['views']]
        sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.iterdir() if f.is_file()}
        for mode,implementation in [('original_white_only',white_only_edges),('white_and_orange_cable',cable_aware_edges)]:
            rim.edge_distance=implementation
            refined,diagnostic=rim.refine_with_agreement(views,np.array(inputs['initial_position']),np.array(inputs['initial_rotation']),raster_offset_px=-.5)
            row={'run':a.run,'capture':folder.name,'mode':mode,'diagnostic':diagnostic,'accepted':refined is not None,
                 'source_sha256':sources,'GT_used_for_fitting_or_acceptance':False}
            if refined is not None:row.update(position=refined[0].tolist(),rotation=refined[1].tolist())
            rows.append(row)
    rim.edge_distance=original
    (a.out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Diagnostic scorer only, after every image-only fit/decision is persisted.
    world=np.array(json.loads((archive/'scene.json').read_text())['reset']['mouth_world'])
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0.]
    truth=np.linalg.inv(base)@world
    for row in rows:
        if 'position' in row:
            row['diagnostic_position_error_mm']=float(np.linalg.norm(row['position']-truth[:3,3])*1000)
            row['diagnostic_angle_error_deg']=float(Rotation.from_matrix(truth[:3,:3].T@np.array(row['rotation'])).magnitude()*180/np.pi)
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k not in ['source_sha256','position','rotation']} for row in rows],indent=2))


if __name__=='__main__':main()
