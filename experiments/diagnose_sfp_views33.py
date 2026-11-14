"""Original NIC weights: reversible image rotations and native tiles, no GT."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_example_policies'))
from aic_example_policies.ros.perception_core import PerceptionCore


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    folder=ROOT/'.artifacts/lastinch_runs/dev_44000_sfpvisual33v2_original_perception/dev_44000_sfpvisual33v2_original_perception/perception_before'
    out=ROOT/'.artifacts/sfp_visual33/view_search';out.mkdir(parents=True,exist_ok=True)
    detector=PerceptionCore(nic_weights=str(ROOT/'aic_example_policies/aic_example_policies/ros/weights/best.pt'))
    rows=[]
    for name in ['left_camera','center_camera','right_camera']:
        image=cv2.imread(str(folder/(name+'_raw.png')));h,w=image.shape[:2]
        table=np.rint(255*np.sqrt(np.arange(256)/255)).astype(np.uint8)
        for gamma in [False,True]:
            source=cv2.LUT(image,table) if gamma else image
            for k in range(4):
                pixels=np.ascontiguousarray(np.rot90(source,k))
                detections=detector.detect_nic(pixels,conf_thresh=.2)
                for d in detections:
                    xy=d['kps'].copy()
                    if k==1:xy=np.c_[w-1-xy[:,1],xy[:,0]]
                    elif k==2:xy=np.c_[w-1-xy[:,0],h-1-xy[:,1]]
                    elif k==3:xy=np.c_[xy[:,1],h-1-xy[:,0]]
                    d['kps']=xy
                rows.append(dict(camera=name,condition=f'gamma{gamma}_rot{k}',detections=detections))
            for x0,y0 in [(0,0),(w//3,0),(0,h//3),(w//3,h//3)]:
                pixels=source[y0:y0+2*h//3,x0:x0+2*w//3]
                detections=detector.detect_nic(pixels,conf_thresh=.2)
                for d in detections:d['kps']+=np.array([x0,y0])
                rows.append(dict(camera=name,condition=f'gamma{gamma}_tile{x0}_{y0}',detections=detections))
        print(name,[(r['condition'],len(r['detections'])) for r in rows if r['camera']==name],flush=True)
    (out/'detections_without_truth.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')


if __name__=='__main__':main()
