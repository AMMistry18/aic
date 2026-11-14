"""Frozen-network NIC appearance diagnostic; no robot motion or pose fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'aic_example_policies'))
from aic_example_policies.ros.perception_core import PerceptionCore


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    weights=ROOT/'aic_example_policies/aic_example_policies/ros/weights/best.pt'
    pc=PerceptionCore(nic_weights=str(weights))
    rows=[]
    for path in sorted(a.images.glob('*_camera_raw.png')):
        original=cv2.imread(str(path))
        conditions=[('original',original)]
        for gamma in [1.5,2.,3.]:
            lut=np.rint((np.arange(256)/255.)**(1/gamma)*255).astype(np.uint8)
            conditions.append((f'gamma_{gamma}',cv2.LUT(original,lut)))
        lab=cv2.cvtColor(original,cv2.COLOR_BGR2LAB)
        lab[:,:,0]=cv2.createCLAHE(2.,(8,8)).apply(lab[:,:,0])
        conditions.append(('clahe',cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)))
        for name,img in conditions:
            detections=pc.detect_nic(img,conf_thresh=.2)
            rows.append({'camera':path.stem,'condition':name,'detections':detections})
            overlay=img.copy()
            for d in detections:
                for k,(x,y) in enumerate(d['kps']):
                    cv2.circle(overlay,(round(float(x)),round(float(y))),3,(0,255,255),-1)
            cv2.imwrite(str(a.out/f'{path.stem}_{name}.jpg'),overlay)
    result={'purpose':'development appearance diagnosis, detection counts are not pose accuracy',
            'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'rows':rows}
    (a.out/'detections.json').write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))
    print(json.dumps([{k:r[k] for k in ['camera','condition']}|{'count':len(r['detections'])} for r in rows],indent=2))


if __name__=='__main__':main()
