"""Original NIC checkpoint on RGB-only photometric variants; no object truth."""
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
    out=ROOT/'.artifacts/sfp_visual33/radiometry';out.mkdir(parents=True,exist_ok=True)
    detector=PerceptionCore(nic_weights=str(ROOT/'aic_example_policies/aic_example_policies/ros/weights/best.pt'))
    rows=[]
    for name in ['left_camera','center_camera','right_camera']:
        image=cv2.imread(str(folder/(name+'_raw.png')))
        variants={'raw':image}
        for gamma in [.5,.33,.25]:
            table=np.rint(255*(np.arange(256)/255)**gamma).astype(np.uint8)
            variants['gamma'+str(gamma)]=cv2.LUT(image,table)
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0]);variants['clahe']=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
        for variant,pixels in variants.items():
            detections=detector.detect_nic(pixels)
            row=dict(camera=name,variant=variant,count=len(detections),detections=detections)
            rows.append(row);print(json.dumps(dict(camera=name,variant=variant,count=len(detections),conf=[d['conf'] for d in detections])),flush=True)
            annotated=pixels.copy()
            for d in detections:
                for k,p in enumerate(d['kps']):
                    point=tuple(np.rint(p).astype(int));cv2.circle(annotated,point,3,(0,0,255),-1);cv2.putText(annotated,str(k),point,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
            cv2.imwrite(str(out/(name+'_'+variant+'.jpg')),annotated)
    (out/'detections_without_truth.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')


if __name__=='__main__':main()
