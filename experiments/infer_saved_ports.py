"""Run existing neural port models on saved pixels; no simulator or ROS access."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import torch
import ultralytics
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pattern',default='*_camera_raw.png');a=p.parse_args()
    weights=Path('/research/aic_example_policies/aic_example_policies/ros/weights');a.out.mkdir(parents=True,exist_ok=True)
    images=sorted(a.images.glob(a.pattern));rows=[]
    if not images:raise ValueError('No matching saved camera images')
    for model_name in ('best_sc_mouth_pose.pt','best.pt'):
        path=weights/model_name;model=YOLO(str(path))
        for image_path in images:
            image=cv2.imread(str(image_path));start=time.monotonic()
            result=model(image,conf=.25,imgsz=960,device=0,verbose=False)[0];torch.cuda.synchronize()
            row={'image':image_path.name,'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                'parameters':{'conf':.25,'imgsz':960,'device':0},
                'model':model_name,'weight_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'ultralytics':ultralytics.__version__,'names':model.names,'wall_s_including_first_call_warmup':time.monotonic()-start,'detections':[]}
            if result.keypoints is not None and result.boxes is not None:
                xy=result.keypoints.xy.cpu().numpy();confidence=result.keypoints.conf.cpu().numpy()
                for i,box in enumerate(result.boxes):
                    row['detections'].append({'bbox':box.xyxy[0].cpu().tolist(),'box_conf':box.conf.item(),'class':int(box.cls.item()),
                        'keypoints_px':xy[i].tolist(),'confidence':confidence[i].tolist()})
            cv2.imwrite(str(a.out/f'{model_name}_{image_path.stem}_prediction.png'),result.plot());rows.append(row)
    (a.out/'predictions.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{'image':r['image'],'model':r['model'],'detections':len(r['detections'])} for r in rows],indent=2))

if __name__=='__main__':main()
