"""Development image metrics, matched after inference; not insertion success."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO


def iou(a,b):
    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));intersection=overlap.prod()
    return intersection/max(1e-9,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);model=YOLO(str(a.weights));rows=[];started=time.monotonic()
    for path in sorted((a.dataset/'images'/a.split).glob('*.png')):
        image=cv2.imread(str(path));h,w=image.shape[:2]
        prediction=model(image,imgsz=960,conf=.2,device='cpu',verbose=False)[0]
        # Persist inference before matching to labeled development targets.
        boxes=prediction.boxes.xyxy.cpu().numpy() if prediction.boxes is not None else np.empty((0,4))
        kps=prediction.keypoints.xy.cpu().numpy() if prediction.keypoints is not None else np.empty((0,0,2))
        labels=[]
        for line in (a.dataset/'labels'/a.split/path.with_suffix('.txt').name).read_text().splitlines():
            if not line.strip():continue
            v=np.array(list(map(float,line.split())));cx,cy,bw,bh=v[1:5]*[w,h,w,h]
            points=v[5:].reshape(-1,3).copy();points[:,:2]*=[w,h]
            labels.append({'box':np.array([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]),'points':points})
        matches=[]
        if labels and len(boxes):
            overlaps=np.array([[iou(g['box'],b) for b in boxes] for g in labels]);gi,pi=linear_sum_assignment(-overlaps)
            for g,d in zip(gi,pi):
                if overlaps[g,d]<.5:continue
                gt=labels[g]['points'];valid=gt[:,2]>0
                errors=np.linalg.norm(kps[d]-gt[:,:2],axis=1)
                matches.append({'GT_index':int(g),'prediction_index':int(d),'IoU':float(overlaps[g,d]),
                    'keypoint_errors_px':errors[valid].tolist(),'center_error_px':float(errors[4]) if len(errors)==5 else None})
        rows.append({'image':path.name,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'GT_objects':len(labels),'predicted_objects':len(boxes),'matches':matches})
    matched=sum(len(r['matches']) for r in rows);GT=sum(r['GT_objects'] for r in rows);pred=sum(r['predicted_objects'] for r in rows)
    errors=[e for r in rows for m in r['matches'] for e in m['keypoint_errors_px']]
    centers=[m['center_error_px'] for r in rows for m in r['matches'] if m['center_error_px'] is not None]
    summary={'images':len(rows),'GT_objects':GT,'predictions':pred,'matched_IoU_0_5':matched,
             'object_recall_at_conf_0_2':matched/max(1,GT),'object_precision_at_conf_0_2':matched/max(1,pred),
             'matched_keypoint_median_px':float(np.median(errors)) if errors else None,
             'matched_mouth_center_median_px':float(np.median(centers)) if centers else None,'wall_s':time.monotonic()-started}
    result={'purpose':'development 2D metrics only; no insertion or calibrated uncertainty claim',
            'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'split':a.split,'summary':summary,'images':rows}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
