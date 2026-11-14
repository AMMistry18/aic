"""Training-domain repair: original-proposal native crops plus full images.

Uses only the existing training/development split. Crop boxes come from the
original network, not object truth. Supervised labels are transformed exactly
into each crop; no image synthesis or final data is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise RuntimeError('Preserve existing output; use a new identity')
    a.out.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    source=ROOT/'.artifacts/perception_data29/qualified';manifest=json.loads((source/'manifest.json').read_text())
    weight=ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt'
    estimator=ScPlugPoseEstimator(weight,device='cuda:0',crop_refine=False);model=estimator._load_model();records=[]
    dataset=a.out/'dataset/sc_plug'
    for case in manifest['cases']:
        split=case['split'];name=case['case_id']
        if split not in ('train','val'):raise ValueError('No final/test split accepted')
        for camera in ['left_camera','center_camera','right_camera']:
            imagepath=source/'dataset/sc_plug/images'/split/(name+'_'+camera+'.png');labelpath=source/'dataset/sc_plug/labels'/split/(name+'_'+camera+'.txt')
            image=cv2.imread(str(imagepath));h,w=image.shape[:2];label=np.loadtxt(labelpath,ndmin=2)
            if len(label)!=1:raise ValueError('Exactly one held SC plug expected')
            parsed=estimator._best_per_result(model([image],verbose=False,conf=estimator.conf_threshold,imgsz=estimator.imgsz,device='cuda:0'))[0]
            if parsed is None or parsed[3] is None:raise ValueError('Original proposal missing; preserve failure, do not substitute GT crop')
            box=parsed[3];cx=(box[0]+box[2])*.5;cy=(box[1]+box[3])*.5;half=max(.5*6*max(box[2]-box[0],box[3]-box[1]),16)
            x0=max(0,math.floor(cx-half));y0=max(0,math.floor(cy-half));x1=min(w,math.ceil(cx+half));y1=min(h,math.ceil(cy+half))
            crop=image[y0:y1,x0:x1];ch,cw=crop.shape[:2]
            transformed=label[0].copy();xywh=label[0,1:5]*[w,h,w,h]
            corners=np.r_[xywh[:2]-xywh[2:]*.5,xywh[:2]+xywh[2:]*.5]-[x0,y0,x0,y0]
            if np.any(corners[:2]<0) or np.any(corners[2:]>[cw,ch]):raise ValueError('Supervised plug box leaves native crop')
            transformed[1:5]=np.r_[(corners[:2]+corners[2:])*.5,corners[2:]-corners[:2]]/[cw,ch,cw,ch]
            kp=transformed[5:].reshape(8,3);kp[:,:2]=(label[0,5:].reshape(8,3)[:,:2]*[w,h]-[x0,y0])/[cw,ch]
            if np.any(kp[:,:2]<0) or np.any(kp[:,:2]>1):raise ValueError('Keypoint outside crop; do not silently clip')
            for suffix,im,lab in [('full',image,label[0]),('native6',crop,transformed)]:
                stem=name+'_'+camera+'_'+suffix;ip=dataset/'images'/split/(stem+'.png');lp=dataset/'labels'/split/(stem+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(ip),im);np.savetxt(lp,lab[None],fmt='%.12g')
            records.append(dict(case_id=name,split=split,camera=camera,crop_xyxy=[x0,y0,x1,y1],source_image_sha256=hashlib.sha256(imagepath.read_bytes()).hexdigest(),source_label_sha256=hashlib.sha256(labelpath.read_bytes()).hexdigest()))
    config=yaml.safe_load((source/'dataset/sc_plug/dataset.yaml').read_text());config['path']=str(dataset);(dataset/'dataset.yaml').write_text(yaml.safe_dump(config))
    plan=dict(purpose=__doc__,cases=manifest['cases'],image_records=records,original_weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),crop_scale=6,full_and_native_images_per_view=2,final_100000_series_used=False)
    (a.out/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    hashes={str(f.relative_to(a.out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.rglob('*') if f.is_file()};(a.out/'dataset_SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(views=len(records),training_images=sum(r['split']=='train' for r in records)*2,development_images=sum(r['split']=='val' for r in records)*2,final_cases_used=0)))


if __name__=='__main__':main()
