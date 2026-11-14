"""Prepare context-suppressed original-proposal crops; training/development only."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from experiments.lastinch_geometry import HOME,robot_fk
from experiments.sc_foreground_refinement30 import suppress_context


def main():
    out=ROOT/'.artifacts/perception_data30/foreground_native6_v1'
    if out.exists():raise FileExistsError('Preserve existing dataset')
    out.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    original=ROOT/'.artifacts/perception_data29/qualified';native=ROOT/'.artifacts/perception_data30/nativecrop_mixed_v1'
    source=json.loads((native/'manifest.json').read_text());weight=ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt'
    est=ScPlugPoseEstimator(weight,device='cuda:0',crop_refine=False);model=est._load_model()
    fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME)
    records=[];dataset=out/'dataset/sc_plug'
    for row in source['image_records']:
        split=row['split'];name=row['case_id'];camera=row['camera'];stem=name+'_'+camera+'_native6'
        if split not in ('train','val'):raise ValueError('Final/test data forbidden')
        path=original/'dataset/sc_plug/images'/split/(name+'_'+camera+'.png');image=cv2.imread(str(path))
        capture=json.loads((original/'cases'/name/'training_capture.json').read_text());K=np.array(capture['cameras'][camera]['K']);C=np.linalg.inv(fk[camera+'/optical'])@fk['gripper/tcp']
        item=est._best_per_result(model([image],verbose=False,conf=est.conf_threshold,imgsz=est.imgsz,device='cuda:0'))[0]
        if item is None or item[3] is None:raise ValueError('Original proposal missing')
        masked,diagnostic=suppress_context(image,item[3],K,C);x0,y0,x1,y1=row['crop_xyxy'];crop=masked[y0:y1,x0:x1]
        ip=dataset/'images'/split/(stem+'.png');lp=dataset/'labels'/split/(stem+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(ip),crop);shutil.copy2(native/'dataset/sc_plug/labels'/split/(stem+'.txt'),lp)
        records.append(dict(**row,**diagnostic))
        if split=='train' and name=='train_80010':
            preview=np.concatenate([cv2.resize(image[y0:y1,x0:x1],(400,400)),cv2.resize(crop,(400,400))],axis=1);cv2.imwrite(str(out/(camera+'_raw_and_preprocessed.png')),preview)
    config=yaml.safe_load((native/'dataset/sc_plug/dataset.yaml').read_text());config['path']=str(dataset);(dataset/'dataset.yaml').write_text(yaml.safe_dump(config))
    plan=dict(purpose=__doc__,cases=source['cases'],image_records=records,source_manifest_sha256=hashlib.sha256((native/'manifest.json').read_bytes()).hexdigest(),
        preprocessing_source_sha256=hashlib.sha256((ROOT/'experiments/sc_foreground_refinement30.py').read_bytes()).hexdigest(),
        original_weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),object_pose_or_segmentation_truth_for_mask=False,
        labels='Original supervised training labels translated into original-network native crop; training labels only',final_100000_series_used=False,production_enabled=False)
    (out/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n');hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()};(out/'dataset_SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(training_images=sum(r['split']=='train' for r in records),development_images=sum(r['split']=='val' for r in records),foreground_pixel_range=[min(r['observed_foreground_pixels'] for r in records),max(r['observed_foreground_pixels'] for r in records)],final_cases_used=0)))


if __name__=='__main__':main()
