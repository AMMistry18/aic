"""Adapt original SFP checkpoints using scene-separated development data only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import cv2
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'.artifacts/sfp_visual33/data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['sfp_nic','sfp_plug'],required=True);p.add_argument('--name',required=True);a=p.parse_args()
    if json.loads((DATA/'progress.json').read_text())['status']!='collection_complete':raise RuntimeError('Incomplete collection')
    if not torch.cuda.is_available():raise RuntimeError('Expected qualified GPU, no CPU fallback')
    for name,digest in json.loads((DATA/'dataset_SHA256SUMS.json').read_text()).items():
        if sha(DATA/name)!=digest:raise RuntimeError('Dataset changed: '+name)
    out=ROOT/'.artifacts/sfp_visual33/training'/a.name
    if out.exists():raise RuntimeError('Preserve prior checkpoint directory')
    out.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    original=ROOT/'aic_example_policies/aic_example_policies/ros/weights'/('best.pt' if a.kind=='sfp_nic' else 'best_sfp_plug_pose.pt')
    args=dict(data=str(DATA/'dataset'/a.kind/'dataset.yaml'),epochs=60,imgsz=960,batch=4,workers=0,
              device=0,amp=False,seed=20261102,deterministic=True,freeze=10,cache=False,
              optimizer='AdamW',lr0=.0002,lrf=.1,cos_lr=True,patience=60,mosaic=0.,mixup=0.,
              degrees=2.,translate=.03,scale=.1,fliplr=0.,flipud=0.,hsv_h=.015,hsv_s=.3,hsv_v=.3,
              plots=False,project=str(out),name='run',exist_ok=False,save=True,close_mosaic=0)
    plan=dict(kind=a.kind,original_weight_sha256=sha(original),manifest_sha256=sha(DATA/'manifest.json'),
              dataset_index_sha256=sha(DATA/'dataset_SHA256SUMS.json'),trainer_sha256=sha(Path(__file__)),
              training_parameters=args,torch_version=torch.__version__,cuda_version=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0),status='running',held_out_final_cases_used=0)
    (out/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    start=time.monotonic();result=YOLO(str(original)).train(**args)
    plan.update(status='complete',wall_s=time.monotonic()-start,metrics={k:float(v) for k,v in result.results_dict.items()},
                checkpoints={n:sha(out/'run/weights'/n) for n in ['best.pt','last.pt']})
    (out/'training_result.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)


if __name__=='__main__':main()
