"""Matched original-checkpoint transfer with a non-saturating precision loss."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.precision_keypoint_loss import install_precision_loss


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['sc_mouth','sc_plug'],required=True);p.add_argument('--name',required=True)
    a=p.parse_args();data=ROOT/'.artifacts/perception_data29/qualified'
    if json.loads((data/'progress.json').read_text())['status']!='collection_complete':raise RuntimeError('Collection incomplete')
    weights=ROOT/'aic_example_policies/aic_example_policies/ros/weights'/('best_sc_mouth_pose.pt' if a.kind=='sc_mouth' else 'best_sc_plug_pose.pt')
    out=ROOT/'.artifacts/perception_data29/training'/a.name
    if out.exists():raise RuntimeError('Preserve previous run; choose a new name')
    out.mkdir(parents=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    baseline=ROOT/'.artifacts/perception_data29/training'/f'{a.kind}_transfer29_v1/training_plan.json'
    plan=json.loads(baseline.read_text());args=plan['training_parameters'].copy();args['project']=str(out)
    plan.update(status='running',loss_ablation='non-saturating Huber; relative scale 0.005; loss scale 0.05',
                source_weights_sha256=sha(weights),training_parameters=args,
                loss_source_sha256=sha(ROOT/'experiments/precision_keypoint_loss.py'),
                trainer_source_sha256=sha(Path(__file__)),
                dataset_hash_index_sha256=sha(data/'dataset_SHA256SUMS.json'))
    (out/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    started=time.monotonic();model=YOLO(str(weights));model.add_callback('on_train_start',install_precision_loss)
    results=model.train(**args)
    plan.update(status='complete',wall_s=time.monotonic()-started,metrics={k:float(v) for k,v in results.results_dict.items()})
    for name in ['best.pt','last.pt']:
        path=out/'run/weights'/name
        if path.exists():plan[name+'_sha256']=sha(path)
    (out/'training_result.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'name':a.name,'wall_s':plan['wall_s']}))


if __name__=='__main__':main()
