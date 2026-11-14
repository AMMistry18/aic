"""Prospective longer original-model adaptation; training split only updates.

Uses the existing frozen 20 training scenes and six development scenes, never
final trials. Writes a new checkpoint directory. Does not change live weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import cv2
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['sc_mouth','sc_plug'],required=True);p.add_argument('--name',required=True);p.add_argument('--loss',choices=['standard','precision','task_edges','task_affine'],required=True);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU qualification required; no silent CPU fallback')
    data=Path(os.environ.get('AIC_TRAIN_DATA_ROOT',str(ROOT/'.artifacts/perception_data29/qualified')));out=ROOT/'.artifacts/perception_data30/training'/a.name
    if out.exists():raise RuntimeError('Choose a new name; preserve all previous checkpoints')
    out.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    original=ROOT/'aic_example_policies/aic_example_policies/ros/weights'/('best_sc_mouth_pose.pt' if a.kind=='sc_mouth' else 'best_sc_plug_pose.pt')
    baseline=json.loads((ROOT/'.artifacts/perception_data29/training'/(a.kind+'_transfer29_v1')/'training_plan.json').read_text())
    args=baseline['training_parameters'].copy();args.update(data=str(data/'dataset'/a.kind/'dataset.yaml'),device=0,batch=4,epochs=a.epochs,patience=a.epochs,seed=20261030,project=str(out),name='run',plots=False,workers=0,amp=False)
    plan=dict(purpose=__doc__,kind=a.kind,loss=a.loss,original_weight_sha256=sha(original),dataset_manifest_sha256=sha(data/'manifest.json'),dataset_sha_index_sha256=sha(data/'dataset_SHA256SUMS.json'),trainer_sha256=sha(Path(__file__)),training_parameters=args,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0),status='running',final_100000_series_used=False)
    (out/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    model=YOLO(str(original))
    if a.loss=='precision':
        from experiments.precision_keypoint_loss import install_precision_loss
        plan['loss_source_sha256']=sha(ROOT/'experiments/precision_keypoint_loss.py');model.add_callback('on_train_start',install_precision_loss)
        (out/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    elif a.loss=='task_edges':
        if a.kind!='sc_plug':raise ValueError('Cuboid-edge loss is only defined for the eight-point plug')
        from experiments.task_edge_keypoint_loss import install_task_edge_loss
        plan['loss_source_sha256']=sha(ROOT/'experiments/task_edge_keypoint_loss.py')
        plan['coordinate_loss_source_sha256']=sha(ROOT/'experiments/precision_keypoint_loss.py')
        plan['projected_edge_tolerance_deg']=.25
        model.add_callback('on_train_start',install_task_edge_loss)
        (out/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    elif a.loss=='task_affine':
        if a.kind!='sc_plug':raise ValueError('Affine cuboid loss requires eight-point SC plug labels')
        from experiments.task_affine_pose_loss import install_task_affine_pose_loss
        plan['loss_source_sha256']=sha(ROOT/'experiments/task_affine_pose_loss.py')
        plan['coordinate_loss_source_sha256']=sha(ROOT/'experiments/precision_keypoint_loss.py')
        plan['rotation_proxy_scale_deg']=.25;plan['proxy_model']='weak-perspective affine rows with Gram-Schmidt rotation'
        model.add_callback('on_train_start',install_task_affine_pose_loss)
        (out/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    start=time.monotonic();result=model.train(**args)
    plan.update(status='complete',wall_s=time.monotonic()-start,metrics={k:float(v) for k,v in result.results_dict.items()})
    for name in ['best.pt','last.pt']:
        file=out/'run/weights'/name
        if file.exists():plan[name+'_sha256']=sha(file)
    (out/'training_result.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({k:plan[k] for k in ['status','kind','loss','wall_s','gpu']}))


if __name__=='__main__':main()
