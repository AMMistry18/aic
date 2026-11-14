"""Bounded offline CPU NIC inference on four historical DEVELOPMENT scenes."""
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'aic_example_policies'),str(ROOT)]
from aic_example_policies.ros.perception_core import PerceptionCore
from experiments.candidates.efficiency46.perception_runtime import bind_inference_threads


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1);out=ROOT/'experiments/perception_efficiency48'
    weights=ROOT/'experiments/recovery_20261112/checkpoints/nic_recovery20261112/best.pt'
    pc=PerceptionCore(nic_weights=str(weights));model=bind_inference_threads(pc._load_yolo());model.overrides['device']='cpu'
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text());rows=[]
    for run in [r for r in ledger['results'] if r['case']['connector']=='sfp' and r.get('status')=='completed'][:4]:
        folder=ROOT/run['evidence_dir'].replace('\\','/')/run['run']/'perception_before'
        calibration=json.loads((folder/'camera_geometry.json').read_text());cameras={};began=time.perf_counter()
        for v in calibration:
            image_path=folder/(v['camera']+'_raw.png');image=cv2.imread(str(image_path));K=np.array(v['K']);T=np.linalg.inv(v['T_base_from_camera'])
            detections=pc.detect_nic(image,conf_thresh=.2)
            cameras[v['camera']]=dict(image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),stamp_s=v['stamp_s'],
                detections=[dict(kps=d['kps'],kp_conf=d['kp_conf'],K=K,T=T,P=K@T[:3]) for d in detections[:5]])
        rows.append(dict(run=run['run'],cameras=cameras,wall_s=time.perf_counter()-began))
        (out/'detections.json').write_text(json.dumps(dict(scope='Historical development NIC detections, no truth or final cases loaded',
            weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),rows=rows),indent=2,
            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
        print(json.dumps(dict(run=run['run'],detections={n:len(v['detections']) for n,v in cameras.items()})),flush=True)


if __name__=='__main__':main()
