"""Exercise all four actual learned detectors in the copied runtime candidate."""
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];DEV=ROOT/'.artifacts/dev46_workspace'
sys.path[:0]=[str(DEV/'aic_model'),str(DEV/'aic_example_policies')]
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import SfpPlugPoseEstimator,PlugPoseView
from aic_example_policies.ros.perception_core import PerceptionCore


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1)
    previous=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    result=dict(scope='Real learned inference through copied candidate modules; not physical insertion qualification',
                final_cases_read=0,ground_truth_loaded=False,rows=[])
    overrides=DEV/'experiments/recovery_20261112/checkpoints'
    pc=PerceptionCore(nic_weights=str(overrides/'nic_recovery20261112/best.pt'),
                      sc_weights=str(overrides/'sc_mouth_recovery20261112/best.pt'))
    for kind,cls in [('sc',ScPlugPoseEstimator),('sfp',SfpPlugPoseEstimator)]:
        old=next(r for r in previous['results'] if r['case']['connector']==kind and r.get('verified_success'))
        folder=ROOT/old['evidence_dir'].replace('\\','/')/old['run']/'perception_before'
        calibration=json.loads((folder/'camera_geometry.json').read_text())
        item=next(v for v in calibration if v['camera']=='center_camera')
        image=cv2.imread(str(folder/'center_camera_raw.png'))
        view=PlugPoseView(item['camera'],image,np.array(item['K']),np.array(item['T_base_from_camera']),item['stamp_s'],item['frame_id'])
        weight=(DEV/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt' if kind=='sc' else
                overrides/'plug_recovery20261112/best.pt')
        estimator=cls(weight,device='cpu',crop_refine=True)
        begin=time.perf_counter();detections=estimator.detect_views([view]);elapsed=time.perf_counter()-begin
        result['rows'].append(dict(detector=kind+'_plug',threads=torch.get_num_threads(),detections=len(detections),
            cold_wall_s=elapsed,weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest()))
        model=pc._load_sc_yolo() if kind=='sc' else pc._load_yolo();model.overrides['device']='cpu'
        begin=time.perf_counter();detections=pc.detect_sc_pose(image) if kind=='sc' else pc.detect_nic(image)
        result['rows'].append(dict(detector=kind+'_port',threads=torch.get_num_threads(),detections=len(detections),
                                  cold_wall_s=time.perf_counter()-begin))
    result['passed']=len(result['rows'])==4 and all(v['threads']==1 for v in result['rows'])
    (ROOT/'experiments/parallel_dev46/thread_candidate_integration.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Integrated thread budget failed')


if __name__=='__main__':main()
