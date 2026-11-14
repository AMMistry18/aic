"""Compare original learned full-frame/cropped inference on old DEV exposures.

Offline diagnostic only: no choosing a favourable crop for online acceptance,
no threshold changes, no simulator truth, and no final-trial images.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion.sfp_plug_pose import SfpPlugPoseEstimator,PlugPoseView
from experiments.candidates.efficiency46.perception_runtime import bind_inference_threads

def main():
    out=ROOT/'experiments/crop_development57';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Preserve diagnostic')
    cv2.setNumThreads(1);torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('Expected development GPU')
    weight=ROOT/'experiments/recovery_20261112/checkpoints/plug_recovery20261112/best.pt'
    estimator=SfpPlugPoseEstimator(weight,device='0',crop_refine=True,crop_pad_scale=6.)
    bind_inference_threads(estimator._load_model())
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    rows=[]
    def encode(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    for run in ledger['results']:
        if run['case']['connector']!='sfp' or run.get('status')!='completed':continue
        folder=ROOT/run['evidence_dir'].replace('\\','/')/run['run']/'perception_before'
        calibration=folder/'camera_geometry.json';spec=json.loads(calibration.read_text())
        views=[PlugPoseView(v['camera'],cv2.imread(str(folder/(v['camera']+'_raw.png'))),
            np.array(v['K']),np.array(v['T_base_from_camera']),v['stamp_s'],v['frame_id']) for v in spec]
        if any(v.image_bgr is None for v in views):raise RuntimeError('Missing original dev image')
        measurements={}
        for mode,pad in [('full_frame',None),('crop4',4.),('crop6',6.),('crop8',8.)]:
            estimator.crop_refine=pad is not None
            if pad is not None:estimator.crop_pad_scale=pad
            began=time.perf_counter();ds=estimator.detect_views(views)
            pose=estimator.estimate_multiview(views,now_s=max(v.stamp_s for v in views),detections=ds)
            measurements[mode]=dict(component_wall_s=time.perf_counter()-began,
                accepted=pose is not None,reason=estimator.last_failure_reason,
                detections=[dataclasses.asdict(d) for d in ds],
                pose=None if pose is None else dataclasses.asdict(pose))
        rows.append(dict(run=run['run'],calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
            images_sha256={v.camera_name:hashlib.sha256((folder/(v.camera_name+'_raw.png')).read_bytes()).hexdigest() for v in views},
            original_stamp_spread_s=max(v.stamp_s for v in views)-min(v.stamp_s for v in views),measurements=measurements))
        data=dict(scope='Historical DEV inference comparison; acceptance is not pose accuracy',
            final_cases_used=0,ground_truth_read=False,production_policy_changed=False,
            original_timestamp_gates_preserved=True,weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),rows=rows)
        (out/'results.json').write_text(json.dumps(data,indent=2,default=encode))
        print(json.dumps(dict(run=run['run'],gates={k:v['reason'] for k,v in measurements.items()})),flush=True)

if __name__=='__main__':main()
