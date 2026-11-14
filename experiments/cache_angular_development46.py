"""Short GPU replay of the ORIGINAL cropped multiview plug estimators.

Only earlier rendered training/development cases are read. Object truth is
attached afterwards for scoring and never initializes or selects a fit.
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
sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sfp_plug_pose import PlugPoseView,SfpPlugPoseEstimator
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator


def main():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    out=ROOT/'experiments/parallel_dev46';out.mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('GPU is unavailable')
    rows=[]
    for kind in ['sc','sfp']:
        folder=ROOT/('.artifacts/perception_data29/qualified' if kind=='sc' else '.artifacts/sfp_visual33/data')
        cases=json.loads((folder/'manifest.json').read_text())['cases'][:4]
        weight=(ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt' if kind=='sc' else
                ROOT/'experiments/recovery_20261112/checkpoints/plug_recovery20261112/best.pt')
        cls=ScPlugPoseEstimator if kind=='sc' else SfpPlugPoseEstimator
        estimator=cls(weight,device='0',crop_refine=True,crop_pad_scale=6.,refinement_strategy='fixed')
        for case in cases:
            capture_path=folder/'cases'/case['case_id']/'training_capture.json'
            capture=json.loads(capture_path.read_text());views=[];inputs=[]
            for name,data in capture['cameras'].items():
                image_path=folder/'dataset'/(kind+'_plug')/'images'/case['split']/(case['case_id']+'_'+name+'.png')
                image=cv2.imread(str(image_path))
                if image is None:raise RuntimeError('Missing original rendered RGB')
                views.append(PlugPoseView(name,image,np.array(data['K']),np.array(data['T_world_from_camera']),data['image_stamp_s'],name+'/optical'))
                inputs.append(dict(camera=name,image=str(image_path.relative_to(ROOT)),image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    K=data['K'],T_world_from_camera=data['T_world_from_camera'],stamp_s=data['image_stamp_s']))
            torch.cuda.synchronize();begin=time.perf_counter()
            detections=estimator.detect_views(views)
            estimate=estimator.estimate_multiview(views,now_s=max(v.stamp_s for v in views),max_age_s=.35,detections=detections)
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            # Scoring metadata is read only after the original estimator returns.
            scene=json.loads((capture_path.parent/'scene.json').read_text())
            row=dict(connector=kind,case_id=case['case_id'],inputs=inputs,
                weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                full_original_crop_multiview_wall_s=elapsed,detections=[dataclasses.asdict(d) for d in detections],
                original_estimate=None if estimate is None else dataclasses.asdict(estimate),
                failure_reason=estimator.last_failure_reason,
                scoring_only=dict(tip_world=scene['reset']['tip_world'],initial_angular_deg=case['angular_deg']),
                own_robot_q=scene['robot_q'])
            def encode(x):
                if isinstance(x,np.ndarray):return x.tolist()
                if isinstance(x,np.generic):return x.item()
                raise TypeError(type(x).__name__)
            rows.append(row)
            (out/'angular_inputs.json').write_text(json.dumps(dict(scope='Historical training/development images; not held-out insertion results',
                final_trials_read=0,device=torch.cuda.get_device_name(0),rows=rows),indent=2,default=encode))
            print(json.dumps(dict(connector=kind,case=case['case_id'],original_accepted=estimate is not None,
                                  elapsed_wall_s=elapsed,reason=estimator.last_failure_reason)),flush=True)


if __name__=='__main__':main()
