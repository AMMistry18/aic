"""Paired CPU/GPU replay of original cropped, weighted development perception.

Original synchronized camera exposures, weights and acceptance gates are used.
No simulator truth is loaded. This is not an end-to-end insertion speed claim.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import SfpPlugPoseEstimator,PlugPoseView


def main():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    old=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    rows=[];out=ROOT/'experiments/parallel_dev46/cropped_device_benchmark.json'
    for kind,cls in [('sc',ScPlugPoseEstimator),('sfp',SfpPlugPoseEstimator)]:
        runs=[r for r in old['results'] if r['case']['connector']==kind and r.get('status')=='completed'][:3]
        weight=(ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt' if kind=='sc' else
                ROOT/'experiments/recovery_20261112/checkpoints/plug_recovery20261112/best.pt')
        models={d:cls(weight,device=d,crop_refine=True,crop_pad_scale=6.,refinement_strategy='fixed') for d in ['cpu','0']}
        for run in runs:
            folder=ROOT/run['evidence_dir'].replace('\\','/')/run['run']/'perception_before'
            calibration=folder/'camera_geometry.json';spec=json.loads(calibration.read_text())
            views=[PlugPoseView(v['camera'],cv2.imread(str(folder/(v['camera']+'_raw.png'))),
                np.asarray(v['K']),np.asarray(v['T_base_from_camera']),v['stamp_s'],v['frame_id']) for v in spec]
            if any(v.image_bgr is None for v in views):raise RuntimeError('Missing development images')
            measurements={}
            for device,model in models.items():
                def invoke():
                    ds=model.detect_views(views)
                    pose=model.estimate_multiview(views,now_s=max(v.stamp_s for v in views),detections=ds)
                    return ds,pose,model.last_failure_reason
                # Cold initialization and one additional warmup are excluded.
                invoke()
                # Ultralytics select_device('cpu') resets Torch's thread count
                # at first model initialization. Reapply the intended bound
                # after initialization to avoid quota/renderer oversubscription.
                initialized_threads=torch.get_num_threads()
                torch.set_num_threads(1);invoke();times=[];outputs=[]
                for repeat in range(3):
                    torch.cuda.synchronize();start=time.perf_counter();ds,pose,reason=invoke()
                    torch.cuda.synchronize();times.append(time.perf_counter()-start)
                    outputs.append(dict(accepted=pose is not None,reason=reason,
                        detections=[dataclasses.asdict(d) for d in ds],
                        pose=None if pose is None else dataclasses.asdict(pose)))
                measurements[device]=dict(wall_s=times,median_wall_s=statistics.median(times),outputs=outputs,
                    initialized_torch_threads=initialized_threads,timed_torch_threads=torch.get_num_threads())
            cpu=measurements['cpu']['outputs'][0];gpu=measurements['0']['outputs'][0]
            comparison=dict(acceptance_agrees=cpu['accepted']==gpu['accepted'],rejection_agrees=cpu['reason']==gpu['reason'])
            if cpu['pose'] is not None and gpu['pose'] is not None:
                a,b=cpu['pose'],gpu['pose'];Ra=np.array(a['rotation_world_from_plug']);Rb=np.array(b['rotation_world_from_plug'])
                comparison.update(position_difference_m=float(np.linalg.norm(np.array(a['position_world'])-b['position_world'])),
                    rotation_difference_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(Ra.T@Rb)-1)/2,-1,1)))))
            rows.append(dict(connector=kind,run=run['run'],weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
                images_sha256={v.camera_name:hashlib.sha256((folder/(v.camera_name+'_raw.png')).read_bytes()).hexdigest() for v in views},
                original_exposure_spread_s=max(v.stamp_s for v in views)-min(v.stamp_s for v in views),
                measurements=measurements,comparison=comparison,
                speedup=measurements['cpu']['median_wall_s']/measurements['0']['median_wall_s']))
            def encode(x):
                if isinstance(x,np.ndarray):return x.tolist()
                if isinstance(x,np.generic):return x.item()
                raise TypeError(type(x).__name__)
            out.write_text(json.dumps(dict(scope='Historical DEVELOPMENT synchronized RGB; original cropped multiview pipeline',
                robot_commands_sent=0,ground_truth_loaded=False,final_cases_used=0,
                concurrency_note='CPU shares laptop with frozen study and separate short Gazebo test; not an isolated throughput benchmark',
                gpu=torch.cuda.get_device_name(0),cpu_quota_cores=1,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows),indent=2,default=encode))
            print(json.dumps(dict(connector=kind,run=run['run'],speedup=rows[-1]['speedup'],comparison=comparison)),flush=True)


if __name__=='__main__':main()
