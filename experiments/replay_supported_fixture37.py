"""Compare original/candidate selection on identical archived RGB observations.

Offline computation benchmark only. Uses original saved image calibration and
adapted checkpoints; no truth poses. Does not test live acquisition freshness.
Run while the owned simulation is stopped to avoid affecting trial deadlines.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from aic_example_policies.ros.perception_core import PerceptionCore
from aic_model.insertion.sfp_plug_pose import PlugPoseView,SfpPlugPoseEstimator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair37/perception_candidate'


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def eager(module,cams,plug):
    candidates=[];calls=0
    for count in range(2,len(cams)+1):
        for subset in itertools.combinations(cams,count):
            for picks in itertools.product(*(cams[c] for c in subset)):
                calls+=1;fit=module.fit_fixture(picks)
                if fit is None:continue
                for slot,offset in enumerate((np.zeros(3),module.SLOT1)):
                    p=np.array(fit['position'])+np.array(fit['rotation'])@offset
                    candidates.append(dict(fit,position=p.tolist(),slot=slot,cameras=list(subset)))
    return module.select_candidate(candidates,plug),calls


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);a=p.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    full=load('fixture_full',OUT/'sfp_fixture_full.py')
    fast=load('fixture_support',OUT/'sfp_fixture_support.py')
    base=ROOT/'.artifacts/sfp_visual33/training'
    estimator=SfpPlugPoseEstimator(base/'plug33_v1/run/weights/best.pt',device='cpu')
    detector=PerceptionCore(nic_weights=str(base/'nic33_v1/run/weights/best.pt'))
    results=[]
    for run in a.runs:
        folder=ROOT/'.artifacts/lastinch_runs'/run/run/'perception_before'
        records=json.loads((folder/'camera_geometry.json').read_text());views=[];hashes={}
        for r in records:
            file=folder/(r['camera']+'_raw.png');hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
            views.append(PlugPoseView(r['camera'],cv2.imread(str(file)),np.array(r['K']),
                np.array(r['T_base_from_camera']),r['stamp_s'],r['frame_id']))
        inference_start=time.perf_counter()
        detections=estimator.detect_views(views)
        measured=estimator.estimate_multiview(views,detections=detections)
        if measured is None:raise RuntimeError('Archived plug estimate rejected: '+str(estimator.last_failure_reason))
        cams={}
        for view in views:
            T=np.linalg.inv(view.T_world_from_camera);ds=detector.detect_nic(view.image_bgr,conf_thresh=.2)
            if ds:cams[view.camera_name]=[dict(kps=d['kps'],kp_conf=d['kp_conf'],K=view.K,T=T,P=view.K@T[:3]) for d in ds[:5]]
        inference_s=time.perf_counter()-inference_start
        start=time.perf_counter();reference,calls=eager(full,cams,measured.position_world);full_s=time.perf_counter()-start
        start=time.perf_counter();observed,diagnostic=fast.fit_supported_fixture(cams,measured.position_world);lazy_s=time.perf_counter()-start
        assert reference==observed,(run,reference,observed)
        row=dict(run=run,exact_seed_dictionary_match=True,detections_per_camera={k:len(v) for k,v in cams.items()},
                 old_fit_calls=calls,new_fit_calls=diagnostic['fixture_fit_calls'],old_selection_wall_s=full_s,
                 new_selection_wall_s=lazy_s,shared_inference_wall_s=inference_s,selected_seed=observed,
                 camera_geometry_sha256=hashlib.sha256((folder/'camera_geometry.json').read_bytes()).hexdigest(),image_sha256=hashes)
        results.append(row)
        (OUT/'replay_progress.json').write_text(json.dumps(dict(offline_only=True,no_simulator_truth=True,
            checkpoint_changes=False,temporal_acquisition_gate_tested=False,results=results),indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
