"""Declared fixture-family proposal on the archived nominal SFP capture."""
import gzip
import itertools
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model'),str(ROOT/'aic_example_policies')]
from experiments.sfp_fixture_proposal33 import fit_fixture,fit_plug_translation,SLOT1
from aic_model.insertion.sfp_plug_pose import SfpPlugPoseEstimator,PlugPoseView
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
from aic_example_policies.ros.perception_core import PerceptionCore


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    name='dev_44000_sfpvisual33v2_original_perception';folder=ROOT/'.artifacts/lastinch_runs'/name/name
    geom=json.loads((folder/'perception_before/camera_geometry.json').read_text());views=[];cams={}
    train=ROOT/'.artifacts/sfp_visual33/training';detector=PerceptionCore(nic_weights=str(train/'nic33_v1/run/weights/best.pt'))
    estimator=SfpPlugPoseEstimator(train/'plug33_v1/run/weights/best.pt',device='cpu')
    for v in geom:
        image=cv2.imread(str(folder/'perception_before'/(v['camera']+'_raw.png')));K=np.array(v['K']);T=np.linalg.inv(v['T_base_from_camera'])
        views.append(PlugPoseView(v['camera'],image,K,np.linalg.inv(T),v['stamp_s'],v['camera']))
        ds=detector.detect_nic(image,conf_thresh=.2)
        if ds:cams[v['camera']]=[dict(kps=d['kps'],kp_conf=d['kp_conf'],K=K,T=T,P=K@T[:3]) for d in ds[:5]]
    plug=estimator.estimate_multiview(views)
    if plug is None:raise RuntimeError(estimator.last_failure_reason)
    candidates=[]
    for n in range(1,len(cams)+1):
        for subset in itertools.combinations(cams,n):
            for picks in itertools.product(*(cams[c] for c in subset)):
                fitted=fit_fixture(picks)
                if not fitted['accepted']:continue
                for slot,offset in enumerate([np.zeros(3),SLOT1]):
                    p=np.array(fitted['position'])+np.array(fitted['rotation'])@offset;distance=float(np.linalg.norm(p-plug.position_world))
                    if distance<=.02:candidates.append(dict(fitted,position=p.tolist(),slot=slot,distance_to_visual_plug_m=distance,cameras=subset))
    candidates.sort(key=lambda r:r['distance_to_visual_plug_m'])
    out=ROOT/'.artifacts/sfp_visual33/fixture_proposal';out.mkdir(parents=True,exist_ok=True)
    result=dict(candidate_count=len(candidates),candidates=candidates,GT_used_for_selection=False)
    if candidates:
        chosen=candidates[0];R=np.array(chosen['rotation']);ds=estimator.detect_views(views)
        result['plug_translation_fit']=fit_plug_translation(views,ds,plug.position_world,R,SFP_PLUG_LOCAL_KEYPOINTS_M)
    (out/'fits_without_truth.json').write_text(json.dumps(result,indent=2)+'\n')
    spec=json.loads((folder.parent/'scene.json').read_text());B=np.eye(4);B[:3,:3]=np.diag([-1.,-1.,1.]);B[:3,3]=[-.2,.2,0]
    truth_mouth=np.linalg.inv(B)@spec['reset']['mouth_world'];truth_tip=np.linalg.inv(B)@spec['reset']['tip_world']
    if candidates:
        chosen=candidates[0];result['chosen_diagnostic_error_mm']=(1000*(np.array(chosen['position'])-truth_mouth[:3,3])).tolist()
        result['chosen_diagnostic_angle_deg']=float(np.rad2deg(Rotation.from_matrix(truth_mouth[:3,:3].T@chosen['rotation']).magnitude()))
        result['plug_diagnostic_error_mm']=(1000*(np.array(result['plug_translation_fit']['position'])-truth_tip[:3,3])).tolist()
    (out/'diagnostic_scores.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2));print(candidates[:1])


if __name__=='__main__':main()
