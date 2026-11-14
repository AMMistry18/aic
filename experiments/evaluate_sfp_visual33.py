"""Original/adapted SFP camera pipeline; save estimates before diagnostic truth.

Scene-separated validation images are development data, never final trials.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from types import MethodType,SimpleNamespace
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model'),str(ROOT/'aic_example_policies')]
os.environ['AIC_SFP_FULL_PORT_ORIENTATION']='1'
from aic_model.insertion.InsertionPolicy import InsertionPolicy
from aic_model.insertion.sfp_plug_pose import SfpPlugPoseEstimator,PlugPoseView
from aic_model.insertion.contract import quat_to_matrix,matrix_to_quat
from aic_example_policies.ros.perception_core import PerceptionCore


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--nic-weights',type=Path,required=True)
    parser.add_argument('--plug-weights',type=Path,required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--joint-nic',action='store_true',help='Offline experimental joint dual-port geometry')
    parser.add_argument('--view-subsets',action='store_true',help='Offline occluded-view association diagnostic')
    parser.add_argument('--no-plug-crop',action='store_true')
    parser.add_argument('--data-root',type=Path,default=ROOT/'.artifacts/sfp_visual33/data');a=parser.parse_args()
    data=a.data_root;out=ROOT/'.artifacts/sfp_visual33/evaluation'/a.name
    if out.exists():raise RuntimeError('Preserve previous evaluation')
    out.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    estimator=SfpPlugPoseEstimator(a.plug_weights,device='cpu',crop_refine=not a.no_plug_crop);detector=PerceptionCore(nic_weights=str(a.nic_weights))
    policy=SimpleNamespace(_pc=detector,get_logger=lambda:SimpleNamespace(info=lambda x:None,warn=lambda x:None))
    for name in ['_estimate_sfp_port_orientation','_reproject_error_px','_select_sfp_candidate']:
        setattr(policy,name,MethodType(getattr(InsertionPolicy,name),policy))
    rows=[]
    for case in json.loads((data/'manifest.json').read_text())['cases']:
        if case['split']!='val':continue
        folder=data/'cases'/case['case_id'];capture=json.loads((folder/'training_capture.json').read_text())
        # The isolated training renderer freezes every model. First camera
        # messages can arrive at different simulation stamps, but represent
        # exactly the same static configuration. Audit that property before
        # forming a static snapshot. This does not test or alter runtime sync.
        world=ET.parse(folder/'training_world.sdf').getroot().find('world')
        if not world.findall('model') or any(m.findtext('static')!='true' for m in world.findall('model')):
            raise RuntimeError('Static snapshot evaluation requires all models frozen')
        views=[];detections={};slot_cams={0:{},1:{}};nic_cams={}
        for camera,v in capture['cameras'].items():
            image=cv2.imread(str(data/'dataset/sfp_plug/images/val'/(case['case_id']+'_'+camera+'.png')))
            K=np.array(v['K']);T=np.linalg.inv(v['T_world_from_camera']);P=K@T[:3]
            views.append(PlugPoseView(camera,image,K,np.linalg.inv(T),0.,camera))
            ds=detector.detect_nic(image,conf_thresh=.2);detections[camera]=len(ds)
            if ds:nic_cams[camera]=[dict(kps=d['kps'],kp_conf=d['kp_conf'],P=P) for d in ds[:5]]
            for slot in (0,1):
                picks=[dict(kps=d['kps'][4*slot:4*slot+4],conf=d['conf'],kp_conf=d['kp_conf'][4*slot:4*slot+4],K=K,T=T,P=P) for d in ds[:5]]
                if picks:slot_cams[slot][camera]=picks
        plug=estimator.estimate_multiview(views)
        row=dict(case_id=case['case_id'],detections=detections,plug_failure=estimator.last_failure_reason,
                 joint_dual_port_geometry=a.joint_nic,
                 view_subset_diagnostic=a.view_subsets,plug_crop_refinement=not a.no_plug_crop,
                 original_capture_stamps_s={c:v['image_stamp_s'] for c,v in capture['cameras'].items()},
                 static_snapshot_stamp_s=0.,runtime_synchronization_tested=False,
                 nic_sha256=hashlib.sha256(a.nic_weights.read_bytes()).hexdigest(),
                 plug_sha256=hashlib.sha256(a.plug_weights.read_bytes()).hexdigest(),GT_used_for_estimation=False)
        if plug is not None:
            row.update(plug_position=plug.position_world.tolist(),plug_rotation=plug.rotation_world_from_plug.tolist(),plug_reprojection_px=plug.reprojection_error_px)
            candidates=[]
            if a.joint_nic and len(nic_cams)>=2:
                from experiments.sfp_joint_geometry33 import fit,SLOT1
                names=list(nic_cams)
                subsets=[names] if not a.view_subsets else [list(c) for n in range(2,len(names)+1) for c in itertools.combinations(names,n)]
                for subset in subsets:
                    for picks in itertools.product(*(nic_cams[c] for c in subset)):
                        fitted=fit(picks)
                        if not fitted['accepted']:continue
                        fitted['view_cameras']=subset
                        for slot,offset in enumerate([np.zeros(3),SLOT1]):
                            R=np.array(fitted['rotation']);position=np.array(fitted['position'])+R@offset
                            error=float(np.mean(fitted['per_view_reprojection_px']))
                            candidates.append(dict(X=position,q_wxyz=matrix_to_quat(R),reproj_px=error,score=error,
                                                   port_slot=slot,joint_fit_diagnostic=fitted))
            elif not a.joint_nic:
                for slot,per_camera in slot_cams.items():
                    for candidate in InsertionPolicy._make_sfp_multiview_candidates(policy,per_camera):
                        candidate['port_slot']=slot;candidates.append(candidate)
            policy._tcp=lambda:(None,None)
            policy._tip_from_tcp=lambda *args:(plug.position_world,plug.rotation_world_from_plug)
            chosen=policy._select_sfp_candidate(candidates,case['target_index'],'SFP development')
            if chosen:
                row.update(port_position=chosen['X'].tolist(),port_rotation=quat_to_matrix(chosen['q_wxyz']).tolist(),
                           port_reprojection_px=chosen['reproj_px'],selected_slot=chosen['port_slot'])
                if 'joint_fit_diagnostic' in chosen:row['joint_fit_diagnostic']=chosen['joint_fit_diagnostic']
            row['candidate_count']=len(candidates)
        rows.append(row);print(case['case_id'],detections,'plug',plug is not None,'port','port_position' in row,flush=True)
    (out/'estimates_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        spec=json.loads((data/'cases'/row['case_id']/'scene.json').read_text())
        for kind,key in [('plug','tip_world'),('port','mouth_world')]:
            if kind+'_position' not in row:continue
            T=np.array(spec['reset'][key]);row[kind+'_diagnostic_error_mm']=float(np.linalg.norm(np.array(row[kind+'_position'])-T[:3,3])*1000)
            row[kind+'_diagnostic_angle_deg']=float(np.rad2deg(Rotation.from_matrix(T[:3,:3].T@row[kind+'_rotation']).magnitude()))
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
