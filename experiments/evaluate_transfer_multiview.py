"""Static development multiview accuracy of actual original pipeline components.

Actual image timestamps are retained. The frozen scene has no object/camera
motion, so geometric replay across asynchronous captures is valid, but this is
explicitly NOT a test of live synchronization, consensus, or physical insertion.
Truth is accessed only after learned predictions and candidate selection.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'aic_model'),str(ROOT/'aic_example_policies')]
from experiments.lastinch_geometry import robot_fk
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import PlugPoseView,fuse_multiview_keypoints
from aic_model.insertion.sfp_controller import solve_tip_in_tcp
from aic_model.insertion.InsertionPolicy import InsertionPolicy
from aic_model.insertion.sc_controller import perceive_sc_port_pose
from aic_example_policies.ros.perception_core import PerceptionCore


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mouth',type=Path,required=True);p.add_argument('--plug',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--strategy',choices=['fixed','geometric_selective'],default='fixed')
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    data=ROOT/'.artifacts/perception_data29/qualified'
    manifest=json.loads((data/'manifest.json').read_text());robot=__import__('xml.etree.ElementTree',fromlist=['parse']).parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot()
    estimator=ScPlugPoseEstimator(a.plug,device='cpu',crop_refine=True,crop_pad_scale=6,refinement_strategy=a.strategy)
    pc=PerceptionCore(sc_weights=str(a.mouth));results=[]
    logger=SimpleNamespace(info=lambda *x:None,warn=lambda *x:None,error=lambda *x:None)
    for case in [c for c in manifest['cases'] if c['split']=='val']:
        folder=data/'cases'/case['case_id'];capture=json.loads((folder/'training_capture.json').read_text())
        # Robot joint positions are permitted robot-state inputs, not object GT.
        robot_q=json.loads((folder/'scene.json').read_text())['robot_q']
        tcp=robot_fk(robot,robot_q)['gripper/tcp'];q=Rotation.from_matrix(tcp[:3,:3]).as_quat();quat=np.r_[q[3],q[:3]]
        views=[];port_views={}
        for name,view in capture['cameras'].items():
            image=cv2.imread(str(data/'dataset/sc_plug/images/val'/f'{case["case_id"]}_{name}.png'))
            K=np.array(view['K']);T=np.array(view['T_world_from_camera'])
            views.append(PlugPoseView(name,image,K,T,view['image_stamp_s'],name))
            port_views[name]=(image,K,np.linalg.inv(T))
        row={'case_id':case['case_id'],'live_synchronization_test':False,'image_stamp_spread_s':float(np.ptp([v.stamp_s for v in views]))}
        try:
            detections=estimator.detect_views(views)
            pos,R,shape,reproj,count,confidence=fuse_multiview_keypoints(views,detections,local_keypoints_m=estimator.local_keypoints_m)
            row.update(plug_position=pos.tolist(),plug_rotation=R.tolist(),plug_reprojection_px=reproj,plug_shape_mm=shape*1000)
            if reproj>6 or shape>.0035:
                row['failure']='original_plug_geometry_gate';results.append(row);continue
            policy=InsertionPolicy.__new__(InsertionPolicy);policy._pc=pc;policy.get_logger=lambda:logger
            policy._build_views=lambda obs:port_views;policy._tcp=lambda:(tcp[:3,3],quat)
            policy._sc_grasp_transform=solve_tip_in_tcp(tcp[:3,3],quat,pos,R)
            selected=perceive_sc_port_pose(policy,SimpleNamespace(),object())
            if selected is None:row['failure']='port_perception'
            else:
                pport,qport,error=selected
                row.update(port_position=pport.tolist(),port_quat_wxyz=qport.tolist(),port_center_reprojection_px=error)
        except ValueError as exc:row['failure']=str(exc)
        results.append(row)
    a.out.with_name(a.out.stem+'_without_truth.json').write_text(json.dumps(results,indent=2)+'\n')
    for row in results:
        # Post-inference evaluator only: no GT position selects a candidate.
        truth=json.loads((data/'cases'/row['case_id']/'scene.json').read_text())['reset']
        tip=np.array(truth['tip_world']);mouth=np.array(truth['mouth_world'])
        if 'plug_position' in row:
            row['diagnostic_plug_error_mm']=float(np.linalg.norm(row['plug_position']-tip[:3,3])*1000)
            row['diagnostic_plug_angle_error_deg']=float(np.rad2deg(Rotation.from_matrix(tip[:3,:3].T@np.array(row['plug_rotation'])).magnitude()))
        if 'port_position' in row:
            row['diagnostic_port_error_mm']=float(np.linalg.norm(row['port_position']-mouth[:3,3])*1000)
            error=mouth[:3,:3].T@((np.array(row['plug_position'])-row['port_position'])-(tip[:3,3]-mouth[:3,3]))
            row['diagnostic_relative_lateral_error_mm']=float(np.linalg.norm(error[:2])*1000)
            row['diagnostic_relative_axial_error_mm']=float(error[2]*1000)
    summary={'static_validation_scenes':len(results),'both_poses_produced':sum('port_position' in r for r in results),
             'live_policy_or_insertion_success_test':False}
    for metric in ['diagnostic_plug_error_mm','diagnostic_plug_angle_error_deg','diagnostic_port_error_mm','diagnostic_relative_lateral_error_mm']:
        values=[r[metric] for r in results if metric in r];summary[metric+'_median']=float(np.median(values)) if values else None
    a.out.write_text(json.dumps({'purpose':__doc__,'summary':summary,'scenes':results},indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
