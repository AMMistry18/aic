"""Replay frozen production perception gates on actual archived live images.

This is static replay, not a fresh synchronization/consensus or insertion test.
No simulator truth is opened. Robot TCP is reconstructed only from calibrated
rigid camera-to-tool kinematics and recorded exposure-time camera transforms.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET
os.environ['AIC_INSERT_VISUAL_REACQUIRE']='1'
os.environ['RL_INSERT_SC_STRICT_PERCEPTION']='1'
os.environ['AIC_SC_RIM_REFINE']='1'
os.environ['AIC_SC_RIM_RASTER_OFFSET_PX']='-0.5'
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model'),str(ROOT/'aic_example_policies')]
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import PlugPoseView,fuse_multiview_keypoints
from aic_model.insertion.sfp_controller import solve_tip_in_tcp
from aic_model.insertion.InsertionPolicy import InsertionPolicy
from aic_model.insertion.sc_controller import perceive_sc_port_pose
from aic_model.insertion import sc_body_refinement as body
from aic_example_policies.ros.perception_core import PerceptionCore
from experiments.lastinch_geometry import HOME,robot_fk
from experiments.summarize_visual_screen30 import NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mouth',type=Path,required=True);p.add_argument('--plug',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--capture',choices=['auto','before'],default='auto');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME)
    estimator=ScPlugPoseEstimator(a.plug,device='cpu',crop_refine=True,crop_pad_scale=6,refinement_strategy='fixed');pc=PerceptionCore(nic_weights=str(ROOT/'aic_example_policies/aic_example_policies/ros/weights/best.pt'),sc_weights=str(a.mouth));results=[]
    for n in NAMES:
        r=ROOT/'.artifacts/lastinch_runs'/n/n;views=[];prepared=[];portviews={};stamps={};hashes={}
        positive='dev_3006_' in n and a.capture=='auto';folder=r/('perception_before' if a.capture=='before' else 'body_inputs/000' if positive else 'perception_after')
        metadata=json.loads((folder/('inputs.json' if positive else 'camera_geometry.json')).read_text());geometry=metadata['views'] if positive else metadata
        for v in geometry:
            name=v['camera'];imagepath=folder/(name+('.png' if positive else '_raw.png'));image=cv2.imread(str(imagepath));hashes[imagepath.name]=hashlib.sha256(imagepath.read_bytes()).hexdigest();K=np.array(v['K'])
            T=np.array(v['T_camera_from_base']) if positive else np.linalg.inv(v['T_base_from_camera'])
            C=np.linalg.inv(fk[name+'/optical'])@fk['gripper/tcp'];B=np.linalg.inv(T)@C
            if positive:B=np.array(v['T_base_from_tcp'])
            stamp=v['stamp_s'];stamps[name]=stamp
            views.append(PlugPoseView(name,image,K,np.linalg.inv(T),stamp,name));portviews[name]=(image,K,T)
            prepared.append(dict(camera=name,image=image,K=K,P=K@T[:3],tool_mask=body.robot_tool_mask(K,T,B,image.shape)))
        row=dict(run=n,purpose='static replay of actual live images; not insertion',source_images_sha256=hashes,simulator_truth_read=False,capture=a.capture,exposure_robot_tcp=B.tolist())
        logs=[];logger=NS(info=lambda m:logs.append(str(m)),warn=lambda m:logs.append(str(m)),error=lambda m:logs.append(str(m)))
        try:
            detections=estimator.detect_views(views);pos,R,shape,error,count,conf=fuse_multiview_keypoints(views,detections,local_keypoints_m=estimator.local_keypoints_m)
            row.update(plug_position=pos.tolist(),plug_rotation=R.tolist(),plug_reprojection_px=error,plug_shape_mm=shape*1000)
            measured=np.eye(4);measured[:3,:3]=R;measured[:3,3]=pos
            row['image_estimated_grasp_tcp']=(np.linalg.inv(B)@measured).tolist()
            if error>6 or shape>.0035:row['stage']='plug_geometry_rejected'
            else:
                policy=InsertionPolicy.__new__(InsertionPolicy);policy._pc=pc;policy.get_logger=lambda:logger;policy._build_views=lambda obs:portviews
                quat=Rotation.from_matrix(B[:3,:3]).as_quat();q=np.r_[quat[3],quat[:3]];policy._tcp=lambda:(B[:3,3],q)
                policy._sc_grasp_transform=solve_tip_in_tcp(B[:3,3],q,pos,R)
                obs=NS(**{name.replace('_camera','')+'_image':NS(header=NS(stamp=stamp)) for name,stamp in stamps.items()})
                selected=perceive_sc_port_pose(policy,NS(id=n),obs);row['rim_diagnostics']=getattr(policy,'_sc_rim_diagnostics',[])
                if selected is None:row['stage']='port_geometry_or_rim_rejected'
                else:
                    pport,qport,reproj=selected;row.update(port_position=pport.tolist(),port_quaternion_wxyz=qport.tolist(),port_seed_reprojection_px=reproj)
                    Rp=Rotation.from_quat(np.r_[qport[1:],qport[0]]).as_matrix();prior,diagnostic=body.aligned_orientation_prior(Rp,R);row['prior_diagnostic']=diagnostic
                    if prior is None:
                        row['stage']='angular_prior_rejected'
                        # Diagnostic ablation only: does translation evidence
                        # exist under the explicitly assumed handoff angle?
                        # This does not change the production rejection above.
                        alternatives=[Rp,Rp@np.diag([-1.,-1.,1.])]
                        candidate=min(alternatives,key=lambda Q:Rotation.from_matrix(Q.T@R).magnitude())
                        row['declared_angular_prior_ablation']=body.fit_translation(prepared,pos,candidate)
                    else:
                        result=body.fit_translation(prepared,pos,prior);row['body_diagnostic']=result;row['stage']='all_static_gates_passed' if result['accepted'] else 'body_fit_rejected'
        except Exception as e:row.update(stage='replay_error',error=repr(e))
        row['production_logs']=logs;results.append(row);print(json.dumps({k:v for k,v in row.items() if k in ['run','stage','error','plug_reprojection_px','prior_diagnostic']}),flush=True)
        (a.out/'replay_without_truth.json').write_text(json.dumps(dict(mouth_sha256=hashlib.sha256(a.mouth.read_bytes()).hexdigest(),plug_sha256=hashlib.sha256(a.plug.read_bytes()).hexdigest(),results=results),indent=2)+'\n')


if __name__=='__main__':main()
