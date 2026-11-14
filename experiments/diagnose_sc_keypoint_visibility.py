"""Offline audit of overconfident keypoints behind robot/cable occluders."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion import sc_body_refinement as body
from aic_model.insertion.sfp_plug_pose import PlugPoseView,PlugKeypointDetection,fuse_multiview_keypoints
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as POINTS
from experiments.lastinch_geometry import HOME,robot_fk


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    audit=json.loads(a.audit.read_text());replay=json.loads(a.replay.read_text());fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME);views=[];masks={}
    for v in audit['views']:
        n=v['camera'];K=np.array(v['K']);T=np.array(v['T_camera_from_base_exact']);C=np.linalg.inv(fk[n+'/optical'])@fk['gripper/tcp'];B=np.linalg.inv(T)@C
        image=cv2.imread(str(a.audit.parent/(n+'_raw.png')));tool=cv2.dilate(body.robot_tool_mask(K,T,B,image.shape),np.ones((3,3),np.uint8))>0
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);orange=cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255]))>0
        masks[n]={'robot':tool,'robot_and_cable':tool|orange}
        views.append(PlugPoseView(n,image,K,np.linalg.inv(T),v['stamp_s'],n))
    rows=[]
    for condition in replay['conditions']:
        for mode in ['unchanged','robot','robot_and_cable']:
            detections=[];excluded={}
            for d in condition['detections']:
                uv=np.array(d['keypoints_px']);confidence=np.array(d['keypoint_confidences']);n=d['camera'];removed=[]
                if mode!='unchanged':
                    mask=masks[n][mode];h,w=mask.shape;xy=np.rint(uv).astype(int);valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                    bad=~valid;bad[valid]|=mask[xy[valid,1],xy[valid,0]];removed=np.flatnonzero(bad).tolist();confidence[bad]=0
                excluded[n]=removed;detections.append(PlugKeypointDetection(n,uv,confidence,d['box_confidence']))
            row=dict(crop_refine=condition['crop_refine'],pad=condition['pad'],mode=mode,excluded_keypoint_ids=excluded,GT_used_for_fit=False)
            try:
                pos,R,shape,error,count,confidence=fuse_multiview_keypoints(views,detections,local_keypoints_m=POINTS)
                row.update(position=pos.tolist(),rotation=R.tolist(),shape_mm=shape*1000,reprojection_px=error,count=count,original_geometry_gates_passed=shape<=.0035 and error<=6)
            except ValueError as e:row['reason']=str(e)
            rows.append(row)
    (a.out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0];truth=np.linalg.inv(B)@np.array(audit['views'][0]['tip_world'])
    for r in rows:
        if 'position' in r:r.update(diagnostic_position_mm=float(np.linalg.norm(np.array(r['position'])-truth[:3,3])*1000),diagnostic_angle_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@np.array(r['rotation'])).magnitude())))
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k not in ['rotation','position']} for r in rows],indent=2))


if __name__=='__main__':main()
