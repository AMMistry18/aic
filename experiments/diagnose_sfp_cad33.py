"""Offline dual-SFP-rim geometry from original learned proposals, no GT fitting."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'aic_model'),str(ROOT/'aic_example_policies')]
from aic_model.insertion.sfp_plug_pose import SfpPlugPoseEstimator,PlugPoseView
from aic_model.insertion.sc_rim_refinement import project
from aic_example_policies.ros.perception_core import PerceptionCore

CORNERS=np.array([[.00685,.0043,0],[-.00685,.0043,0],[-.00685,-.0043,0],[.00685,-.0043,0]])
# Public NIC Card Mount geometry: entrance1 is -23.2 mm in entrance0 X.
SLOT1=np.array([-.0232,0,0])
POINTS=np.concatenate([CORNERS,CORNERS+SLOT1])
EDGES=np.concatenate([np.concatenate([c[j]+np.linspace(.05,.95,25)[:,None]*(c[(j+1)%4]-c[j]) for j in range(4)]) for c in [CORNERS,CORNERS+SLOT1]])


def fit(views,p,R,plug):
    prepared=[]
    for v in views:
        image=v['image'];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edge=cv2.Canny(gray,15,40)
        # Reject edges on the visually estimated held connector. Public shape
        # bounds and a small uncertainty dilation are not object truth masks.
        from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
        puv=project(v['P'],(plug.rotation_world_from_plug@SFP_PLUG_LOCAL_KEYPOINTS_M.T).T+plug.position_world)-.5
        occluded=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(occluded,cv2.convexHull(np.rint(puv).astype(np.int32)),255)
        occluded=cv2.dilate(occluded,np.ones((9,9),np.uint8))
        field=cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float)
        uv=project(v['P'],(R@EDGES.T).T+p)-.5
        visible=map_coordinates(occluded,uv[:,::-1].T,order=0,mode='constant',cval=255)==0
        if visible.sum()<30:return dict(accepted=False,reason='insufficient_visible_rim')
        prepared.append((v['P'],field,visible))
    def pose(x):return p+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@R
    def optimize(indices):
        def residual(x):
            pos,rot=pose(x);world=(rot@EDGES.T).T+pos
            return np.concatenate([np.minimum(map_coordinates(prepared[i][1],(project(prepared[i][0],world[prepared[i][2]])-.5)[:,::-1].T,order=1,mode='constant',cval=20),12) for i in indices])
        opts=[least_squares(residual,[x,y,0,0,0,0],bounds=([-4]*3+[-8]*3,[4]*3+[8]*3),loss='soft_l1',f_scale=1,max_nfev=120,diff_step=1e-4) for x,y in [(0,0),(-.5,0),(.5,0),(0,-.5),(0,.5)]]
        best=min(opts,key=lambda x:x.cost);return best,float(np.mean(residual(best.x)))
    best,error=optimize(range(len(views)));position,rotation=pose(best.x)
    pairs=[]
    for omit in range(len(views)):
        pair,_=optimize([i for i in range(len(views)) if i!=omit]);pp,rr=pose(pair.x)
        pairs.append(dict(position_m=float(np.linalg.norm(pp-position)),angle_deg=float(np.rad2deg(Rotation.from_matrix(rotation.T@rr).magnitude()))))
    return dict(accepted=error<=1 and max(x['position_m'] for x in pairs)<=.0001 and max(x['angle_deg'] for x in pairs)<=.1,
                position=position.tolist(),rotation=rotation.tolist(),edge_error_px=error,pairs=pairs)


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    run='dev_44000_sfpvisual33v2_original_perception';folder=ROOT/'.artifacts/lastinch_runs'/run/run
    geometry=json.loads((folder/'perception_before/camera_geometry.json').read_text());views=[];detections=[]
    weights=ROOT/'aic_example_policies/aic_example_policies/ros/weights'
    detector=PerceptionCore(nic_weights=str(weights/'best.pt'));est=SfpPlugPoseEstimator(weights/'best_sfp_plug_pose.pt',device='cpu')
    for v in geometry:
        image=cv2.imread(str(folder/'perception_before'/(v['camera']+'_raw.png')));K=np.array(v['K']);T=np.linalg.inv(v['T_base_from_camera'])
        views.append(dict(camera=v['camera'],image=image,K=K,T=T,P=K@T[:3],stamp=v['stamp_s']))
        detections.append(detector.detect_nic(image,conf_thresh=.2))
    plug=est.estimate_multiview([PlugPoseView(v['camera'],v['image'],v['K'],np.linalg.inv(v['T']),v['stamp'],v['camera']) for v in views])
    if plug is None:raise RuntimeError(est.last_failure_reason)
    seeds=[]
    for v,ds in zip(views,detections):
        for d in ds:
            count,rvecs,tvecs,_=cv2.solvePnPGeneric(POINTS,d['kps'].astype(float),v['K'],None,flags=cv2.SOLVEPNP_IPPE)
            for rvec,tvec in zip(rvecs,tvecs):
                cam=np.eye(4);cam[:3,:3]=cv2.Rodrigues(rvec)[0];cam[:3,3]=tvec.ravel();pose=np.linalg.inv(v['T'])@cam
                distances=[np.linalg.norm(pose[:3,3]+pose[:3,:3]@offset-plug.position_world) for offset in [np.zeros(3),SLOT1]]
                if min(distances)>.025 or pose[2,2]>-.8:continue
                seeds.append(dict(camera=v['camera'],position=pose[:3,3].tolist(),rotation=pose[:3,:3].tolist(),nearest_plug_m=float(min(distances))))
    out=ROOT/'.artifacts/sfp_visual33/cad_diagnostic';out.mkdir(parents=True,exist_ok=True)
    (out/'seeds_without_truth.json').write_text(json.dumps(seeds,indent=2)+'\n')
    results=[]
    for seed in seeds:
        result=fit(views,np.array(seed['position']),np.array(seed['rotation']),plug);results.append(dict(seed=seed,**result));print(result,flush=True)
    (out/'fits_without_truth.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
