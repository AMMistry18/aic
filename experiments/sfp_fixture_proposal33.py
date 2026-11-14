"""Offline candidate: learned keypoints with a declared level fixture family.

No live target or grasp transform is used. The public source NIC tilt is an
explicit workcell prior; full arbitrary fixture orientation is not covered.
"""
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from experiments.sfp_joint_geometry33 import POINTS,SLOT1

# Source NIC entrance family, also used by the earlier diagnostic controller.
FIXED=Rotation.from_euler('x',3.12895).as_matrix()


def fit_fixture(picks):
    first=picks[0];count,rs,ts,_=cv2.solvePnPGeneric(POINTS,first['kps'].astype(float),first['K'],None,flags=cv2.SOLVEPNP_IPPE)
    starts=[]
    for r,t in zip(rs,ts):
        camera=np.eye(4);camera[:3,:3]=cv2.Rodrigues(r)[0];camera[:3,3]=t.ravel();world=np.linalg.inv(first['T'])@camera
        if world[2,2]>-.8:continue
        starts.append((world[:3,3],float(np.rad2deg(np.arctan2(world[1,0],world[0,0])))))
    if not starts:return dict(accepted=False,reason='no_downward_seed')
    fits=[]
    for p,yaw in starts:
        def pose(x):return p+x[:3]*.001,Rotation.from_euler('z',yaw+x[3],degrees=True).as_matrix()@FIXED
        def residual(x):
            position,R=pose(x);points=(R@POINTS.T).T+position;rows=[]
            for pick in picks:
                q=(pick['P']@np.c_[points,np.ones(8)].T).T
                good=np.asarray(pick['kp_conf'])>=.2
                rows.extend(((q[:,:2]/q[:,2:]-pick['kps'])[good]*np.sqrt(np.asarray(pick['kp_conf'])[good,None])).ravel())
            return np.array(rows)
        best=least_squares(residual,np.zeros(4),bounds=([-20,-20,-20,-20],[20]*4),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=150)
        position,R=pose(best.x);error=float(np.mean(abs(residual(best.x))))
        fits.append((best,error,position,R))
    best,error,position,R=min(fits,key=lambda f:f[0].cost)
    return dict(accepted=bool(best.success and error<=5),position=position.tolist(),rotation=R.tolist(),
                residual_px=error,assumption='public_level_NIC_fixture_family',precision_certified=False)


def fit_plug_translation(views,detections,initial,R,local):
    data={v.camera_name:v for v in views}
    def residual(x):
        world=(R@local.T).T+initial+x*.001;rows=[]
        for d in detections:
            view=data[d.camera_name];T=np.linalg.inv(view.T_world_from_camera);P=view.K@T[:3]
            q=(P@np.c_[world,np.ones(len(local))].T).T;good=np.asarray(d.keypoint_confidences)>=.15
            rows.extend(((q[:,:2]/q[:,2:]-d.keypoints_px)[good]*np.sqrt(d.keypoint_confidences[good,None])).ravel())
        return np.array(rows)
    best=least_squares(residual,np.zeros(3),bounds=([-5]*3,[5]*3),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=150)
    error=float(np.mean(abs(residual(best.x))))
    return dict(position=(initial+best.x*.001).tolist(),rotation=R.tolist(),residual_px=error,
                hypothesis='upstream angular alignment; translation from learned keypoints',precision_certified=False)
