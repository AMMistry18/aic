"""Image perturbation ensemble prototype; no policy or checkpoint mutation."""
import cv2
import numpy as np
from aic_model.insertion.sfp_plug_pose import PlugPoseView,PlugKeypointDetection

# Fixed before evaluating diagnostic truth. Pixel shifts probe sampling phase;
# small image rotations probe orientation equivariance, then are exactly undone
# in keypoint coordinates. Camera intrinsics/extrinsics remain unchanged.
AUGMENTATIONS=((0.,0.,0.),(.5,0.,0.),(-.5,0.,0.),(0.,.5,0.),(0.,-.5,0.),(0.,0.,1.),(0.,0.,-1.))


def collect(estimator,views):
    groups=[]
    for dx,dy,angle in AUGMENTATIONS:
        warped=[];inverses={}
        for v in views:
            h,w=v.image_bgr.shape[:2];H=np.eye(3);H[:2]=cv2.getRotationMatrix2D(((w-1)*.5,(h-1)*.5),angle,1.)
            H[0,2]+=dx;H[1,2]+=dy;inverses[v.camera_name]=np.linalg.inv(H)
            image=v.image_bgr if (dx,dy,angle)==(0.,0.,0.) else cv2.warpAffine(v.image_bgr,H[:2],(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            warped.append(PlugPoseView(v.camera_name,image,v.K,v.T_world_from_camera,v.stamp_s,v.frame_id))
        detections=[]
        for d in estimator.detect_views(warped):
            q=(inverses[d.camera_name]@np.c_[d.keypoints_px,np.ones(len(d.keypoints_px))].T).T
            detections.append(PlugKeypointDetection(d.camera_name,q[:,:2]/q[:,2:],d.keypoint_confidences,d.box_confidence))
        groups.append(detections)
    return groups


def combine(groups,indices):
    selected=[{d.camera_name:d for d in groups[i]} for i in indices];names=set(selected[0])
    if len(names)!=3 or any(set(s)!=names for s in selected):raise ValueError('three cameras required in every ensemble member')
    result=[]
    for name in sorted(names):
        ds=[s[name] for s in selected];uv=np.array([d.keypoints_px for d in ds]);conf=np.array([d.keypoint_confidences for d in ds])
        if np.any(conf<.15):raise ValueError('every ensemble member must retain usable keypoints')
        average=np.sum(uv*conf[:,:,None],axis=0)/np.sum(conf,axis=0)[:,None]
        result.append(PlugKeypointDetection(name,average,conf.mean(0),float(np.mean([d.box_confidence for d in ds]))))
    return result
