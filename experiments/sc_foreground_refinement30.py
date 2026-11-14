"""Offline candidate: original detector, context-suppressed crop refinement.

Only original RGB proposals and robot-owned tool/camera calibration define the
preprocessing. No object pose, truth mask or simulator segmentation is used.
White SC colour is an explicit domain assumption. Not deployed in the policy.
"""
import math
import cv2
import numpy as np
from aic_model.insertion.sc_body_refinement import robot_tool_mask
from aic_model.insertion.sfp_plug_pose import PlugKeypointDetection


def crop_window(box,shape):
    h,w=shape[:2];cx=(box[0]+box[2])*.5;cy=(box[1]+box[3])*.5
    half=max(3*max(box[2]-box[0],box[3]-box[1]),16.)
    return max(0,math.floor(cx-half)),max(0,math.floor(cy-half)),min(w,math.ceil(cx+half)),min(h,math.ceil(cy+half))


def suppress_context(image,box,K,T_camera_from_tcp):
    h,w=image.shape[:2];cx=(box[0]+box[2])*.5;cy=(box[1]+box[3])*.5;half=(box[2:]-box[:2])*.75
    x0,y0=np.maximum(0,np.floor([cx,cy]-half)).astype(int);x1,y1=np.minimum([w,h],np.ceil([cx,cy]+half)).astype(int)
    roi=np.zeros((h,w),bool);roi[y0:y1,x0:x1]=True
    robot=robot_tool_mask(K,np.eye(4),T_camera_from_tcp,image.shape)
    robot=cv2.dilate(robot,np.ones((3,3),np.uint8))!=0
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # Same white-body colour family as the existing body fitter. Soft value
    # transition preserves antialiasing; geometry is not replaced by a CAD mask.
    alpha=np.clip((hsv[:,:,2].astype(float)-70)/40,0,1)*(hsv[:,:,1]<60)*roi*~robot
    count=int(np.count_nonzero(alpha>.5))
    if count<50:raise ValueError('insufficient_observed_white_SC_pixels')
    masked=np.rint(48+alpha[:,:,None]*(image.astype(float)-48)).clip(0,255).astype(np.uint8)
    return masked,dict(observed_foreground_pixels=count,proposal_roi_xyxy=[int(x0),int(y0),int(x1),int(y1)],uses_object_pose_or_segmentation_truth=False)


def detect(original_estimator,refiner,views,camera_from_tcp,*,anchors=None,pixel_phase_ensemble=False):
    model=original_estimator._load_model();kwargs=dict(verbose=False,conf=original_estimator.conf_threshold,imgsz=original_estimator.imgsz)
    if original_estimator.device:kwargs['device']=original_estimator.device
    parsed=original_estimator._best_per_result(model([v.image_bgr for v in views],**kwargs))
    crops=[];metadata=[]
    for v,item in zip(views,parsed):
        if item is None or item[3] is None:raise ValueError('original_proposal_missing')
        box=np.asarray(anchors[v.camera_name] if anchors is not None else item[3],dtype=float)
        masked,diagnostic=suppress_context(v.image_bgr,box,v.K,camera_from_tcp[v.camera_name])
        x0,y0,x1,y1=crop_window(box,v.image_bgr.shape)
        diagnostic.update(original_proposal_box=item[3].tolist(),refinement_proposal_box=box.tolist(),
                          refinement_window_xyxy=[x0,y0,x1,y1],rigid_crop_anchor_applied=anchors is not None,
                          estimate_frozen_or_copied=False)
        crops.append(masked[y0:y1,x0:x1]);metadata.append((v.camera_name,x0,y0,diagnostic))
    phases=[(0.,0.),(.5,0.),(-.5,0.),(0.,.5),(0.,-.5)] if pixel_phase_ensemble else [(0.,0.)]
    batches=[]
    for dx,dy in phases:
        shifted=[crop if dx==dy==0 else cv2.warpAffine(crop,np.array([[1.,0.,dx],[0.,1.,dy]]),(crop.shape[1],crop.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(48,48,48)) for crop in crops]
        batches.append(original_estimator._best_per_result(refiner(shifted,**kwargs)))
    outputs=[]
    for index in range(len(crops)):
        items=[b[index] for b in batches]
        if any(item is None for item in items):raise ValueError('foreground_ensemble_member_missing')
        xy=np.array([item[0]-delta for item,delta in zip(items,phases)]);conf=np.array([item[1] for item in items])
        if np.any(conf<.15):raise ValueError('foreground_ensemble_keypoint_unusable')
        average=np.sum(xy*conf[:,:,None],axis=0)/conf.sum(0)[:,None]
        outputs.append((average,conf.mean(0),float(np.mean([item[2] for item in items])),None))
        metadata[index][3].update(pixel_phase_count=len(phases),maximum_keypoint_phase_spread_px=float(np.max(np.linalg.norm(xy-average,axis=2))))
    detections=[];diagnostics=[]
    for meta,item in zip(metadata,outputs):
        name,x0,y0,diagnostic=meta
        if item is None:raise ValueError('foreground_refinement_missing')
        xy,conf,box_conf,_=item;detections.append(PlugKeypointDetection(name,xy+[x0,y0],conf,box_conf));diagnostics.append(dict(camera=name,**diagnostic))
    return detections,diagnostics
