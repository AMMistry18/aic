"""Training-only affine-camera rotation proxy for the SC cuboid pose head.

Weak perspective is an approximation. Relative proxy supervision is not a
calibrated 3D accuracy bound; actual multiview and physical tests decide utility.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M
from experiments.precision_keypoint_loss import PrecisionKeypointLoss


class TaskAffinePoseLoss(nn.Module):
    def __init__(self,angle_tolerance_deg=.25):
        super().__init__();self.tolerance=math.radians(angle_tolerance_deg)
        points=torch.tensor(SC_PLUG_LOCAL_KEYPOINTS_M,dtype=torch.float32)
        points=points-points.mean(0)
        self.register_buffer('object_pseudoinverse',torch.linalg.pinv(points))
        scale=.5*math.tan(self.tolerance)
        self.coordinate=PrecisionKeypointLoss(relative_scale=scale,loss_scale=.05*scale/.005)

    def rotation_proxy(self,keypoints):
        xy=keypoints[...,:2];centered=xy-xy.mean(1,keepdim=True)
        affine=(self.object_pseudoinverse.to(xy)@centered).transpose(1,2)
        first=F.normalize(affine[:,0],dim=1,eps=1e-6)
        second=affine[:,1]-(affine[:,1]*first).sum(1,keepdim=True)*first
        second=F.normalize(second,dim=1,eps=1e-6)
        third=torch.linalg.cross(first,second,dim=1)
        return torch.stack([first,second,third],dim=1)

    def rotation_term(self,pred,gt,mask):
        A=self.rotation_proxy(pred);B=self.rotation_proxy(gt)
        residual=torch.linalg.vector_norm(A-B,dim=(1,2))/(math.sqrt(2)*self.tolerance)
        huber=torch.where(residual<=1,.5*residual.square(),residual-.5)
        valid=mask.bool().all(1)
        return .05*(huber*valid).sum()/valid.sum().clamp_min(1)

    def forward(self,pred,gt,mask,area):
        if pred.shape[1]!=8:raise ValueError('SC cuboid objective requires eight keypoints')
        return self.coordinate(pred,gt,mask,area)+self.rotation_term(pred,gt,mask)


def install_task_affine_pose_loss(trainer):
    from ultralytics.utils.loss import v8PoseLoss
    criterion=v8PoseLoss(trainer.model);criterion.keypoint_loss=TaskAffinePoseLoss();trainer.model.criterion=criterion
