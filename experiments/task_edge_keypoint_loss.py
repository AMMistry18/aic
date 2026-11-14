"""Training-only tolerance-scaled coordinate and projected cuboid-edge loss.

Projected angular accuracy is not certified 3D pose accuracy. This objective
keeps the original pose head/checkpoint format; runtime gates are unchanged.
"""
import math
import torch
from torch import nn
from experiments.precision_keypoint_loss import PrecisionKeypointLoss

EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


class TaskEdgeKeypointLoss(nn.Module):
    def __init__(self,angle_tolerance_deg=.25):
        super().__init__();self.angle_tolerance_rad=math.radians(angle_tolerance_deg)
        # A point at half an object's apparent extent moves by approximately
        # extent*theta/2 under a small in-plane rotation. This is an objective
        # scale, not a bound on 3D error or an altered acceptance threshold.
        scale=.5*math.tan(self.angle_tolerance_rad)
        self.coordinate=PrecisionKeypointLoss(relative_scale=scale,loss_scale=.05*scale/.005)

    def angular_term(self,pred,gt,mask):
        first=torch.tensor([e[0] for e in EDGES],device=pred.device);second=torch.tensor([e[1] for e in EDGES],device=pred.device)
        a=pred[:,second,:2]-pred[:,first,:2];b=gt[:,second,:2]-gt[:,first,:2]
        cross=a[...,0]*b[...,1]-a[...,1]*b[...,0];dot=(a*b).sum(-1)
        angle=torch.atan2(cross,dot+1e-8)
        valid=mask[:,first]&mask[:,second]&(b.square().sum(-1)>1e-8)
        residual=angle.abs()/self.angle_tolerance_rad
        huber=torch.where(residual<=1,.5*residual.square(),residual-.5)
        per_object=(huber*valid).sum(1)/valid.sum(1).clamp_min(1)
        return .05*per_object.mean()

    def forward(self,pred_kpts,gt_kpts,kpt_mask,area):
        if pred_kpts.shape[1]!=8:raise ValueError('SC cuboid objective requires eight keypoints')
        return self.coordinate(pred_kpts,gt_kpts,kpt_mask,area)+self.angular_term(pred_kpts,gt_kpts,kpt_mask.bool())


def install_task_edge_loss(trainer):
    from ultralytics.utils.loss import v8PoseLoss
    criterion=v8PoseLoss(trainer.model);criterion.keypoint_loss=TaskEdgeKeypointLoss();trainer.model.criterion=criterion
