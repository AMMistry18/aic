"""Training-only precision loss ablation; no runtime inference changes.

Ultralytics 8.4.48 uses sigma=1/N for non-COCO landmark layouts in its OKS
loss. Here a non-saturating Huber residual targets 0.5% of box linear scale.
This is a declared optimization experiment, not calibrated pose uncertainty.
"""
import torch
from torch import nn


class PrecisionKeypointLoss(nn.Module):
    def __init__(self,relative_scale=.005,loss_scale=.05):
        super().__init__();self.relative_scale=relative_scale;self.loss_scale=loss_scale

    def forward(self,pred_kpts,gt_kpts,kpt_mask,area):
        tolerance=(self.relative_scale*area.reshape(-1,1,1).sqrt()).clamp_min(1e-6)
        residual=((pred_kpts[...,:2]-gt_kpts[...,:2])/tolerance).abs()
        huber=torch.where(residual<=1,.5*residual.square(),residual-.5).mean(-1)
        visibility_factor=kpt_mask.shape[1]/(kpt_mask.sum(dim=1)+1e-9)
        return self.loss_scale*(visibility_factor[:,None]*huber*kpt_mask).mean()


def install_precision_loss(trainer):
    from ultralytics.utils.loss import v8PoseLoss
    model=trainer.model
    criterion=v8PoseLoss(model)
    criterion.keypoint_loss=PrecisionKeypointLoss()
    # EMA is created before on_train_start. Exported EMA checkpoints retain the
    # standard PoseModel class and need no custom training class at inference.
    model.criterion=criterion
