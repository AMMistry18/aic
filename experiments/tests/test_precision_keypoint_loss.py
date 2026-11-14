"""Run in the Torch environment; verifies visibility and useful far gradients."""
import pytest
torch=pytest.importorskip('torch')
from experiments.precision_keypoint_loss import PrecisionKeypointLoss


def test_precision_loss_has_finite_nonzero_gradient_for_large_error():
    prediction=torch.tensor([[[40.,0.],[1.,0.]]],requires_grad=True)
    loss=PrecisionKeypointLoss()(prediction,torch.zeros_like(prediction),torch.ones(1,2),torch.tensor([[100.]]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
    assert prediction.grad[0,0,0]>0 and prediction.grad[0,1,0]>0


def test_invisible_keypoints_do_not_influence_loss_or_gradient():
    prediction=torch.tensor([[[40.,0.],[1.,0.]]],requires_grad=True)
    loss=PrecisionKeypointLoss()(prediction,torch.zeros_like(prediction),torch.tensor([[0.,1.]]),torch.tensor([[100.]]))
    loss.backward()
    assert prediction.grad[0,0].abs().sum()==0
    assert prediction.grad[0,1,0]>0
