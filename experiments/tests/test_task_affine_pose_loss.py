import math
import torch
from experiments.task_affine_pose_loss import TaskAffinePoseLoss
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M


def points(degrees=0.):
    X=torch.tensor(SC_PLUG_LOCAL_KEYPOINTS_M,dtype=torch.float32)
    t=math.radians(degrees);R=torch.tensor([[math.cos(t),0.,math.sin(t)],[0.,1.,0.],[-math.sin(t),0.,math.cos(t)]])
    # Orthographic synthetic control, so recovered proxy should be exact.
    return ((X@R.T)[:,:2]*1000)[None]


def test_rotation_proxy_is_invariant_to_translation_and_common_image_scale():
    loss=TaskAffinePoseLoss();gt=points(13);a=loss.rotation_proxy(gt);b=loss.rotation_proxy(gt*1.7+torch.tensor([34.,71.]))
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    assert loss.rotation_term(gt,gt,torch.ones((1,8),dtype=torch.bool))==0


def test_out_of_plane_rotation_is_penalized_at_task_scale():
    loss=TaskAffinePoseLoss();mask=torch.ones((1,8),dtype=torch.bool)
    small=loss.rotation_term(points(.25),points(),mask);large=loss.rotation_term(points(2),points(),mask)
    assert large>6*small and large>.3


def test_masked_or_degenerate_predictions_have_finite_gradients():
    loss=TaskAffinePoseLoss();gt=points();mask=torch.ones((1,8),dtype=torch.bool);pred=torch.zeros_like(gt,requires_grad=True)
    value=loss(pred,gt,mask,torch.tensor([[100.]]));value.backward()
    assert torch.isfinite(value) and torch.isfinite(pred.grad).all()
    mask[:]=False;pred=gt.clone().requires_grad_();value=loss(pred,gt,mask,torch.tensor([[100.]]));value.backward()
    assert value==0 and torch.isfinite(pred.grad).all()
