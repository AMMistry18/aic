import math
import torch
from experiments.task_edge_keypoint_loss import TaskEdgeKeypointLoss


def points():
    # Two projected planes of a non-coplanar cuboid.
    return torch.tensor([[[-2.,-1.],[2.,-1.],[2.,1.],[-2.,1.],[-1.7,-1.2],[1.7,-1.2],[1.7,.8],[-1.7,.8]]])


def test_projected_edge_term_ignores_common_translation_but_detects_rotation():
    gt=points();mask=torch.ones((1,8),dtype=torch.bool);loss=TaskEdgeKeypointLoss()
    assert loss.angular_term(gt+torch.tensor([13.,-4.]),gt,mask)<1e-6
    theta=math.radians(2);R=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    rotated=gt@R.T
    assert loss.angular_term(rotated,gt,mask)>.3
    assert loss(rotated,gt,mask,torch.tensor([[8.]]))>loss(gt,gt,mask,torch.tensor([[8.]]))


def test_missing_and_degenerate_edges_have_finite_loss_and_gradient():
    gt=points();pred=gt.clone().requires_grad_();mask=torch.zeros((1,8),dtype=torch.bool);loss=TaskEdgeKeypointLoss()
    value=loss(pred,gt,mask,torch.tensor([[8.]]));value.backward()
    assert value==0 and torch.isfinite(pred.grad).all()
    pred=torch.zeros_like(gt,requires_grad=True);mask[:]=True
    value=loss(pred,gt,mask,torch.tensor([[8.]]));value.backward()
    assert torch.isfinite(value) and torch.isfinite(pred.grad).all()


def test_tolerance_scale_penalizes_one_degree_more_than_quarter_degree():
    gt=points();mask=torch.ones((1,8),dtype=torch.bool);loss=TaskEdgeKeypointLoss();values=[]
    for degrees in [.25,1.]:
        t=math.radians(degrees);R=torch.tensor([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
        values.append(loss.angular_term(gt@R.T,gt,mask))
    assert values[1]>4*values[0]
