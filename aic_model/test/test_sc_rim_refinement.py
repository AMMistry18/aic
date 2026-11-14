import numpy as np
import cv2
from aic_model.insertion.sc_rim_refinement import refine_with_agreement


def test_missing_view_cannot_authorize_refinement():
    fit,diagnostics=refine_with_agreement([],np.zeros(3),np.eye(3))
    assert fit is None
    assert diagnostics['reason']=='three_views_required'


def test_images_without_blue_edges_cannot_authorize_pose():
    views=[]
    for x in [-.03,0,.03]:
        K=np.array([[300.,0,100],[0,300,100],[0,0,1]])
        T=np.c_[np.eye(3),[-x,0,.2]]
        views.append({'image':np.zeros((200,200,3),np.uint8),'P':K@T})
    fit,diagnostics=refine_with_agreement(views,np.zeros(3),np.eye(3))
    assert fit is None
    assert diagnostics['reason']=='fit_quality_rejected'


def test_completely_occluded_rim_is_rejected_without_optimizer_failure():
    K=np.array([[300.,0,100],[0,300,100],[0,0,1]])
    views=[{'image':np.full((200,200,3),255,np.uint8),'P':K@np.c_[np.eye(3),[x,0,.2]]} for x in [-.03,0,.03]]
    fit,diagnostics=refine_with_agreement(views,np.zeros(3),np.eye(3))
    assert fit is None
    assert diagnostics['reason']=='fit_quality_rejected'


def test_cable_crossing_is_not_treated_as_a_port_front_edge():
    from aic_model.insertion.sc_rim_refinement import edge_distance
    image=np.zeros((200,200,3),np.uint8)
    image[40:160,40:160]=[255,100,0]
    image[:,96:105]=[0,128,255]
    distance,occluded=edge_distance(image)
    assert occluded[40,100]!=0
    assert distance[40,100]>1.
    assert occluded[40,60]==0
    assert np.min(distance[38:42,50:70])==0.


def test_cable_hiding_the_entire_target_cannot_authorize_pose():
    K=np.array([[300.,0,100],[0,300,100],[0,0,1]])
    orange=np.zeros((200,200,3),np.uint8);orange[:]=[0,128,255]
    views=[{'image':orange,'P':K@np.c_[np.eye(3),[x,0,.2]]} for x in [-.03,0,.03]]
    fit,diagnostics=refine_with_agreement(views,np.zeros(3),np.eye(3))
    assert fit is None
    assert diagnostics['reason']=='fit_quality_rejected'
