"""Mechanism checks: a confident bad crop must not replace consistent evidence."""
import numpy as np

from aic_model.insertion.sfp_plug_pose import PlugPoseView,PlugKeypointDetection
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as POINTS
from aic_model.insertion.perception_quality import select_geometric_refinement


def observations(offset=(0.,0.,0.)):
    K=np.array([[800.,0,320],[0,800.,240],[0,0,1]])
    views=[];detections=[]
    world=POINTS+np.array([0,0,.4])+np.array(offset)
    for name,x in [('left',-.08),('right',.08),('center',0.)]:
        T=np.eye(4);T[0,3]=x
        xyz=world-T[:3,3];uv=(K@xyz.T).T;uv=uv[:,:2]/uv[:,2:]
        views.append(PlugPoseView(name,np.zeros((480,640,3),np.uint8),K,T,1.,name))
        detections.append(PlugKeypointDetection(name,uv,np.ones(8)*.99,.99))
    return views,detections


def choose(views,alternatives):
    return select_geometric_refinement(views,alternatives,POINTS,
        max_reprojection_error_px=6.,max_keypoint_rmse_m=.0035,max_pose_disagreement_m=.00025)


def test_inconsistent_high_confidence_crop_cannot_replace_valid_full_frame():
    views,full=observations()
    crop=list(full)
    # Same high confidence, but independently displaced camera predictions.
    crop[0]=PlugKeypointDetection(full[0].camera_name,full[0].keypoints_px+[0,40],np.ones(8),1.)
    ds,diag=choose(views,[('full',full),('crop',crop)])
    assert diag['selected']=='full'
    assert len(ds)==3
    assert not diag['conditions'][1]['accepted_geometry']


def test_two_consistent_but_different_object_hypotheses_cause_abstention():
    views,one=observations()
    _,two=observations((.004,0,0))
    ds,diag=choose(views,[('one',one),('two',two)])
    assert not ds
    assert diag['reason']=='valid_refinements_disagree'


def test_no_pose_is_invented_when_all_views_missing():
    views,_=observations()
    ds,diag=choose(views,[('full',[]),('crop',[])])
    assert not ds
    assert diag['reason']=='all_refinements_failed_original_geometry_gates'


def test_selective_estimator_retains_age_and_geometry_rejection_contract():
    from aic_model.insertion.sfp_plug_pose import SfpPlugPoseEstimator
    views,full=observations();bad=list(full)
    bad[0]=PlugKeypointDetection(full[0].camera_name,full[0].keypoints_px+[0,40],np.ones(8),1.)
    estimator=SfpPlugPoseEstimator(model=object(),local_keypoints_m=POINTS,
        crop_refine=True,refinement_strategy='geometric_selective')
    estimator._predict_fixed=lambda views,crop_refine=None:full if crop_refine is False else bad
    estimate=estimator.estimate_multiview(views,now_s=1.,max_age_s=.1)
    assert estimate is not None
    assert np.linalg.norm(estimate.position_world-[0,0,.4])<1e-9
    assert estimator.last_refinement_diagnostics['selected']=='full_frame'
    assert estimator.estimate_multiview(views,now_s=2.,max_age_s=.1) is None
    assert estimator.last_failure_reason.startswith('image_age:')
    estimator.refinement_strategy='fixed'
    assert estimator.estimate_multiview(views,now_s=1.,max_age_s=.1) is None
