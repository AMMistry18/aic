from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from aic_model.insertion import sc_body_refinement as body
from aic_model.insertion.sc_rim_refinement import project


def synthetic_views():
    points,_=body.template();position=np.array([0.,0.,.4]);views=[]
    K=np.array([[1200.,0,320],[0,1200.,320],[0,0,1.]])
    for i,(x,y) in enumerate([(-.08,0),(.08,0),(0,.06)]):
        T=np.eye(4);T[:3,3]=[-x,-y,0.];P=K@T[:3]
        # Pixel-area antialiasing is needed for the subpixel edge observation;
        # integer filled polygons have a systematic half-pixel expansion.
        high=np.zeros((640*8,640*8,3),np.uint8)
        uv=project(P,points+position)-.5
        cv2.fillConvexPoly(high,cv2.convexHull(np.rint((uv+.5)*8-.5).astype(np.int32)),(220,220,220))
        image=cv2.resize(high,(640,640),interpolation=cv2.INTER_AREA)
        views.append({'camera':str(i),'image':image,'P':P,'tool_mask':np.zeros((640,640),np.uint8)})
    return position,views


def test_multiview_translation_fit_recovers_shift_without_object_truth_input():
    actual,views=synthetic_views()
    result=body.fit_translation(views,actual+[.0006,-.0003,.0003],np.eye(3))
    assert result['accepted'],result
    assert np.linalg.norm(np.asarray(result['position'])-actual)<.0002
    assert result['mean_support_error_px']<=1.
    assert max(result['leave_one_view_out_disagreement_m'])<=.00025


def test_occluded_or_absent_body_is_rejected():
    position,views=synthetic_views()
    for view in views:view['tool_mask'][:]=255
    assert not body.fit_translation(views,position,np.eye(3))['accepted']
    for view in views:view['tool_mask'][:]=0;view['image'][:]=0
    assert body.fit_translation(views,position,np.eye(3))['reason']=='insufficient_white_pixels'


def test_single_view_cannot_certify_body_translation():
    position,views=synthetic_views()
    assert body.fit_translation(views[:1],position,np.eye(3))['reason']=='three_unique_views_required'


def test_tool_projection_is_invariant_to_common_base_frame_change():
    K=np.array([[900.,0,320],[0,900.,320],[0,0,1.]])
    camera=np.eye(4);camera[2,3]=.4;tcp=np.eye(4)
    change=np.eye(4);change[:3,3]=[.2,-.1,.1]
    first=body.robot_tool_mask(K,camera,tcp,(640,640))
    second=body.robot_tool_mask(K,camera@np.linalg.inv(change),change@tcp,(640,640))
    assert first.any()
    np.testing.assert_array_equal(first,second)


def test_body_prior_cannot_silently_replace_full_visual_rotation(monkeypatch):
    from aic_model.insertion.sc_controller import refine_sc_handoff_translation
    monkeypatch.delenv('AIC_SC_KEEP_HANDOFF_ORIENTATION',raising=False)
    with pytest.raises(ValueError,match='declared aligned'):
        refine_sc_handoff_translation(None,None,np.eye(3))


def test_angular_consistency_screen_and_port_half_turn_symmetry():
    from scipy.spatial.transform import Rotation
    port=np.eye(3)
    measured=Rotation.from_euler('z',180.,degrees=True).as_matrix()
    prior,diagnostic=body.aligned_orientation_prior(port,measured)
    np.testing.assert_allclose(prior,measured,atol=1e-12)
    assert diagnostic['port_symmetry_branch_deg']==180
    assert not diagnostic['angular_prior_certified']
    prior,diagnostic=body.aligned_orientation_prior(port,Rotation.from_euler('x',2.,degrees=True).as_matrix())
    assert prior is None
    assert diagnostic['prior_consistency_limit_deg']==pytest.approx(np.degrees(np.arctan2(.00025,.01564)))


def test_explicit_angular_screen_ablation_is_labeled_and_does_not_change_default():
    from scipy.spatial.transform import Rotation
    measured=Rotation.from_euler('x',2.,degrees=True).as_matrix()
    default,baseline=body.aligned_orientation_prior(np.eye(3),measured)
    assert default is None and baseline['angular_consistency_screen_applied']
    prior,ablation=body.aligned_orientation_prior(np.eye(3),measured,skip_consistency_for_ablation=True)
    np.testing.assert_allclose(prior,np.eye(3))
    assert ablation['diagnostic_ablation'] and not ablation['angular_consistency_screen_applied']
    assert not ablation['angular_prior_certified']
    assert ablation['prior_consistency_limit_deg']==baseline['prior_consistency_limit_deg']
