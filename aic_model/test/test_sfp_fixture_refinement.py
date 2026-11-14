import numpy as np
import pytest
from aic_model.insertion.sfp_fixture_refinement import select_candidate,fit_fixture,project,FIXED,POINTS


def candidate(p,cameras):
    return dict(position=p,rotation=FIXED.tolist(),cameras=cameras)


def test_single_camera_depth_ghost_cannot_win_nearest_tip_association():
    mouth=np.array([0.,0.,0.]);plug=mouth-FIXED[:,2]*.01
    ghost=candidate((plug+FIXED[:,2]*.003).tolist(),['right'])
    stereo=candidate(mouth.tolist(),['left','center','right'])
    assert select_candidate([ghost,stereo],plug) is stereo


def test_outside_lateral_and_axial_envelope_abstains():
    with pytest.raises(ValueError):select_candidate([candidate([.01,0,0],['left','right'])],np.array([0.,0.,.01]))


def test_calibrated_dual_rim_fit_recovers_public_family():
    p=np.array([.02,.01,.5]);K=np.array([[600.,0,320],[0,600.,240],[0,0,1.]])
    picks=[]
    for x in [-.04,.04]:
        T=np.eye(4);T[0,3]=x;P=K@T[:3]
        picks.append(dict(K=K,T=T,P=P,kp_conf=np.ones(8),kps=project(P,(FIXED@POINTS.T).T+p)))
    result=fit_fixture(picks)
    assert result is not None
    np.testing.assert_allclose(result['position'],p,atol=1e-7)
    np.testing.assert_allclose(result['rotation'],FIXED,atol=1e-7)
