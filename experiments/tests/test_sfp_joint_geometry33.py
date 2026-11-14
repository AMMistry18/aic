import numpy as np
from scipy.spatial.transform import Rotation
from experiments.sfp_joint_geometry33 import fit,POINTS


def test_joint_port_fit_preserves_tilt_and_public_slot_pitch():
    rotation=Rotation.from_euler('xyz',[179.,.6,13.],degrees=True).as_matrix();position=np.array([.01,.02,.4])
    world=(rotation@POINTS.T).T+position;K=np.array([[1200.,0,640],[0,1200,480],[0,0,1]])
    picks=[]
    for x,y in [(-.08,0),(.08,0),(0,.06)]:
        T=np.eye(4);T[:3,3]=[-x,-y,0];P=K@T[:3];q=(P@np.c_[world,np.ones(8)].T).T
        picks.append(dict(P=P,kps=q[:,:2]/q[:,2:],kp_conf=np.ones(8)))
    result=fit(picks)
    assert result['accepted']
    assert np.linalg.norm(np.asarray(result['position'])-position)<1e-9
    assert np.linalg.norm(np.asarray(result['rotation'])-rotation)<1e-8


def test_one_view_cannot_supply_multiview_geometry():
    assert fit([])['reason']=='insufficient_views'
