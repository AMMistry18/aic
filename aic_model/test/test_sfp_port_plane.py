import numpy as np
from scipy.spatial.transform import Rotation
from aic_model.insertion.InsertionPolicy import InsertionPolicy,LOCAL_SFP_PORT_KPS
from aic_model.insertion.contract import quat_to_matrix


def test_full_port_frame_preserves_source_cage_tilt(monkeypatch):
    monkeypatch.setenv('AIC_SFP_FULL_PORT_ORIENTATION','1')
    rotation=Rotation.from_euler('xyz',[179.3,.4,23.],degrees=True).as_matrix()
    points=(rotation@LOCAL_SFP_PORT_KPS.T).T+np.array([.02,-.1,.18])
    quat,yaw=InsertionPolicy._estimate_sfp_port_orientation(None,points)
    assert np.linalg.norm(quat_to_matrix(quat)-rotation)<1e-10


def test_legacy_port_plane_behavior_remains_opt_in(monkeypatch):
    monkeypatch.delenv('AIC_SFP_FULL_PORT_ORIENTATION',raising=False)
    rotation=Rotation.from_euler('xyz',[179.3,.4,23.],degrees=True).as_matrix()
    points=(rotation@LOCAL_SFP_PORT_KPS.T).T
    quat,_=InsertionPolicy._estimate_sfp_port_orientation(None,points)
    assert np.allclose(quat_to_matrix(quat)[:,2],[0,0,-1])
