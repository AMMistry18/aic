import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.tactile_geometry import moment_at_tip,bounded_forward_target

def test_force_applied_at_tip_does_not_masquerade_as_tip_torque():
    offset=np.array([.01,-.02,.226]);force=np.array([1.,-.4,.1]);local_moment=np.array([.01,.02,-.03])
    sensor_moment=local_moment+np.cross(offset,force)
    assert np.linalg.norm(sensor_moment)>.2
    np.testing.assert_allclose(moment_at_tip(force,sensor_moment,offset),local_moment,atol=1e-15)
    R=Rotation.from_euler('xyz',[.3,-.1,.7]).as_matrix()
    np.testing.assert_allclose(moment_at_tip(R@force,R@sensor_moment,R@offset),R@local_moment,atol=1e-15)

def test_blocked_axial_motion_is_bounded_without_erasing_lateral_probe_or_withdrawal():
    axis=np.array([0.,0.,1.]);actual=np.array([1.,2.,3.])
    desired=actual+np.array([.001,-.002,.012])
    command=bounded_forward_target(desired,actual,axis)
    np.testing.assert_allclose(command-actual,[.001,-.002,.0005],atol=1e-15)
    withdrawal=actual-np.array([0.,0.,.01])
    np.testing.assert_array_equal(bounded_forward_target(withdrawal,actual,axis),withdrawal)
    with pytest.raises(ValueError):bounded_forward_target(desired,actual,[0.,0.,0.])
