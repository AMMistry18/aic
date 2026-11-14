import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.wrench_noise import tip_axis_noise_sigma,moment_guard_estimate

def test_shifted_sensor_noise_propagation_matches_independent_samples_and_rotation():
    axis=np.array([.1,.2,np.sqrt(.95)]);offset=np.array([.01,-.02,.22])
    predicted=tip_axis_noise_sigma(axis,offset)
    rng=np.random.default_rng(1729)
    force=rng.normal(0,.2,(100000,3));moment=rng.normal(0,.05,(100000,3))
    measured=(moment-np.cross(offset,force))@axis
    assert abs(measured.std()-predicted)<.0005
    R=Rotation.from_euler('xyz',[.3,-.2,.6]).as_matrix()
    assert tip_axis_noise_sigma(R@axis,R@offset)==pytest.approx(predicted)

def test_guard_retains_0p08_physical_moment_threshold_with_uncertainty_margin():
    # A steady moment at the threshold must trigger even with zero sample
    # scatter: known source uncertainty is never discarded.
    tare=np.zeros(100)
    assert moment_guard_estimate(np.full(50,.08),tare,.05146)['upper_abs_bound_Nm']>.08
    assert moment_guard_estimate(np.zeros(50),tare,.05146)['upper_abs_bound_Nm']<.08
    assert moment_guard_estimate(np.full(50,.06),tare,.05146)['upper_abs_bound_Nm']>.08
    with pytest.raises(ValueError):moment_guard_estimate(np.zeros(49),tare,.05)
    with pytest.raises(ValueError):moment_guard_estimate(np.full(50,np.nan),tare,.05)
