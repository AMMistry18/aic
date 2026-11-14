"""Distinct sensor samples, stationarity and unchanged noise-margin controls."""
import numpy as np
from aic_model.insertion.contact_registration import register_lateral


def fixture(noise=.12,stale=False):
    actual=np.array([.00025,-.0001,0.]);bias=np.array([.00045,.0002,0.]);physical_force=np.zeros(3)
    force=np.zeros(3);t=0.;sensor_t=None;rng=np.random.default_rng(609)
    def read():
        nonlocal sensor_t,force
        stamp=np.floor((t+1e-9)/.02)*.02
        if sensor_t is None or stamp>sensor_t:
            sensor_t=stamp;force=physical_force+rng.normal(0,noise,3)
        return actual+bias,np.eye(3),force.copy(),t
    def send(target,R):
        nonlocal actual,physical_force
        desired=target-bias
        actual=np.clip(desired,[-.00035,-.0004,-1],[.00035,.0004,1])
        physical_force=(actual-desired)*400
    def sleep(dt):
        nonlocal t
        t+=dt
    return register_lateral(read=read,send=send,sleep=sleep,clock=lambda:t,
        check_deadline=lambda:None,axes=np.eye(3),command_dt=.01,plateau_window_s=.4,
        wrench_stamp=lambda:0. if stale else sensor_t)


def test_force_samples_count_sensor_exposures_instead_of_loop_reads():
    center,records=fixture()
    assert center is not None
    np.testing.assert_allclose(center[:2],[.00045,.0002],atol=2e-5)
    for record in records:
        assert record['force_difference_N']>record['force_difference_3sem_N']
        for check in record['plateau_checks']:
            assert check['actual_sensor_timestamps'] and check['force_sampling_qualified']
            assert 10<=check['unique_samples']<=22
            assert check['robot_samples']>check['unique_samples']


def test_stale_wrench_cannot_create_a_contact_measurement():
    center,records=fixture(stale=True)
    assert center is None
    assert records[-1]['reason']=='force_samples_missing_or_stale'
    assert not records[-1]['plateau_checks'][-1]['force_sampling_qualified']


def test_high_noise_does_not_pass_by_relaxing_three_sem_margin():
    center,records=fixture(noise=.8)
    assert center is None
    assert records[-1]['force_difference_N']<=records[-1]['force_difference_3sem_N']
