import numpy as np
import pytest
from aic_model.insertion.tracking_compensation import LateralTrackingCompensator


def test_unknown_constant_lateral_load_is_rejected_using_own_robot_motion():
    tracker=LateralTrackingCompensator(np.eye(3)[:,:2])
    actual=np.zeros(3);desired=np.array([0.,0.,.01]);load=np.array([.03,-.02,0.])
    for i in range(241):
        force=tracker.update(i*.025,desired,actual,.2)
        # A damped 90 N/m Cartesian plant. Its load is unknown to the policy.
        actual+=.025*((force+load)/90-actual)/.2
    assert np.linalg.norm(actual[:2])<.00002
    assert tracker.force_base[2]==0.
    assert tracker.peak<=.5


def test_contact_and_probe_freeze_prevent_integrating_against_a_wall():
    tracker=LateralTrackingCompensator(np.eye(3)[:,:2])
    tracker.update(0.,np.zeros(3),np.zeros(3),0.)
    tracker.update(.05,np.ones(3)*.001,np.zeros(3),0.)
    before=tracker.integral_base.copy()
    tracker.update(.1,np.ones(3),np.zeros(3),1.)
    np.testing.assert_array_equal(tracker.integral_base,before)
    assert np.linalg.norm(tracker.force_base)<=.50000001
    tracker.update(.15,np.ones(3),np.zeros(3),0.,freeze=True)
    np.testing.assert_array_equal(tracker.integral_base,before)


def test_force_norm_and_clock_gap_are_bounded():
    tracker=LateralTrackingCompensator(np.eye(3)[:,:2])
    tracker.update(0.,np.ones(3),np.zeros(3),0.)
    tracker.update(10.,np.ones(3),np.zeros(3),0.)
    assert np.linalg.norm(tracker.force_base)==pytest.approx(.5)
    assert tracker.force_base[2]==0.
    with pytest.raises(RuntimeError,match='backwards'):tracker.update(9.,np.zeros(3),np.zeros(3),0.)


def test_nonfinite_force_cannot_drive_tracking_compensation():
    tracker=LateralTrackingCompensator(np.eye(3)[:,:2])
    with pytest.raises(RuntimeError,match='Invalid'):tracker.update(0.,np.zeros(3),np.zeros(3),float('nan'))


def test_bounded_proportional_response_limits_transient_after_load_reversal():
    tracker=LateralTrackingCompensator(np.eye(3)[:,:2]);actual=np.zeros(3)
    errors=[]
    for i in range(401):
        t=i*.005;load=np.array([.08 if t<1. else -.08,0.,0.])
        force=tracker.update(t,np.zeros(3),actual,.2)
        actual+=.005*((force+load)/90-actual)/.2
        errors.append(abs(actual[0]))
    assert max(errors)<.00025
    assert tracker.peak<=.5
