"""Damped drag/deadlock regressions; ideal fixtures, not Gazebo results."""
import numpy as np
import pytest
from aic_model.insertion.sfp_contact_datum import next_axial_target,next_axial_integral
from aic_model.insertion.tracking_compensation import bounded_axial_feedforward


def fixture(load,compensate):
    point=command=.002;integral=0.;terminal=False;settled=None;dt=.02
    for i in range(1601):
        t=i*dt;terminal=terminal or point>.044
        drag=load if isinstance(load,float) else (.03 if point<.02 else .15 if point<.03 else .25)
        command=next_axial_target(point,command,dt,drag,terminal=terminal)
        assert command-point<=(.0003 if terminal else .0008)+1e-12
        if compensate:integral=next_axial_integral(integral,point,command,dt,drag,terminal=terminal)
        point+=(500*(command-point)+integral-drag)/100*dt
        if abs(point-.0458)<=.0001:
            if settled is None:settled=t
            if t-settled>=1.2:return True,t
        else:settled=None
    return False,t


def test_prior_terminal_lead_stalls_under_drag():
    assert not fixture(.2,False)[0]


@pytest.mark.parametrize('load',[.03,.1,.15,.2,.25,'depth_varying'])
def test_bounded_tracking_reaches_terminal_depth_under_unknown_drag(load):
    accepted,time=fixture(load,True)
    assert accepted and time<25


def test_force_cap_freezes_the_axial_estimate():
    assert next_axial_integral(.1,.044,.0443,.02,3.,terminal=True)==.1


def test_combined_feedforward_keeps_total_force_bound_and_rejects_bad_input():
    assert np.linalg.norm(bounded_axial_feedforward([.5,0,0],[0,0,1],.25))<=.5+1e-12
    np.testing.assert_array_equal(bounded_axial_feedforward([.1,0,0],[0,0,1],0),[.1,0,0])
    with pytest.raises(RuntimeError):bounded_axial_feedforward([0,0,0],[0,0,1],.251)
    with pytest.raises(RuntimeError):bounded_axial_feedforward([0,0,0],[0,0,1],float('nan'))
