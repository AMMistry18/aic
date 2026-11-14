"""Ideal damped-motion regressions; not physical Gazebo outcomes."""
import numpy as np
import pytest
from aic_model.insertion.sfp_contact_datum import next_axial_target,next_axial_integral,axial_tracking_force
from aic_model.insertion.tracking_compensation import bounded_axial_feedforward


def fixture(profile,load):
    point=command=.002;integral=0.;terminal=False;settled=None;peak=-np.inf
    for i in range(1601):
        t=i*.02;terminal=terminal or point>.044
        drag=load if point<.03 else load*1.5
        command=next_axial_target(point,command,.02,drag,terminal=terminal,profile=profile)
        integral=next_axial_integral(integral,point,command,.02,drag,terminal=terminal,profile=profile)
        ff=axial_tracking_force(integral,command,drag,terminal=terminal,profile=profile)
        assert command-point<=(.0003 if terminal else .0008)+1e-12
        assert abs(ff)<=1.+1e-12
        point+=(500*(command-point)+ff-drag)/100*.02
        peak=max(peak,point)
        if abs(point-.0458)<=.0001:
            if settled is None:settled=t
            if t-settled>=1.2:return t,peak
        else:settled=None
    raise AssertionError('Did not complete dwell')


@pytest.mark.parametrize('load',[.03,.1,.15,.2,.3])
def test_fast_reaches_seat_and_keeps_independent_depth_bound(load):
    time,peak=fixture('fast36',load)
    assert time<17 and peak<.04605


def test_matched_ideal_speed_improvement():
    slow,_=fixture('legacy36',.1)
    fast,_=fixture('fast36',.1)
    assert fast<.8*slow


def test_excitation_stops_at_force_pause_and_final_target():
    assert axial_tracking_force(.2,.04,3.,terminal=False,profile='fast36')==0.
    assert axial_tracking_force(.2,.0458,.1,terminal=True,profile='fast36')==.2
    assert axial_tracking_force(.2,.04,.1,terminal=False,profile='fast36')==.7
    assert np.linalg.norm(bounded_axial_feedforward([.5,0,0],[0,0,1],1.,axial_bound_N=1.,total_bound_N=1.2))<=1.2
    with pytest.raises(RuntimeError):
        bounded_axial_feedforward([0,0,0],[0,0,1],1.01,axial_bound_N=1.,total_bound_N=1.2)
