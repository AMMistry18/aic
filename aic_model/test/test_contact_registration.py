import numpy as np
import pytest
from aic_model.insertion.contact_registration import register_lateral,opposed_midpoint


def test_free_or_one_sided_motion_does_not_localize():
    assert opposed_midpoint((.0015,0),(-.0015,0),.0015) is None
    assert opposed_midpoint((.0003,-.4),(-.0015,0),.0015) is None
    assert opposed_midpoint((.0003,-.4),(-.0003,.4),.0015)==pytest.approx(0.)


def test_entrapped_contact_cannot_calibrate_a_bore_midpoint():
    # Recorded failure shape: force changes, but only 68 um motion on one side
    # of the starting point. Do not equate this with two observed bore walls.
    assert opposed_midpoint((.001238,.10),(.001170,.50),.0015) is None
    assert opposed_midpoint((.000219,-.42),(.000005,.15),.0015) is None


def test_opposed_walls_recover_relative_bias_without_known_grasp_input():
    # A test double supplies contact-limited robot observations. The routine
    # receives no wall locations or hidden bias; it must infer the midpoint.
    actual=np.array([.00025,-.0001,0.]);bias=np.array([.00045,.0002,0.]);force=np.zeros(3);t=0.
    def read():return actual+bias,np.eye(3),force.copy(),t
    def send(target,R):
        nonlocal actual,force
        desired=target-bias
        actual=np.clip(desired,[-.00035,-.0004,-1],[.00035,.0004,1])
        force=(actual-desired)*400
    def sleep(dt):
        nonlocal t
        t+=dt
    center,records=register_lateral(read=read,send=send,sleep=sleep,clock=lambda:t,
                                     check_deadline=lambda:None,axes=np.eye(3))
    np.testing.assert_allclose(center[:2],bias[:2],atol=2e-5)
    assert center[2]==pytest.approx(.0005)
    assert all(r['observable'] for r in records)
    assert t<25


def test_force_guard_aborts_before_a_command():
    def read():return np.zeros(3),np.eye(3),np.array([7.,0,0]),0.
    with pytest.raises(RuntimeError,match='force/input'):
        register_lateral(read=read,send=lambda *x:pytest.fail('command after force limit'),sleep=lambda x:None,
                         clock=lambda:0.,check_deadline=lambda:None,axes=np.eye(3))


def test_slow_free_tracking_is_not_mistaken_for_opposed_contact():
    actual=np.zeros(3);t=0.
    def read():return actual.copy(),np.eye(3),np.array([.2,.1,0.]),t
    def send(target,R):
        nonlocal actual
        actual=actual+.0625*(target-actual)
    def sleep(dt):
        nonlocal t
        t+=dt
    center,records=register_lateral(read=read,send=send,sleep=sleep,clock=lambda:t,
        check_deadline=lambda:None,axes=np.eye(3))
    assert center is None
    assert not records[-1]['observable']


@pytest.mark.parametrize('depth,accepted,should_hold',[(.002,True,False),(.0153,False,False),(.0153,True,True)])
def test_terminal_centering_requires_deep_entry_and_observable_contact(monkeypatch,depth,accepted,should_hold):
    from types import SimpleNamespace
    from aic_model.insertion.sc_controller import ScInsertionController
    from aic_model.insertion import observable_terminal
    monkeypatch.setenv('AIC_INSERT_OBSERVABLE_TERMINAL','1')
    monkeypatch.setenv('AIC_SC_TERMINAL_CONTACT_REGISTER','1')
    controller=ScInsertionController.__new__(ScInsertionController)
    controller._terminal_contact_registered=False
    controller.config=SimpleNamespace(seat_candidate_depth_m=.0152)
    controller.Rs=np.eye(3);controller._errors=lambda:(depth,None,None,None,None)
    controller.log=SimpleNamespace(warn=lambda *x:None)
    calls=[];holds=[]
    controller._contact_register_grasp=lambda **kw:calls.append(kw) or accepted
    monkeypatch.setattr(observable_terminal,'hold_candidate',lambda *args:holds.append(args) or 'candidate')
    outcome=controller._wait_for_insertion_event(np.zeros(3))
    assert bool(holds)==should_hold
    assert bool(calls)==(depth>=.0152)
    if calls:assert calls[0]['axial_preload_m']==0.
    if should_hold:
        assert outcome=='candidate'
        controller._wait_for_insertion_event(np.zeros(3))
        assert len(calls)==1
