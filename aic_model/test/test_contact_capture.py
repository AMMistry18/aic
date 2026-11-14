import numpy as np
import pytest
from aic_model.insertion.contact_capture import seek_entry


def simulate(opening=True,force=0.):
    point=np.zeros(3);t=0.;commands=[]
    def read():return point.copy(),np.eye(3),np.array([force,0.,0.]),t
    def send(target,R):
        nonlocal point
        commands.append((point.copy(),target.copy()))
        # Hidden opening geometry belongs to the test double, not the policy.
        enters=opening and np.linalg.norm(target[:2]-[.0006,-.0004])<.00028
        point=target.copy()
        if not enters:point[2]=min(point[2],0.)
    def sleep(dt):
        nonlocal t
        t+=dt
    accepted,diagnostic=seek_entry(read=read,send=send,sleep=sleep,clock=lambda:t,
        check_deadline=lambda:None,axes=np.eye(3),budget_s=12.)
    return accepted,diagnostic,commands


def test_search_finds_hidden_opening_by_robot_observed_entry_motion():
    accepted,d,commands=simulate()
    assert accepted and d['observed_gain_m']>=.002
    assert d['reason']=='observed_entry_gain'
    assert not d['simulator_object_inputs']
    assert max(np.linalg.norm(goal-start) for start,goal in commands)<=.000500001


def test_blocked_sweep_does_not_claim_capture_from_lateral_motion():
    accepted,d,commands=simulate(opening=False)
    assert not accepted and d['observed_gain_m']==pytest.approx(0.)
    assert d['reason']=='search_budget_exhausted'
    np.testing.assert_allclose(commands[-1][0],commands[-1][1])


def test_force_guard_retains_only_measured_pose_hold():
    accepted,d,commands=simulate(force=7.)
    assert not accepted and d['reason']=='force_or_input_guard'
    assert len(commands)==1
    np.testing.assert_allclose(commands[0][0],commands[0][1])
