from types import SimpleNamespace
import numpy as np
from aic_model.insertion import sc_controller as sc


def make_policy(monkeypatch,results,exposures):
    monkeypatch.setenv('AIC_SC_BODY_FRESH_RETRY','1')
    original=np.array([.001,.002,.003]);counter=[];observations=iter(exposures)
    policy=SimpleNamespace(_sc_grasp_transform=(original.copy(),np.eye(3)),
                           sleep_for=lambda t:None,_body_retry_observation=lambda:next(observations,None))
    monkeypatch.setattr(sc,'_sc_camera_frame_stamps',lambda obs:{'camera':obs})
    def once(p,o,R):
        accepted,shift=results[len(counter)];counter.append(o)
        p._sc_body_diagnostic=dict(accepted=accepted,reason='accepted' if accepted else 'leave_one_view_out_disagreement')
        if accepted:p._sc_grasp_transform=(original+shift,np.eye(3))
        return accepted
    monkeypatch.setattr(sc,'_refine_sc_handoff_translation_once',once)
    return policy,original,counter


def test_fresh_retry_requires_two_consistent_accepted_grasps(monkeypatch):
    p,old,seen=make_policy(monkeypatch,[(False,0),(True,.00002),(True,.00003)],[2,3])
    assert sc.refine_sc_handoff_translation(p,1,np.eye(3))
    assert seen==[1,2,3]
    assert np.allclose(p._sc_grasp_transform[0],old+.000025)


def test_cached_capture_cannot_count_as_new_evidence(monkeypatch):
    p,old,seen=make_policy(monkeypatch,[(False,0),(True,.00002),(True,.00003)],[2,2])
    assert not sc.refine_sc_handoff_translation(p,1,np.eye(3))
    assert seen==[1,2]
    assert np.array_equal(p._sc_grasp_transform[0],old)
    assert p._sc_body_diagnostic['reason']=='fresh_retry_not_confirmed'


def test_retry_does_not_accept_inconsistent_grasp_candidates(monkeypatch):
    p,old,seen=make_policy(monkeypatch,[(False,0),(True,.00002),(True,.0003),(False,0)],[2,3,4])
    assert not sc.refine_sc_handoff_translation(p,1,np.eye(3))
    assert np.array_equal(p._sc_grasp_transform[0],old)


def test_partial_candidate_still_needs_fresh_confirmation_and_is_not_full_pose_acceptance(monkeypatch):
    monkeypatch.setenv('AIC_SC_BODY_FRESH_RETRY','1')
    monkeypatch.setattr(sc,'_sc_camera_frame_stamps',lambda o:{'camera':o})
    old=np.array([.001,.002,.003]);seen=[]
    p=SimpleNamespace(_sc_grasp_transform=(old.copy(),np.eye(3)),sleep_for=lambda t:None,
                      _body_retry_observation=lambda:2)
    def once(policy,obs,R):
        seen.append(obs);policy._sc_grasp_transform=(old+.00002,np.eye(3))
        policy._sc_body_diagnostic=dict(accepted=False,lateral_control_candidate_used=True,axial_measurement_pending=True)
        return True
    monkeypatch.setattr(sc,'_refine_sc_handoff_translation_once',once)
    assert sc.refine_sc_handoff_translation(p,1,np.eye(3))
    assert seen==[1,2]
    assert not p._sc_body_diagnostic['accepted']
    assert p._sc_body_diagnostic['axial_measurement_pending']
