"""Capture-time contract with compute delay; mocked fits are not pose evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import pytest
from aic_model.insertion import sfp_controller

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('aic_model.insertion.exposure40_candidate',ROOT/'.artifacts/repair40/exposure_candidate/sfp_fixture_refinement.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)

def setup(monkeypatch,*,initial=13.4,motion=0.,estimate_stamp=13.2,mutate=False):
    state=N(time=initial,calls=[],detect_calls=0);position=np.array([1.,2.,3.]);quat=np.array([1.,0.,0.,0.])
    views=[N(camera_name=name,stamp_s=13.2,K=np.eye(3),T_world_from_camera=np.eye(4),image_bgr=None) for name in ['left','center','right']]
    monkeypatch.setattr(sfp_controller,'_plug_views_from_observation',lambda *args:views)
    monkeypatch.setattr(sfp_controller,'solve_tip_in_tcp',lambda p,q,*args:(p.copy(),q.copy()))
    def at_stamp(stamp):
        state.calls.append((stamp,state.time))
        if state.time-stamp>10:raise RuntimeError('Requested exposure expired from TF history')
        return position,quat
    def detect(*args):
        state.detect_calls+=1;state.time+=12
        if mutate:position[0]+=.00005
        return []
    measured=N(stamp_s=estimate_stamp,position_world=np.zeros(3),rotation_world_from_plug=np.eye(3),age_s=.2)
    fit=dict(position=[0.,0.,.01],rotation=np.eye(3).tolist())
    monkeypatch.setattr(candidate,'fit_fixture',lambda picks:fit)
    monkeypatch.setattr(candidate,'select_candidate',lambda picks,plug:picks[0])
    monkeypatch.setattr(candidate,'fit_plug_translation',lambda *args:(np.zeros(3),.1))
    monkeypatch.setattr(candidate,'refine_rim',lambda *args:(np.zeros(3),np.eye(3),{}))
    policy=N(_parent_node=N(get_clock=lambda:N(now=lambda:N(nanoseconds=round(state.time*1e9)))),
        _sfp_plug_estimator=N(detect_views=detect,estimate_multiview=lambda *args,**kw:measured),
        _sfp_config=N(plug_max_age_s=.3),_pc=N(detect_nic=lambda *args,**kw:[{'kps':np.zeros((8,2)),'kp_conf':np.ones(8)}]),
        _tcp_at_stamp=at_stamp,_tcp=lambda:(np.array([1.+motion,2.,3.]),quat))
    return policy,state

def test_compute_outlives_tf_history_but_grasp_keeps_correct_exposure_pose(monkeypatch):
    policy,state=setup(monkeypatch,motion=.00005)
    candidate.perceive(policy,object())
    assert state.calls==[(13.2,13.4)]
    np.testing.assert_array_equal(policy._sfp_grasp_transform[0],[1.,2.,3.])
    assert policy._sfp_fixture_diagnostic['exposure_tcp_retained_before_inference']

def test_pose_snapshot_is_copied_instead_of_retaining_mutable_provider_arrays(monkeypatch):
    policy,state=setup(monkeypatch,mutate=True)
    candidate.perceive(policy,object())
    np.testing.assert_array_equal(policy._sfp_grasp_transform[0],[1.,2.,3.])

def test_unavailable_exposure_rejects_before_inference_without_latest_fallback(monkeypatch):
    policy,state=setup(monkeypatch,initial=25.)
    with pytest.raises(ValueError,match='Exposure TCP unavailable'):candidate.perceive(policy,object())
    assert state.detect_calls==0

def test_stationarity_gate_remains_active_after_compute_delay(monkeypatch):
    policy,_=setup(monkeypatch,motion=.001)
    with pytest.raises(ValueError,match='Wrist moved'):candidate.perceive(policy,object())

def test_estimated_stamp_without_captured_pose_cannot_use_current_pose(monkeypatch):
    policy,_=setup(monkeypatch,estimate_stamp=13.3)
    with pytest.raises(ValueError,match='no captured exposure TCP'):candidate.perceive(policy,object())
