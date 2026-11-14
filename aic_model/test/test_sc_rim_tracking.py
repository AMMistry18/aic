from types import SimpleNamespace
import numpy as np
from aic_model.insertion import sc_controller as sc
from aic_model.insertion import sc_rim_refinement as rim


def fixture(stamp):
    task=SimpleNamespace(id='trial',target_module_name='sc_port_2',port_name='sc_port_2')
    header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp,nanosec=0))
    obs=SimpleNamespace(**{f'{c}_image':SimpleNamespace(header=header) for c in ['left','center','right']})
    views={f'{c}_camera':(np.zeros((5,5,3),np.uint8),np.eye(3),np.eye(4)) for c in ['left','center','right']}
    pc=SimpleNamespace(build_projection_matrix=lambda K,T:K@T[:3],detect_sc_pose=lambda *a,**k:(_ for _ in ()).throw(AssertionError('repeated neural detection')))
    policy=SimpleNamespace(_build_views=lambda obs:views,_pc=pc,get_logger=lambda:SimpleNamespace(info=lambda *x:None,warn=lambda *x:None))
    policy._sc_rim_track={'position':np.zeros(3),'rotation':np.eye(3),'key':sc._rim_task_key(task),
                          'seed_reproj_px':2.,'stamps':{name:1. for name in views}}
    return policy,task,obs


def test_repeated_exposure_never_counts_as_another_consensus_sample(monkeypatch):
    monkeypatch.setenv('AIC_SC_RIM_REFINE','1')
    policy,task,obs=fixture(1)
    assert sc.perceive_sc_port_pose(policy,task,obs) is None


def test_fresh_tracking_uses_image_fit_without_repeating_neural_detection(monkeypatch):
    monkeypatch.setenv('AIC_SC_RIM_REFINE','1')
    policy,task,obs=fixture(2)
    monkeypatch.setattr(rim,'refine_with_agreement',lambda *a,**k:((np.array([.00001,0,0]),np.eye(3)),{'reason':'accepted'}))
    result=sc.perceive_sc_port_pose(policy,task,obs)
    assert result is not None
    assert result[2]==2.
    assert policy._sc_rim_diagnostics[-1]['stage']=='tracking'


def test_static_fixture_jump_is_rejected_even_when_views_agree(monkeypatch):
    monkeypatch.setenv('AIC_SC_RIM_REFINE','1')
    policy,task,obs=fixture(2)
    monkeypatch.setattr(rim,'refine_with_agreement',lambda *a,**k:((np.array([.001,0,0]),np.eye(3)),{'reason':'accepted'}))
    assert sc.perceive_sc_port_pose(policy,task,obs) is None
    assert policy._sc_rim_diagnostics[-1]['reason']=='static_fixture_temporal_disagreement'
