"""Component removals retain independent axial drive and measurement behavior."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from aic_model.insertion.tracking_compensation import install
from aic_model.insertion.research_ablation import axial_correction


def command_force(mode):
    seen=[]
    policy=NS(_research_ablation=mode,_tcp=lambda:(np.zeros(3),[1.,0.,0.,0.]),
        _parent_node=NS(get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_000_000_000))))
    controller=NS(policy=policy,Rp=np.eye(3),move_robot=lambda **kw:seen.append(kw),
                  get_observation=lambda:object(),_force_magnitude=lambda _:0.,
                  _axial_tracking_force_N=.1)
    message=NS(header=NS(frame_id='base_link'),wrench_feedback_gains_at_tip=[0.]*6,
        pose=NS(position=NS(x=.001,y=0.,z=0.)),feedforward_wrench_at_tip=NS(force=NS(x=0.,y=0.,z=0.)))
    install(controller);controller.move_robot(motion_update=message)
    assert len(seen)==1
    return message.feedforward_wrench_at_tip.force,policy._lateral_tracking_compensator.diagnostics()


def test_lateral_removal_changes_only_lateral_feedback():
    full,fd=command_force('full');removed,rd=command_force('no_lateral_tracking')
    assert full.x>.3 and removed.x==0.
    assert full.z==removed.z==.1
    assert not fd['disabled_by_ablation'] and rd['disabled_by_ablation']


def test_axial_removal_does_not_mutate_the_measured_datum():
    policy=NS(_research_ablation='no_axial_correction');measured=.0003
    assert axial_correction(policy,measured)==0. and measured==.0003
    policy._research_ablation='full';assert axial_correction(policy,measured)==measured
    with pytest.raises(ValueError):axial_correction(policy,float('nan'))
