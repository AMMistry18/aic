from types import SimpleNamespace
import numpy as np
import pytest
from geometry_msgs.msg import Pose
from aic_model.policy import Policy
from aic_model.insertion import visual_reacquisition as vr


class RetreatPolicy:
    def __init__(self):
        self.p=np.array([.2,.1,.3]);self.time=0.;self._sc_grasp_transform=(np.zeros(3),np.eye(3))
        self.force=np.zeros(6);self.targets=[]
        self._parent_node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=round(self.time*1e9))))
    def _tcp(self):return self.p.copy(),np.array([1.,0,0,0])
    def _wrench_vector(self,observation):return self.force.copy()
    def _enforce_action_deadline(self,move):pass
    def get_logger(self):return SimpleNamespace(info=lambda *x:None)
    def sleep_for(self,dt):self.time+=dt
    def set_pose_target(self,move,pose,**kwargs):
        self.p=np.array([getattr(pose.position,k) for k in 'xyz']);self.targets.append(self.p.copy())
        move()


def test_visual_retreat_is_bounded_and_uses_measured_axis():
    p=RetreatPolicy();origin=p.p.copy()
    assert vr.bounded_retreat(p,lambda:object(),lambda:None,'sc')
    assert np.linalg.norm(p.p-(origin+[0,0,-.01]))<=.00025
    assert max(np.linalg.norm(x-origin) for x in p.targets)<=.01050001
    assert p._visual_reacquisition_events[0]['outcome']=='retreat_completed'


def test_visual_retreat_holds_and_fails_on_force_change():
    p=RetreatPolicy();origin=p.p.copy()
    assert not vr.bounded_retreat(p,lambda:object(),lambda:setattr(p,'force',np.array([4.,0,0,0,0,0])),'sc')
    np.testing.assert_allclose(p.p,origin)
    assert p._visual_reacquisition_events[0]['outcome']=='force_or_tracking_abort'


def test_visual_retreat_requires_measured_grasp_and_bounds():
    p=RetreatPolicy();p._sc_grasp_transform=None
    assert not vr.bounded_retreat(p,lambda:object(),lambda:None,'sc')
    with pytest.raises(ValueError):vr.bounded_retreat(p,lambda:object(),lambda:None,'sc',distance_m=.02)
    assert not p.targets


def test_visual_retreat_rejects_untracked_endpoint():
    class Stuck(RetreatPolicy):
        def set_pose_target(self,move,pose,**kwargs):self.targets.append(self.p.copy())
    p=Stuck()
    assert not vr.bounded_retreat(p,lambda:object(),lambda:None,'sc')
    assert p._visual_reacquisition_events[0]['outcome']=='tracking_timeout'


def test_visual_retreat_corrects_bounded_static_tracking_error():
    class Biased(RetreatPolicy):
        def set_pose_target(self,move,pose,**kwargs):
            super().set_pose_target(move,pose,**kwargs)
            self.p+=np.array([.0003,0.,0.])
    p=Biased();origin=p.p.copy()
    assert vr.bounded_retreat(p,lambda:object(),lambda:None,'sc')
    assert np.linalg.norm(p.p-(origin+[0,0,-.01]))<=.00025
    event=p._visual_reacquisition_events[0]
    assert np.linalg.norm(event['tracking_correction_m'])<=.000500001


def test_command_rejection_propagates_out_of_pose_helper():
    class Dummy(Policy):
        def __init__(self):
            self._parent_node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:None)))
        def get_logger(self):return SimpleNamespace(info=lambda *x:None)
        def insert_cable(self,*args):return False
    # Use a real ROS stamp from its clock interface, as MotionUpdate validates it.
    from builtin_interfaces.msg import Time
    p=Dummy();p._parent_node.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:Time()))
    def rejected(**kwargs):raise RuntimeError('force guard')
    with pytest.raises(RuntimeError,match='force guard'):p.set_pose_target(rejected,Pose())
