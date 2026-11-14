from types import SimpleNamespace
import pytest
from geometry_msgs.msg import Pose
from aic_model.policy import Policy


def send(monkeypatch,value):
    if value is None:monkeypatch.delenv('AIC_INSERT_NATIVE_WRENCH_GAIN',raising=False)
    else:monkeypatch.setenv('AIC_INSERT_NATIVE_WRENCH_GAIN',value)
    parent=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:None)))
    # Use the actual ROS stamp type; no node/executor is needed for this check.
    from builtin_interfaces.msg import Time
    parent.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:Time()))
    messages=[]
    Policy.set_pose_target(SimpleNamespace(_parent_node=parent),lambda **kw:messages.append(kw['motion_update']),Pose())
    return messages[0]


def test_default_preserves_original_native_feedback(monkeypatch):
    assert list(send(monkeypatch,None).wrench_feedback_gains_at_tip)==[.5,.5,.5,0.,0.,0.]


def test_zero_native_feedback_keeps_pose_impedance_command(monkeypatch):
    m=send(monkeypatch,'0')
    assert list(m.wrench_feedback_gains_at_tip)==[0.]*6
    assert m.target_stiffness[0]==90.


@pytest.mark.parametrize('value',['nan','inf','-1','1.1'])
def test_invalid_gain_rejected_before_robot_command(monkeypatch,value):
    with pytest.raises(ValueError,match='feedback gain'):send(monkeypatch,value)
