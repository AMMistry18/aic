from types import SimpleNamespace as NS
import pytest
from aic_model.insertion.wrist_observation import WristSensorCache


def cache(now):
    node=NS(get_clock=lambda:NS(now=lambda:NS(nanoseconds=int(now*1e9))))
    return WristSensorCache(node,subscribe=False)


def message(stamp,frame='ati/tool_link'):
    return NS(header=NS(stamp=NS(sec=int(stamp),nanosec=round((stamp-int(stamp))*1e9)),frame_id=frame))


def test_force_updates_independently_while_camera_exposure_is_preserved():
    sensor=cache(1.);image=object();combined=NS(wrist_wrench=message(.8),left_image=image)
    stamps=[]
    for t in [.94,.95,.96,.97,.98,.99]:
        sensor.accept(message(t));merged=sensor.merge(combined)
        stamps.append(sensor.stamp(merged.wrist_wrench))
        assert merged.left_image is image
    assert len(set(stamps))==6
    assert sensor.stamp(combined.wrist_wrench)==.8


@pytest.mark.parametrize('stamp,frame',[(.5,'ati/tool_link'),(1.1,'ati/tool_link'),(.99,'unknown')])
def test_stale_future_or_unresolved_frame_cannot_reach_control(stamp,frame):
    sensor=cache(1.);sensor.accept(message(stamp,frame))
    with pytest.raises(RuntimeError):sensor.merge(NS())


def test_out_of_order_sample_cannot_replace_a_newer_wrist_measurement():
    sensor=cache(1.);sensor.accept(message(.99));sensor.accept(message(.95))
    assert sensor.stamp(sensor.merge(NS()).wrist_wrench)==pytest.approx(.99)


def test_real_ros_observation_is_copied_without_mutating_camera_or_old_wrench():
    from aic_model_interfaces.msg import Observation
    from geometry_msgs.msg import WrenchStamped
    observation=Observation();sample=WrenchStamped()
    sample.header.frame_id='ati/tool_link';sample.header.stamp.nanosec=990000000
    sensor=cache(1.);sensor.accept(sample);merged=sensor.merge(observation)
    assert merged is not observation
    assert merged.left_image is observation.left_image
    assert observation.wrist_wrench.header.stamp.nanosec==0
    assert merged.wrist_wrench is sample
