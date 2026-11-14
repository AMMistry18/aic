"""A moving wrist must use the camera pose at exposure time."""
from types import SimpleNamespace
import numpy as np
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time as Stamp
from tf2_ros import TransformException
from aic_model.insertion.InsertionPolicy import InsertionPolicy


def test_grasp_tcp_lookup_matches_visual_exposure_after_inference_delay():
    from aic_model.insertion.sfp_controller import solve_tip_in_tcp
    calls=[]
    def lookup(target,source,time,timeout):
        calls.append(time.nanoseconds)
        message=TransformStamped();message.transform.rotation.w=1.
        message.transform.translation.x=time.nanoseconds/1e9
        return message
    policy=InsertionPolicy.__new__(InsertionPolicy)
    policy._parent_node=SimpleNamespace(_tf_buffer=SimpleNamespace(lookup_transform=lookup))
    position,quat=policy._tcp_at_stamp(10.)
    grasp,_=solve_tip_in_tcp(position,quat,np.array([10.02,0.,0.]),np.eye(3))
    np.testing.assert_allclose(grasp,[.02,0.,0.],atol=1e-12)
    assert calls==[10_000_000_000]


def test_camera_projection_uses_each_exposure_time():
    calls=[]
    def lookup(target,source,time,timeout):
        calls.append((target,source,time.nanoseconds))
        msg=TransformStamped();msg.transform.rotation.w=1.
        # Synthetic wrist moving at 1 m/s: latest (10 s) is grossly wrong.
        msg.transform.translation.x=time.nanoseconds/1e9 if time.nanoseconds else 10.
        return msg
    policy=InsertionPolicy.__new__(InsertionPolicy)
    policy._parent_node=SimpleNamespace(_tf_buffer=SimpleNamespace(lookup_transform=lookup))
    policy._get_cam_data=lambda obs,cam:(np.zeros((4,4,3),np.uint8),np.eye(3))
    obs=SimpleNamespace(**{f'{name}_image':SimpleNamespace(header=SimpleNamespace(stamp=Stamp(sec=i+1)))
        for i,name in enumerate(('left','center','right'))})
    views=policy._build_views(obs)
    assert [views[f'{c}_camera'][2][0,3] for c in ('left','center','right')]==[1,2,3]
    assert {t for _,_,t in calls}=={1_000_000_000,2_000_000_000,3_000_000_000}


def test_missing_exposure_transform_drops_view_without_latest_fallback():
    def lookup(*args):raise TransformException('exposure transform unavailable')
    policy=InsertionPolicy.__new__(InsertionPolicy)
    policy._parent_node=SimpleNamespace(_tf_buffer=SimpleNamespace(lookup_transform=lookup))
    policy._get_cam_data=lambda obs,cam:(np.zeros((4,4,3),np.uint8),np.eye(3))
    policy.get_logger=lambda:SimpleNamespace(warn=lambda msg:None)
    obs=SimpleNamespace(**{f'{name}_image':SimpleNamespace(header=SimpleNamespace(stamp=Stamp(sec=1)))
        for name in ('left','center','right')})
    assert policy._build_views(obs)=={}
