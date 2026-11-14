from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from aic_model.insertion.sc_controller import ScInsertionController


def test_sc_force_uses_wrist_axes_and_moment_is_shifted_to_visual_tip():
    tcp=np.array([.1,.2,.3]);R=Rotation.from_euler('z',90,degrees=True).as_matrix()
    q=Rotation.from_matrix(R).as_quat();quat=np.r_[q[3],q[:3]]
    tip_in_tcp=np.array([.012,-.004,.023]);sensor_to_tip=tip_in_tcp+[0,0,.172]
    force=np.array([3.,-2.,1.]);moment=np.cross(sensor_to_tip,force)
    controller=ScInsertionController.__new__(ScInsertionController)
    controller.Rp=Rotation.from_euler('z',.3).as_matrix()
    controller.policy=SimpleNamespace(_tcp=lambda:(tcp,quat),_wrench_vector=lambda obs:np.r_[force,moment],_wrench_baseline=np.zeros(6))
    # A different plug orientation must not change the sensor force's axes.
    controller._tip_pose=lambda:(tcp+R@tip_in_tcp,R@Rotation.from_euler('x',.7).as_matrix())
    observation=SimpleNamespace(wrist_wrench=SimpleNamespace(header=SimpleNamespace(frame_id='ati/tool_link')))
    f,m=controller._wrench_plug_frame(observation)
    np.testing.assert_allclose(f,controller.Rp.T@R@force,atol=1e-12)
    np.testing.assert_allclose(m,np.zeros(3),atol=1e-12)


def test_sc_does_not_invent_torque_origin_for_unknown_sensor_frame():
    controller=ScInsertionController.__new__(ScInsertionController);controller.Rp=np.eye(3)
    controller.policy=SimpleNamespace(_tcp=lambda:(np.zeros(3),np.array([1.,0,0,0])),
        _wrench_vector=lambda obs:np.ones(6),_wrench_baseline=np.zeros(6))
    observation=SimpleNamespace(wrist_wrench=SimpleNamespace(header=SimpleNamespace(frame_id='unknown')))
    force,moment=controller._wrench_plug_frame(observation)
    np.testing.assert_allclose(force,np.ones(3));assert np.isnan(moment).all()
