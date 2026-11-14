"""Exercise the actual candidate send hook, including base-to-TCP conversion."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('aic_model.insertion.path_tracking_candidate',ROOT/'.artifacts/repair37/waypoint_candidate/tracking_compensation.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def fixture():
    sent=[];quat=np.array([np.sqrt(.5),0,0,np.sqrt(.5)])
    policy=N(_tcp=lambda:(np.zeros(3),quat),_parent_node=N(get_clock=lambda:N(now=lambda:N(nanoseconds=1000000000))))
    controller=N(Rp=np.eye(3),policy=policy,get_observation=lambda:object(),_force_magnitude=lambda _:0.,
                 move_robot=lambda **kw:sent.append(kw),_in_contact_probe=True)
    msg=N(header=N(frame_id='base_link'),wrench_feedback_gains_at_tip=[0.]*6,
          pose=N(position=N(x=0.,y=0.,z=0.)),feedforward_wrench_at_tip=N(force=N(x=1.,y=2.,z=3.)))
    candidate.install(controller)
    return controller,msg,sent


def test_base_damping_force_is_sent_in_tcp_coordinates_and_preserves_support():
    controller,msg,sent=fixture()
    controller._path_damping_force_base_N=np.array([.3,0,.4])
    controller._axial_tracking_force_N=.15;controller._axial_tracking_bound_N=1.;controller._combined_tracking_bound_N=1.2
    controller.move_robot(motion_update=msg)
    force=msg.feedforward_wrench_at_tip.force
    np.testing.assert_allclose([force.x,force.y,force.z],[1.,1.7,3.55],atol=1e-14)
    assert len(sent)==1


@pytest.mark.parametrize('bad',[[.501,0,0],[0,0,float('nan')],[0,0]])
def test_invalid_path_force_is_rejected_before_a_motion_command(bad):
    controller,msg,sent=fixture();controller._path_damping_force_base_N=np.array(bad)
    with pytest.raises(RuntimeError,match='free-path'):controller.move_robot(motion_update=msg)
    assert not sent


def test_existing_controller_without_path_compensation_preserves_its_command():
    controller,msg,sent=fixture();controller.move_robot(motion_update=msg)
    force=msg.feedforward_wrench_at_tip.force
    assert [force.x,force.y,force.z]==[1.,2.,3.]
    assert len(sent)==1


def test_stop_suppresses_old_integral_and_travel_force_but_keeps_payload_support():
    controller,msg,sent=fixture()
    controller.policy._lateral_tracking_compensator.integral_base=np.array([.3,0,0])
    controller._path_damping_force_base_N=np.array([.3,0,.4])
    controller._axial_tracking_force_N=.124;controller._axial_tracking_bound_N=1.;controller._combined_tracking_bound_N=1.2
    controller._tracking_motion_stopped=True
    controller.move_robot(motion_update=msg)
    f=msg.feedforward_wrench_at_tip.force
    assert [f.x,f.y,f.z]==[1.,2.,3.] and len(sent)==1
