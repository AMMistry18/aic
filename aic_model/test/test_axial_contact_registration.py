import numpy as np
from types import SimpleNamespace
from aic_model.insertion.axial_contact_registration import blocked_window, register_sc_shoulder


def trace(*,travel=0.,reaction=.12,advance=.00016,depth=.0155):
    t=np.arange(0, .821, .02)
    return np.c_[t,depth+travel*t/t[-1],depth+.0001+advance*t/t[-1],reaction*t/t[-1]].tolist()


def test_axial_stop_needs_command_motion_and_wrist_reaction():
    assert blocked_window(trace()) is not None
    assert blocked_window(trace(travel=.00016)) is None  # unobstructed descent
    assert blocked_window(trace(reaction=0.)) is None  # no wrist evidence
    assert blocked_window(trace(advance=0.)) is None  # idle robot is not a stop


def test_axial_stop_rejects_discontinuous_or_sparse_observations():
    samples=trace()
    assert blocked_window(samples[::4]) is None
    samples[15][0]=samples[14][0]
    assert blocked_window(samples) is None
    samples=trace();samples[-1][3]=float('nan')
    assert blocked_window(samples) is None


def test_stop_depth_is_measured_not_assigned_to_public_seat():
    result=blocked_window(trace(depth=.01524))
    assert result is not None
    assert abs(result['stop_depth_m']-.01524)<1e-12


def test_repeatable_shoulder_measurement_updates_grasp_and_preserves_loaded_tcp_target():
    # A deterministic spring/rigid-stop fixture tests the estimator/control
    # interface, not Gazebo insertion success. The implementation sees only
    # robot kinematics and reaction force, never this fixture's stop parameter.
    class Fixture:
        def __init__(self):
            self.time=0.;self.tcp=.01556;self.command=.01582;self.stop=.01556
            self._sc_grasp_transform=(np.array([0.,0.,-.00018]),np.eye(3))
            self._parent_node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=round(self.time*1e9))))
        def _tcp(self):return np.array([0.,0.,self.tcp]),np.array([1.,0.,0.,0.])
        def _enforce_action_deadline(self,move):pass
        def sleep_for(self,dt):self.time+=dt
        def get_logger(self):return SimpleNamespace(info=lambda x:None,warn=lambda x:None)
        def set_pose_target(self,move,target,**kwargs):
            self.command=float(target[2]);self.tcp=min(self.command,self.stop)
    p=Fixture()
    c=SimpleNamespace(policy=p,port_pos=np.zeros(3),Rp=np.eye(3),DAMPING=np.eye(6),move_robot=None,
                      config=SimpleNamespace(seat_candidate_depth_m=.0152,force_abort_n=6.,seat_force_cap_n=3.),
                      _last_seat_command_depth=.01564,
                      _tip_pose=lambda:(p._tcp()[0]+p._sc_grasp_transform[0],np.eye(3)),
                      get_observation=lambda:None,_force_magnitude=lambda o:abs(500*(p.command-p.tcp)),
                      _wrench_plug_frame=lambda o:(np.array([0.,0.,-500*(p.command-p.tcp)]),np.zeros(3)),
                      _seating_stiffness=lambda:np.eye(6),
                      _tcp_target=lambda tip,R:np.array(tip)-p._sc_grasp_transform[0],
                      _hold_tip=lambda tip,R:None)
    target=register_sc_shoulder(c,np.eye(3))
    assert target is not None
    assert p._axial_contact_registration['reason']=='repeatable_shoulder_registered'
    assert p._axial_contact_registration['repeat_stop_difference_m']<1e-9
    assert abs(c._tip_pose()[0][2]-.01564)<1e-9
    assert abs(c._tcp_target(target,np.eye(3))[2]-p.command)<1e-9
