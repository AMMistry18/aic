"""Damped ideal waypoint checks only; not Gazebo or real-arm stability proof."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from aic_model.insertion.tracking_compensation import LateralTrackingCompensator,bounded_axial_feedforward

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.artifacts/repair37/waypoint_candidate'


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


path=load('aic_model.insertion.waypoint_motion',folder/'waypoint_motion.py')
candidate=load('aic_model.insertion.smooth_candidate',folder/'sfp_contact_datum.py')


class DampedRobot(candidate.DatumController):
    def __init__(self,profile,load):
        self.t=0.;self.p=np.zeros(3);self.target=self.p.copy();self.extra=np.zeros(3)
        self.Rp=np.eye(3);self.held_rotation=np.eye(3);self._motion_profile=profile
        self.policy=SimpleNamespace(sleep_for=self.sleep);self.diagnostic={}
        self.bias=np.asarray(load);self.tracker=LateralTrackingCompensator(self.Rp[:,:2])
        self.leads=[]
    def clock(self):return self.t
    def read(self):return self.p.copy(),np.eye(3),self.bias.copy(),self.t
    def send(self,p,R):
        self.target=p.copy();self.leads.append(np.linalg.norm(p-self.p))
        lateral=self.tracker.update(self.t,p,self.p,float(np.linalg.norm(self.bias)))
        lateral+=getattr(self,'_path_damping_force_base_N',np.zeros(3))
        self.extra=bounded_axial_feedforward(lateral,self.Rp[:,2],getattr(self,'_axial_tracking_force_N',0.),
            axial_bound_N=1.,total_bound_N=1.2)
    def sleep(self,dt):
        for _ in range(10):
            self.p+=(np.array([350.,350.,500.])*(self.target-self.p)+self.extra-self.bias)/np.array([90.,90.,100.])*(dt/10)
        self.t+=dt


@pytest.mark.parametrize('goal',[[0,0,.011],[.004,0,0],[.002,0,.003],[0,0,-.00075]])
@pytest.mark.parametrize('load',[[0,0,.03],[.03,-.02,.1],[.06,.03,-.1]])
def test_measured_arrival_lead_and_force_bounds(goal,load):
    robot=DampedRobot('fast37',load);robot.line(np.array(goal,dtype=float))
    assert np.linalg.norm(robot.p-goal)<.00004
    assert max(robot.leads)<=.0005+1e-12
    assert robot.t<8.
    assert robot._axial_tracking_force_N==0
    np.testing.assert_array_equal(robot._path_damping_force_base_N,np.zeros(3))


def test_smooth_path_preserves_endpoints_velocity_acceleration():
    goal=np.array([.004,-.002,.011]);start=np.zeros(3)
    _,_,duration=path.smooth_waypoint(start,goal,0.)
    rows=[path.smooth_waypoint(start,goal,t) for t in np.linspace(0,duration,1001)]
    np.testing.assert_array_equal(rows[0][0],start)
    np.testing.assert_allclose(rows[-1][0],goal,atol=1e-15)
    np.testing.assert_array_equal(rows[0][1],np.zeros(3));np.testing.assert_array_equal(rows[-1][1],np.zeros(3))
    velocities=np.array([r[1] for r in rows])
    assert np.max(np.linalg.norm(velocities,axis=1))<=.005+1e-12
    assert np.max(np.linalg.norm(np.diff(velocities,axis=0),axis=1)/(duration/1000))<=.02+1e-8


def test_explicit_slow_entry_keeps_the_original_waypoint_branch():
    robot=DampedRobot('fast37',[0,0,.03]);robot.line(np.array([0,0,.002]),speed=.001)
    assert 'trajectory' not in robot.diagnostic['waypoints'][0]


def test_uncompensated_load_still_times_out_instead_of_relaxing_arrival():
    robot=DampedRobot('fast37',[0,0,.3])
    with pytest.raises(RuntimeError,match='tracking timeout'):robot.line(np.array([0,0,.003]))
    assert np.linalg.norm(robot.p-[0,0,.003])>=.00004


def test_matched_ideal_clearance_path_settles_faster():
    times={}
    for profile in ['fast36','fast37']:
        robot=DampedRobot(profile,[0,0,.03]);robot.line(np.array([0,0,-.00075]));times[profile]=robot.t
    assert times['fast37']<.7*times['fast36']


def test_damping_uses_native_base_frame_even_with_rotated_port_axes():
    from scipy.spatial.transform import Rotation
    robot=DampedRobot('fast37',[0,0,0]);robot.Rp=Rotation.from_euler('y',35,degrees=True).as_matrix()
    robot.tracker=LateralTrackingCompensator(robot.Rp[:,:2])
    robot.line(np.array([.001,0,.003]))
    for row in robot.diagnostic['waypoints'][0]['samples']:
        np.testing.assert_allclose(row['damping_force_base_N'],
            np.asarray(robot.DAMPING[:3])*row['scheduled_velocity_base_m_s'],atol=1e-15)


def test_deadline_disables_motion_compensation_before_generic_hold():
    robot=DampedRobot('fast37',[0,0,0]);robot._axial_tracking_force_N=.124
    robot._path_damping_force_base_N=np.array([0,0,.5])
    robot.policy._action_deadline_wall=-1.
    robot.move_robot=lambda **kw:None
    observed=[]
    def deadline(move):
        observed.append((robot._tracking_motion_stopped,robot._axial_tracking_force_N,
                         robot._path_damping_force_base_N.copy()))
        raise TimeoutError('forced deadline fixture')
    robot.policy._enforce_action_deadline=deadline
    with pytest.raises(TimeoutError):robot.guard()
    assert observed[0][0] and observed[0][1]==0.
    np.testing.assert_array_equal(observed[0][2],np.zeros(3))
    assert robot.diagnostic['reason']=='action_deadline'
