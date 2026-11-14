"""Observable probe contracts in ideal fixtures; not Gazebo insertion results."""
from types import SimpleNamespace
import numpy as np
import pytest
from aic_model.insertion.sfp_contact_datum import DatumController,next_axial_target


class IdealLip(DatumController):
    def __init__(self,*,wall=True,reaction=True):
        self.t=0.;self.p=np.array([0.,0.,-.0015]);self.target=self.p.copy()
        self.Rp=np.eye(3);self.port_pos=np.zeros(3);self.held_rotation=np.eye(3)
        self.wall=wall;self.reaction=reaction;self.diagnostic={'axial_probe_samples':[]}
        self.policy=SimpleNamespace(sleep_for=self.sleep)

    def clock(self):return self.t

    def read(self):
        force=np.array([0.,0.,-500*max(0.,self.target[2]-self.p[2])]) if self.reaction else np.zeros(3)
        return self.p.copy(),np.eye(3),force,self.t

    def send(self,p,R):
        np.testing.assert_allclose(R,self.held_rotation)
        self.target=p.copy()

    def sleep(self,dt):
        self.t+=dt;self.p=self.target.copy()
        if self.wall and abs(self.p[0])>=.001:self.p[2]=min(self.p[2],.0000535)


@pytest.mark.parametrize('sign',[-1,1])
def test_opposed_public_lip_stops_need_command_clipping_and_reaction(sign):
    probe=IdealLip();result=probe.mouth_stop(sign*.002)
    assert abs(result['stop_depth_m']-.0000535)<1e-10
    assert result['reaction_increase_N']>=.025
    assert result['observed_travel_m']<=.000025
    assert result['elastic_lead_m']>=.00015


@pytest.mark.parametrize('wall,reaction',[(False,True),(True,False)])
def test_free_motion_or_reactionless_stop_cannot_register_datum(wall,reaction):
    with pytest.raises(RuntimeError,match='No mouth lip|unobservable'):
        IdealLip(wall=wall,reaction=reaction).mouth_stop(.002)


def test_opposite_probe_uses_measured_clearance_with_biased_visual_mouth():
    probe=IdealLip();probe.port_pos[2]=-.0011
    first=probe.mouth_stop(.002)
    probe.line(probe.p-probe.Rp[:,2]*.00075)
    before=probe.t
    second=probe.mouth_stop(-.002,start_depth_m=first['stop_depth_m']-.00075)
    assert abs(first['stop_depth_m']-second['stop_depth_m'])<.0001
    # Independent measurement remains necessary: the new start reference
    # shortens free travel but does not itself return a contact datum.
    assert second['reaction_increase_N']>=.025
    assert second['command_advance_m']>=.0001
    assert probe.t-before<7


def test_bounded_travel_can_complete_against_declared_damping_before_budget():
    # First-order damped fixture with 0.03 N axial drag. This checks that the
    # bounded command can actually finish; it does not model Gazebo contacts.
    depth=command=.002;time=0.;dt=.02
    while time<32 and depth<.0457:
        command=next_axial_target(depth,command,dt,.03)
        velocity=(500*(command-depth)-.03)/100
        depth+=velocity*dt;time+=dt
    assert depth>=.0457
    assert time<25


def test_terminal_braking_and_force_cap_retain_original_bounds():
    assert next_axial_target(.0445,.045,1.,0.)<=.0448
    assert next_axial_target(.01,.0104,.02,3.)==.0104
    with pytest.raises(RuntimeError):next_axial_target(float('nan'),0.,.02,0.)


def test_waypoint_tracking_is_enabled_and_probe_state_restored():
    probe=IdealLip(wall=False);probe._in_contact_probe=True;flags=[]
    send=probe.send
    def record(p,R):
        flags.append(probe._in_contact_probe);send(p,R)
    probe.send=record
    probe.line(np.array([.001,0.,-.0015]))
    assert flags and not any(flags)
    assert probe._in_contact_probe is True


def test_waypoint_failure_restores_probe_state():
    probe=IdealLip();probe._in_contact_probe=True
    probe.sleep=lambda dt:setattr(probe,'t',probe.t+dt)
    with pytest.raises(RuntimeError,match='waypoint tracking timeout'):
        probe.line(np.array([.001,0.,-.0015]),budget=.1)
    assert probe._in_contact_probe is True
