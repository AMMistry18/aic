"""Candidate controller integration in ideal noisy fixtures, not Gazebo."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.artifacts/repair36/pulse_lip_candidate'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


load('aic_model.insertion.qualified_lip_contact',folder/'qualified_lip_contact.py')
candidate=load('aic_model.insertion.pulse_lip_candidate',folder/'sfp_contact_datum.py')


class Lip(candidate.DatumController):
    def __init__(self,seed,*,wall=True,reaction=True):
        self.t=0.;self.p=np.array([.002,0.,-.0015]);self.target=self.p.copy()
        self.Rp=np.eye(3);self.held_rotation=np.eye(3);self.port_pos=np.zeros(3)
        self._axial_tracking_bound_N=1.;self._axial_tracking_force_N=0.
        self.diagnostic={'axial_probe_samples':[]};self.policy=SimpleNamespace(sleep_for=self.sleep)
        self.wall=wall;self.reaction=reaction;self.rng=np.random.default_rng(seed)
        self.noise=.0

    def clock(self):return self.t
    def line(self,goal):self.p=goal.copy();self.target=goal.copy()
    def read(self):
        self._last_wrist_stamp_s=self.t
        force=max(0.,500*(self.target[2]-self.p[2])+self._axial_tracking_force_N) if self.reaction else 0.
        return self.p.copy(),np.eye(3),np.array([0.,0.,-(force+self.noise)]),self.t
    def send(self,p,R):self.target=p.copy()
    def sleep(self,dt):
        self.t+=dt;self.noise=self.rng.normal(0,.2)
        self.p=self.target.copy();self.p[2]+=self._axial_tracking_force_N/500.
        if self.wall:self.p[2]=min(self.p[2],.0000535)


@pytest.mark.parametrize('seed',range(7000,7032))
def test_noisy_lip_is_qualified_once_with_reversible_excitation(seed):
    probe=Lip(seed)
    stop=probe.mouth_stop(.002)
    assert stop['qualified_reversible_excitation']
    assert abs(stop['stop_depth_m']-.0000535)<1e-12
    assert len(probe.diagnostic['qualified_lip_excitations'])==1
    assert probe._axial_tracking_force_N==0.


@pytest.mark.parametrize('seed',range(7000,7032))
def test_kinematic_clipping_without_reaction_cannot_qualify(seed):
    probe=Lip(seed,reaction=False)
    with pytest.raises(RuntimeError,match='unobservable'):probe.mouth_stop(.002)
    assert probe._axial_tracking_force_N==0.


def test_free_air_never_qualifies():
    probe=Lip(7000,wall=False)
    with pytest.raises(RuntimeError,match='No mouth lip'):probe.mouth_stop(.002)
