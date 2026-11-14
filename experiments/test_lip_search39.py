"""Search/timing controls and actual policy-observation replay, no truth input."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from aic_model.insertion.lip_search import stationary_search_candidate
from aic_model.insertion.qualified_lip_contact import fixed_triplet
from aic_model.insertion.sfp_contact_datum import DatumController


class Lip(DatumController):
    """Ideal contact/no-contact response with independent 0.2 N noise."""
    def __init__(self,seed,*,wall=True,reaction=True):
        self.t=0.;self.p=np.array([.002,0.,-.0015]);self.target=self.p.copy()
        self.Rp=np.eye(3);self.held_rotation=np.eye(3);self.port_pos=np.zeros(3)
        self._axial_tracking_bound_N=1.5;self._axial_tracking_force_N=0.
        self.diagnostic={'axial_probe_samples':[]};self.policy=SimpleNamespace(sleep_for=self.sleep)
        self.wall=wall;self.reaction=reaction;self.rng=np.random.default_rng(seed);self.noise=0.
    def clock(self):return self.t
    def line(self,goal):self.p=goal.copy();self.target=goal.copy()
    def read(self):
        self._last_wrist_stamp_s=self.t
        reaction=max(0.,500*(self.target[2]-self.p[2])+self._axial_tracking_force_N) if self.reaction else 0.
        return self.p.copy(),np.eye(3),np.array([0.,0.,-(reaction+self.noise)]),self.t
    def send(self,p,R):self.target=p.copy()
    def sleep(self,dt):
        self.t+=dt;self.noise=self.rng.normal(0,.2)
        self.p=self.target.copy();self.p[2]+=self._axial_tracking_force_N/500.
        if self.wall:self.p[2]=min(self.p[2],.0000535)


@pytest.mark.parametrize('seed',range(7000,7032))
def test_current_controller_requires_one_observable_reversible_reaction(seed):
    positive=Lip(seed);stop=positive.mouth_stop(.002)
    assert stop['qualified_reversible_excitation']
    assert len(positive.diagnostic['qualified_lip_excitations'])==1
    negative=Lip(seed,reaction=False)
    with pytest.raises(RuntimeError,match='unobservable'):negative.mouth_stop(.002)
    assert negative._axial_tracking_force_N==0.

def test_current_controller_does_not_accept_free_air():
    probe=Lip(7000,wall=False)
    with pytest.raises(RuntimeError,match='No mouth lip'):probe.mouth_stop(.002)


def test_sampling_recovers_after_command_clamps_without_erasing_search():
    t=np.r_[np.arange(0,.4,.02),np.arange(.46,1.5,.02)]
    rows=np.c_[t,np.zeros(len(t)),np.minimum(t*.0004,.0003),np.zeros(len(t))].tolist()
    assert stationary_search_candidate([r for r in rows if r[0]<.8]) is None
    proof=stationary_search_candidate(rows)
    assert proof is not None and proof['command_advance_m']<.0001
    assert proof['search_command_advance_m']>=.0001

@pytest.mark.parametrize('kind',['static_waypoint','moving_free','recent_gap','no_elastic_lead','time_reversal'])
def test_trigger_still_rejects_invalid_or_unloaded_search(kind):
    t=np.arange(0,1.5,.02);command=np.minimum(t*.0004,.0003);point=np.zeros(len(t))
    if kind=='static_waypoint':command[:]=.0003
    if kind=='moving_free':point=command-.00008
    if kind=='recent_gap':t[-1]+=.08
    if kind=='no_elastic_lead':point=command
    if kind=='time_reversal':t[-2]=t[-1]+.01
    assert stationary_search_candidate(np.c_[t,point,command,np.zeros(len(t))].tolist()) is None

def test_actual_failed_search_becomes_ready_only_after_sampling_recovers():
    root=Path(__file__).resolve().parents[1]
    run='dev_45001_causal38v1_original_perception_sfp_contact_datum_handoff'
    p=json.loads((root/'.artifacts/lastinch_runs'/run/run/'policy.json').read_text())
    raw=[s for s in p['sfp_contact_datum']['axial_probe_samples'] if s['offset_m']==-.002]
    rows=[[r['sim_s'],r['measured_depth_m'],r['applied_command_m'],-r['force_port_N'][2]] for r in raw]
    found=None
    for i in range(len(rows)):
        candidate=stationary_search_candidate(rows[:i+1])
        if candidate is not None:found=i;break
    assert found is not None and rows[found][0]-rows[0][0]<6.
    window=np.array([r for r in rows[:found+1] if rows[found][0]-r[0]<=.83])
    assert np.max(np.diff(window[:,0]))<=.05
    assert np.ptp(window[:,1])<=.000025
    assert rows[found][0]<rows[-1][0]-20.

def test_kinematic_clip_without_force_reaction_still_fails_acceptance():
    t=np.arange(21)*.02
    for seed in range(7000,7032):
        rng=np.random.default_rng(seed)
        blocks=[np.c_[t+offset,np.zeros(21),rng.normal(0,.2,21)] for offset in [0.,.6,1.2]]
        result=fixed_triplet(*blocks,low_end=.4,high_end=1.,release_end=1.6,kinematic_blocked=True)
        assert not result['accepted']
