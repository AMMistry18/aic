"""Impact-decay regression and negative controls, ideal fixtures only."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path('/research')


def load(name,file):
    spec=importlib.util.spec_from_file_location('aic_model.insertion.'+name,file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


folder=Path(os.environ.get('AIC_LIP_RETRY_CANDIDATE_DIR',str(ROOT/'.artifacts/repair36/lip_retry_candidate')))
baseline=load('lip_baseline',folder/'sfp_contact_datum_v29.py')
candidate=load('lip_candidate',folder/'sfp_contact_datum_v30.py')


class ImpactLip(candidate.DatumController):
    def __init__(self,*,wall=True,reaction=True,release=True):
        self.t=0.;self.p=np.array([0.,0.,-.0015]);self.target=self.p.copy();self.Rp=np.eye(3)
        self.port_pos=np.zeros(3);self.held_rotation=np.eye(3);self.contact_time=None;self.impact_velocity=0.
        self.wall=wall;self.reaction=reaction;self.release=release;self.diagnostic={'axial_probe_samples':[]}
        self.policy=SimpleNamespace(sleep_for=self.sleep)

    def clock(self):return self.t

    def read(self):
        force=500*max(0.,self.target[2]-self.p[2])
        if self.contact_time is not None:
            force+=1.*self.impact_velocity/.0004*np.exp(-(self.t-self.contact_time)/.25)
        if not self.reaction:force=0.
        return self.p.copy(),np.eye(3),np.array([0.,0.,-force]),self.t

    def send(self,p,R):self.target=p.copy()

    def sleep(self,dt):
        previous=self.p.copy();self.t+=dt;desired=self.target.copy()
        if self.wall and abs(desired[0])>=.001 and desired[2]>=.0000535:
            if self.contact_time is None:
                self.contact_time=self.t;self.impact_velocity=max(0.,(desired[2]-previous[2])/dt)
            desired[2]=.0000535
        elif self.release or self.contact_time is None:self.contact_time=None
        if not self.release and self.contact_time is not None:desired[2]=.0000535
        self.p=desired


def test_existing_single_press_misses_impact_decay_contact():
    probe=ImpactLip()
    with pytest.raises(RuntimeError,match='unobservable'):
        baseline.DatumController.mouth_stop(probe,.002)


def test_release_then_slower_reload_retains_complete_contact_test():
    probe=ImpactLip();result=probe.mouth_stop(.002)
    assert result['unload_reload_retry_used']
    assert len(probe.diagnostic['lip_retries'])==1
    assert probe.diagnostic['lip_retries'][0]['released_distance_m']>=.00012
    assert result['reaction_increase_N']>=.025
    assert result['command_advance_m']>=.0001
    assert result['observed_travel_m']<=.000025
    assert result['elastic_lead_m']>=.00015
    assert probe.t<28+2


@pytest.mark.parametrize('kwargs',[dict(wall=False),dict(reaction=False),dict(release=False)])
def test_free_air_missing_reaction_or_no_release_cannot_register(kwargs):
    probe=ImpactLip(**kwargs)
    with pytest.raises(RuntimeError,match='No mouth lip|unobservable'):
        probe.mouth_stop(.002)
