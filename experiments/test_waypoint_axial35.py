"""Ideal damped-load qualification for the reviewable waypoint candidate."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

candidate=os.environ.get('AIC_WAYPOINT_CANDIDATE')
if candidate:
    spec=importlib.util.spec_from_file_location('aic_model.insertion.waypoint_candidate',candidate)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
else:
    from aic_model.insertion import sfp_contact_datum as module


class DampedWaypoint(module.DatumController):
    def __init__(self,load,blocked=False):
        self.t=0.;self.p=np.array([0.,0.,-.0015]);self.target=self.p.copy()
        self.Rp=np.eye(3);self.held_rotation=np.eye(3);self.load=load;self.blocked=blocked
        self.diagnostic={};self.policy=SimpleNamespace(sleep_for=self.sleep)

    def clock(self):return self.t

    def read(self):return self.p.copy(),np.eye(3),np.array([0.,0.,self.load]),self.t

    def send(self,p,R):self.target=p.copy()

    def sleep(self,dt):
        self.p+=(500*(self.target-self.p)-np.array([0.,0.,self.load]))/100*dt
        if self.blocked:self.p[2]=min(-.0015,self.p[2])
        self.t+=dt


@pytest.mark.parametrize('load',[-.06,-.03,0.,.03,.06])
@pytest.mark.parametrize('distance',[-.012,.012])
def test_damped_tracking_reaches_unchanged_waypoint_gate(load,distance):
    controller=DampedWaypoint(load);goal=controller.p+np.array([0.,0.,distance])
    controller.line(goal)
    assert np.linalg.norm(goal-controller.p)<.00004
    assert controller.t<12
    rows=controller.diagnostic['waypoints'][0]['samples']
    assert max(r['pose_lead_m'] for r in rows)<=.0005+1e-12
    assert max(abs(r['axial_integral_N']) for r in rows)<=.15


def test_blocked_waypoint_still_abstains_instead_of_certifying_motion():
    controller=DampedWaypoint(.03,blocked=True)
    with pytest.raises(RuntimeError,match='waypoint tracking timeout'):
        controller.line(controller.p+np.array([0.,0.,.001]))


def test_nonfinite_and_contact_force_cannot_grow_integral():
    assert module.waypoint_axial_integral(.02,.001,.02,1.)==.02
    with pytest.raises(RuntimeError):module.waypoint_axial_integral(0,float('nan'),.02,0)
