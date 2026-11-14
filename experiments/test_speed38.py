"""Travel-only ideal dynamics and guards; not physical success evidence."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'.artifacts/speed38/candidate/aic_model/aic_model/insertion'
def load(name):
 spec=importlib.util.spec_from_file_location('aic_model.insertion.speed38_'+name,FOLDER/(name+'.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

motion=load('sfp_contact_datum');tracking=load('tracking_compensation')

def ideal(profile,load):
 p=command=.002;integral=0.;terminal=False;settled=None;peak=p
 trace=[]
 for i in range(1601):
  t=i*.02;terminal=terminal or p>.044
  drag=load if p<.03 else load*1.5
  command=motion.next_axial_target(p,command,.02,drag,terminal=terminal,profile=profile)
  integral=motion.next_axial_integral(integral,p,command,.02,drag,terminal=terminal,profile=profile)
  ff=motion.axial_tracking_force(integral,command,drag,terminal=terminal,profile=profile)
  assert command<=.0458 and command-p<=(.0003 if terminal else .0008)+1e-12
  assert abs(ff)<=(1.5 if profile=='fast38' else 1.)+1e-12
  p+=(500*(command-p)+ff-drag)/100*.02;peak=max(peak,p)
  trace.append((t,p,ff))
  if abs(p-.0458)<=.0001:
   if settled is None:settled=t
   if t-settled>=1.2:return t,peak,trace
  else:settled=None
 raise AssertionError('No stable estimated-depth dwell')

@pytest.mark.parametrize('load',[.03,.1,.15,.2,.3])
def test_faster_with_same_pose_lead_and_seat_bounds(load):
 old,_,_=ideal('fast37',load);new,peak,_=ideal('fast38',load)
 assert new<.8*old and peak<.04605

def test_force_pause_is_unchanged_and_removes_damping_drive():
 assert motion.next_axial_target(.03,.0305,.02,3.,profile='fast38')==.0305
 assert motion.next_axial_integral(.1,.03,.0305,.02,3.,terminal=False,profile='fast38')==.1
 assert motion.axial_tracking_force(.1,.03,3.,terminal=False,profile='fast38')==0.
 assert motion.axial_tracking_force(.1,.0458,.1,terminal=True,profile='fast38')==.1

def test_bounded_compensation_and_rejection_beyond_new_bound():
 f=tracking.bounded_axial_feedforward([.5,0,0],[0,0,1],1.5,axial_bound_N=1.5,total_bound_N=1.75)
 assert np.linalg.norm(f)<=1.75
 with pytest.raises(RuntimeError):
  tracking.bounded_axial_feedforward([0,0,0],[0,0,1],1.501,axial_bound_N=1.5,total_bound_N=1.75)

def test_legacy_profiles_are_numerically_unchanged():
 assert motion.motion_parameters('legacy36')==(.003,.0005,.25,False)
 assert motion.motion_parameters('fast37')==(.005,.001,.5,True)
