"""Motion integration and measurement regressions; not insertion evidence."""
import ast
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT/'experiments/recovery_20261112/speed43_candidate'


def functions(root, name, selected):
    tree=ast.parse((root/'aic_model/aic_model/insertion'/name).read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in selected]
    namespace={'np':np}
    exec(compile(tree,name,'exec'),namespace)
    return namespace


MOTION=['motion_parameters','next_axial_target','next_axial_integral','axial_tracking_force']
motion=functions(CANDIDATE,'sfp_contact_datum.py',MOTION)
baseline=functions(ROOT,'sfp_contact_datum.py',MOTION)
waypoint=functions(CANDIDATE,'waypoint_motion.py',['smooth_waypoint'])['smooth_waypoint']


def integrate(profile,drag):
    depth=command=.002; integral=0.; settled=None; terminal=False; peak=depth
    for i in range(1601):
        t=i*.02; terminal=terminal or depth>.044
        command=motion['next_axial_target'](depth,command,.02,drag,terminal=terminal,profile=profile)
        integral=motion['next_axial_integral'](integral,depth,command,.02,drag,terminal=terminal,profile=profile)
        force=motion['axial_tracking_force'](integral,command,drag,terminal=terminal,profile=profile)
        assert command<=.0458 and command-depth<=(.0003 if terminal else .0008)+1e-12
        assert abs(force)<=(3. if profile=='fast43' else 1.5)+1e-12
        depth+=(500*(command-depth)+force-drag)/100*.02
        peak=max(peak,depth)
        if abs(depth-.0458)<=.0001:
            if settled is None:settled=t
            if t-settled>=1.2:return t,peak
        else:settled=None
    raise AssertionError('No dwell in bounded damped integration')


@pytest.mark.parametrize('drag',[.03,.15,.3,.45])
def test_faster_damped_travel_without_depth_overshoot(drag):
    old,_=integrate('fast38',drag); new,peak=integrate('fast43',drag)
    assert new<old*.85 and peak<.04605


def test_force_pause_removes_drive_and_retains_command():
    for profile,threshold in [('fast38',3.),('fast43',6.)]:
        assert motion['next_axial_target'](.03,.0305,.02,threshold,profile=profile)==.0305
        assert motion['axial_tracking_force'](.2,.03,threshold,terminal=False,profile=profile)==0.


def test_original_profiles_unchanged_numerically():
    for profile in ['legacy36','fast36','fast37','fast38']:
        for force in [0.,1.,3.,7.]:
            assert motion['motion_parameters'](profile)==baseline['motion_parameters'](profile)
            for key,args,kw in [
                ('next_axial_target',(.03,.0302,.02,force),{}),
                ('next_axial_integral',(.1,.03,.0302,.02,force),{'terminal':False}),
                ('axial_tracking_force',(.1,.03,force),{'terminal':False})]:
                assert motion[key](*args,profile=profile,**kw)==baseline[key](*args,profile=profile,**kw)


def test_quintic_path_obeys_velocity_and_acceleration_limits():
    for distance in [.0002,.002,.01,.04]:
        a=np.zeros(3);b=np.array([distance,0.,0.])
        _,_,duration=waypoint(a,b,0,max_speed=.010,max_acceleration=.040)
        times=np.linspace(0,duration,2001)
        rows=[waypoint(a,b,t,max_speed=.010,max_acceleration=.040) for t in times]
        velocities=np.array([r[1] for r in rows]); acceleration=np.diff(velocities,axis=0)/(times[1]-times[0])
        assert np.linalg.norm(velocities,axis=1).max()<=.010+1e-10
        assert np.linalg.norm(acceleration,axis=1).max()<=.040+1e-8
        assert np.allclose(rows[-1][0],b) and np.allclose(rows[-1][1],0)


def test_contact_observability_functions_unchanged():
    # Motion tuning must not weaken the pure opposed-wall measurement decision.
    original=functions(ROOT,'contact_registration.py',['opposed_midpoint'])['opposed_midpoint']
    candidate=functions(CANDIDATE,'contact_registration.py',['opposed_midpoint'])['opposed_midpoint']
    for plus,minus in [((.0005,1.),(-.0005,-1.)),((.0015,0.),(-.0015,0.)),
                       ((.00005,1.),(-.00005,-1.)),((.0005,0.),(-.0005,0.))]:
        assert original(plus,minus,.0015)==candidate(plus,minus,.0015)
