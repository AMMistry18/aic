"""Guarded search for entry before assuming that contact brackets a bore.

Inputs are visual geometry, robot tip estimates and wrist force. Axial gain is
robot-observed motion under a rigid grasp, not a simulator insertion event.
"""
import numpy as np


def seek_entry(*, read, send, sleep, clock, check_deadline, axes,
               command_dt=.025, budget_s=40.):
    initial,rotation,_,_=read()
    initial=np.asarray(initial,dtype=float).copy()
    axes=np.asarray(axes,dtype=float)
    started=clock();theta=0.;last=started;captured_since=None;rows=[]
    radius_limit=.003;pitch=.0003;speed=.003;lead=.0005;gain_required=.002
    outcome='search_budget_exhausted'
    while clock()-started<budget_s:
        check_deadline()
        point,_,force,stamp=read()
        point=np.asarray(point,dtype=float);force=np.asarray(force,dtype=float)
        now=clock();dt=min(.05,max(0.,now-last));last=now
        if not np.isfinite([*point,*force,stamp]).all() or np.linalg.norm(force)>6.:
            outcome='force_or_input_guard';break
        relative=axes.T@(point-initial)
        if np.linalg.norm(relative[:2])>radius_limit+.0005 or abs(relative[2])>.0045:
            outcome='measured_workspace_guard';break
        # Continuous forward seeking, always <=0.5 mm ahead of observation.
        # The old fixed 0.8 mm press could never prove a 1.5 mm capture gain.
        if relative[2]>=gain_required:
            if captured_since is None:captured_since=now
            desired=point.copy()
            if now-captured_since>=.3:
                outcome='observed_entry_gain';break
        else:
            captured_since=None
            radius=pitch*theta/(2*np.pi)
            if radius>radius_limit:
                outcome='search_radius_exhausted';break
            lateral=np.array([radius*np.cos(theta),radius*np.sin(theta)])
            # Do not advance the path faster than the robot can follow it.
            if np.linalg.norm(relative[:2]-lateral)<=.00025:
                theta+=speed*dt/max(np.hypot(radius,pitch/(2*np.pi)),pitch/(2*np.pi))
            desired=initial+axes[:,:2]@lateral+axes[:,2]*min(.004,relative[2]+lead)
        delta=desired-point
        target=point+delta*min(1.,lead/max(np.linalg.norm(delta),1e-12))
        send(target,rotation)
        rows.append({'sim_s':float(now),'wrench_stamp_s':float(stamp),
                     'relative_tip_m':relative.tolist(),'force_N':force.tolist(),
                     'requested_relative_m':(axes.T@(target-initial)).tolist()})
        sleep(command_dt)
    # Stop on the measured pose on every exit; never retain a forward press.
    point,_,force,_=read();send(point,rotation)
    return outcome=='observed_entry_gain',{
        'reason':outcome,'sim_s':float(clock()-started),'observed_gain_m':float((axes.T@(point-initial))[2]),
        'required_gain_m':gain_required,'radius_limit_m':radius_limit,'pitch_m':pitch,
        'max_tip_lead_m':lead,'peak_force_N':max([float(np.linalg.norm(r['force_N'])) for r in rows]+[float(np.linalg.norm(force))]),
        'trace':rows,'simulator_object_inputs':False}
