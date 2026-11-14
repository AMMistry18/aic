"""Bounded, observable lateral grasp-bias correction from opposed contact.

Only controller tip estimates, exposure-derived port geometry and wrist force
are inputs. No simulator contact or object frame is consumed.
"""
import numpy as np


def opposed_midpoint(positive, negative, amplitude):
    hi,fp=positive;lo,fn=negative
    if not np.isfinite([hi,lo,fp,fn,amplitude]).all() or amplitude<=0:
        return None
    # Same clipping/force-difference tests used by the separate diagnostic
    # controller. A free sweep, one-sided blockage or no force contrast is
    # unobservable, even if the requested motion completes. A span no larger
    # than the 0.25 mm terminal lateral tolerance is also insufficient to
    # distinguish two walls from an entrapped plug. This is a necessary
    # observability screen, not proof of correct bore association.
    if not (amplitude-hi>.0001 and lo+amplitude>.0001
            and .00025<hi-lo<2*amplitude-.0002 and abs(fp-fn)>.05):
        return None
    return float((hi+lo)*.5)


def register_lateral(*, read, send, sleep, clock, check_deadline, axes, command_dt=.025,
                     axial_preload_m=.0005,plateau_window_s=.2,wrench_stamp=None):
    """read -> (estimated_tip, estimated_rotation, tared_force_axes, sim_s).

    send receives a bounded tip target and the original held orientation.
    Position plateaus are robot observations, not simulator object truth.
    """
    initial,rotation,_,_=read();origin=initial.copy()
    if not 0<=axial_preload_m<=.0005:raise ValueError('Axial registration preload must be <=0.5 mm')
    if not .2<=plateau_window_s<=.6:raise ValueError('Plateau window must be within 0.2-0.6 seconds')
    center=initial+axes[:,2]*axial_preload_m
    amplitude=.0015;records=[];started=clock()

    def segment(start,goal,duration):
        t0=clock();samples=[]
        while clock()-t0<duration:
            check_deadline()
            if clock()-started>25:
                raise RuntimeError('Contact registration exceeded 25 simulation seconds')
            point,_,force,stamp=read()
            if not np.isfinite(point).all() or not np.isfinite(force).all() or np.linalg.norm(force)>6:
                raise RuntimeError('Contact registration force/input guard')
            if np.linalg.norm(axes[:,:2].T@(point-origin))>.003:
                raise RuntimeError('Measured contact excursion exceeded 3 mm')
            fraction=min(1.,max(0.,(clock()-t0)/max(duration,.001)))
            desired=start+(goal-start)*fraction
            delta=desired-point;norm=np.linalg.norm(delta)
            target=point+delta*min(1.,.0005/max(norm,1e-12))
            if np.linalg.norm(axes[:,:2].T@(target-origin))>.003:
                raise RuntimeError('Contact registration exceeded 3 mm lateral workspace')
            sensor_stamp=stamp if wrench_stamp is None else float(wrench_stamp())
            if not np.isfinite(sensor_stamp):raise RuntimeError('Invalid contact force sensor stamp')
            send(target,rotation);samples.append((stamp,point.copy(),force.copy(),sensor_stamp))
            sleep(command_dt)
        return samples

    plateau_checks=[]
    def settled_plateau(goal):
        rows=segment(goal,goal,.4);began=clock()
        while True:
            rows+=segment(goal,goal,.2)
            end=rows[-1][0]
            unique={r[0]:r for r in rows if r[0]>=end-plateau_window_s}
            recent=list(unique.values())
            if len(recent)>=5:
                t=np.array([r[0] for r in recent]);t-=t.mean()
                positions=np.array([r[1] for r in recent])
                velocity=np.sum(t[:,None]*(positions-positions.mean(0)),axis=0)/max(float(t@t),1e-12)
                force_end=recent[-1][3]
                sensor_unique={r[3]:r for r in recent if r[3]>=force_end-plateau_window_s}
                sensor_rows=list(sensor_unique.values());sensor_times=np.array(sorted(sensor_unique))
                # Keep the legacy caller's sampling contract unchanged. SFP
                # explicitly supplies the actual independent wrench stamp.
                qualified=True
                if wrench_stamp is not None:
                    qualified=(len(sensor_rows)>=max(5,round(plateau_window_s*25))
                        and sensor_times[-1]-sensor_times[0]>=plateau_window_s-.06
                        and np.max(np.diff(sensor_times),initial=0)<=.05
                        and max(r[0]-r[3] for r in recent)<=.05)
                plateau_checks.append({'end_wrench_stamp_s':float(force_end),'end_sim_s':float(end),
                    'unique_samples':len(sensor_rows),'robot_samples':len(recent),
                    'force_window_s':plateau_window_s,'actual_sensor_timestamps':wrench_stamp is not None,
                    'force_sampling_qualified':bool(qualified),'velocity_base_m_s':velocity.tolist()})
                if np.linalg.norm(velocity)<=.0001 and qualified:
                    return sensor_rows
            if clock()-began>=1.6:return None

    segment(initial,center,.5);segment(center,center,.5)
    for axis in range(2):
        offset=axes[:,axis]*amplitude;plus=center+offset;minus=center-offset
        segment(center,plus,1.5)
        positive=settled_plateau(plus)
        segment(plus,minus,3.)
        negative=settled_plateau(minus)
        if positive is None or negative is None:
            reason=('force_samples_missing_or_stale' if plateau_checks and
                not plateau_checks[-1].get('force_sampling_qualified',True) else 'plateau_not_stationary')
            records.append({'axis':axis,'observable':False,'reason':reason,'plateau_checks':plateau_checks.copy()})
            return None,records
        def plateau(rows):
            end=max(r[3] for r in rows);rows=[r for r in rows if r[3]>=end-plateau_window_s]
            forces=np.array([r[2][axis] for r in rows])
            return (float(np.mean([(axes.T@(r[1]-center))[axis] for r in rows])),
                    float(forces.mean()),float(forces.std(ddof=1)/np.sqrt(len(forces))))
        hi,lo=plateau(positive),plateau(negative)
        midpoint=opposed_midpoint(hi[:2],lo[:2],amplitude)
        difference=abs(hi[1]-lo[1]);noise_margin=3*np.hypot(hi[2],lo[2])
        if difference<=noise_margin:midpoint=None
        records.append({'axis':axis,'positive_m_force_N':hi,'negative_m_force_N':lo,
                        'midpoint_m':midpoint,'observable':midpoint is not None,
                        'force_difference_N':difference,'force_difference_3sem_N':float(noise_margin),
                        'plateau_checks':plateau_checks.copy()})
        if midpoint is None:
            return None,records
        old=center.copy();center+=axes[:,axis]*midpoint
        segment(minus,center,1.5);segment(center,center,.5)
    return center,records
