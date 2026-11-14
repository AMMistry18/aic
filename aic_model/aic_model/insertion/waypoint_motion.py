"""Smooth bounded free-space path candidate; no object or simulator inputs."""
import numpy as np


def smooth_waypoint(start,goal,elapsed_s):
    start=np.asarray(start,dtype=float);goal=np.asarray(goal,dtype=float)
    if start.shape!=(3,) or goal.shape!=(3,) or not np.isfinite([*start,*goal,elapsed_s]).all() or elapsed_s<0:
        raise RuntimeError('Invalid free-waypoint path observation')
    delta=goal-start;length=float(np.linalg.norm(delta))
    # Quintic max normalized speed 1.875 and acceleration 10/sqrt(3).
    duration=max(.6,1.875*length/.005,np.sqrt((10/np.sqrt(3))*length/.020))
    u=min(1.,elapsed_s/duration)
    blend=10*u**3-15*u**4+6*u**5
    derivative=30*u*u*(1-u)**2/duration
    return start+delta*blend,delta*derivative,float(duration)
