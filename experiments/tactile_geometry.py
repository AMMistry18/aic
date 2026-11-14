"""ROS-free mechanics for the development tactile controller."""
import numpy as np

def moment_at_tip(force_sensor,moment_sensor,sensor_to_tip):
    """All vectors are expressed in the sensor frame, lengths in metres."""
    return np.asarray(moment_sensor)-np.cross(sensor_to_tip,force_sensor)

def bounded_forward_target(desired,actual,insertion_axis,max_lead_m=.0005):
    axis=np.asarray(insertion_axis,dtype=float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-6:
        raise ValueError('Insertion axis must be a finite unit vector')
    if not np.isfinite(max_lead_m) or max_lead_m<=0:raise ValueError('Positive finite lead bound required')
    desired=np.asarray(desired,dtype=float);actual=np.asarray(actual,dtype=float)
    return desired-axis*max(0.,float((desired-actual)@axis)-max_lead_m)
