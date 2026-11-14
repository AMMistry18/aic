"""Source-sensor noise propagation and a conservative scalar moment guard.

This estimates a low-frequency physical moment, not a bound on raw Gaussian
sensor samples. It introduces a finite observation window; raw data and the
instantaneous translational force abort must be retained separately.
"""
import math
import numpy as np

def tip_axis_noise_sigma(axis,sensor_to_tip,force_sigma=.2,torque_sigma=.05):
    axis=np.asarray(axis,dtype=float);offset=np.asarray(sensor_to_tip,dtype=float)
    if axis.shape!=(3,) or offset.shape!=(3,) or not np.isfinite([*axis,*offset]).all():raise ValueError('Finite 3D vectors required')
    if abs(np.linalg.norm(axis)-1)>1e-6:raise ValueError('Unit tip axis required')
    return math.sqrt(torque_sigma**2+force_sigma**2*float(np.cross(axis,offset)@np.cross(axis,offset)))

def moment_guard_estimate(window,tare,source_sigma,z=3.2905267314919255):
    window=np.asarray(window,dtype=float);tare=np.asarray(tare,dtype=float)
    if window.ndim!=1 or tare.ndim!=1 or len(window)<50 or len(tare)<100:
        raise ValueError('Guard requires 50 unique current and 100 stationary tare samples')
    if not np.isfinite(window).all() or not np.isfinite(tare).all() or not np.isfinite(source_sigma) or source_sigma<=0:
        raise ValueError('Finite samples and positive source sigma required')
    # Measured variability can only increase uncertainty above source noise.
    sigma=max(source_sigma,float(np.std(tare,ddof=1)),float(np.std(window,ddof=1)))
    estimate=float(window.mean()-tare.mean())
    se=sigma*math.sqrt(1/len(window)+1/len(tare))
    return {'estimate_Nm':estimate,'upper_abs_bound_Nm':abs(estimate)+z*se,
        'standard_error_Nm':se,'sigma_used_Nm':sigma,'gaussian_two_sided_confidence':.999,
        'window_samples':len(window),'tare_samples':len(tare)}
