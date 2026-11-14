"""Offline free-waypoint bias-compensation candidate, not deployed.

Exercises a bounded own-robot axial integral against load-biased damped
tracking. These are ideal control fixtures, never Gazebo insertion results.
"""
import json
import numpy as np


def update_integral(integral_N, error_m, dt_s, force_N):
    if not np.isfinite([integral_N,error_m,dt_s,force_N]).all() or dt_s<0 or force_N<0:
        raise ValueError('Invalid free-waypoint observation')
    if force_N>=1.:return integral_N
    return float(np.clip(integral_N+200.*error_m*min(.04,dt_s),-.15,.15))


def fixture(*,load_N,distance_m,compensate):
    actual=command=integral=0.;dt=.02;settled=None;peak_lead=0.;trace=[]
    for i in range(601):
        t=i*dt
        command+=float(np.clip(distance_m-command,-.002*dt,.002*dt))
        # Integrate only after the nominal command reaches its destination,
        # so damping during travel is not mistaken for static load bias.
        if compensate and abs(command-distance_m)<1e-9:
            integral=update_integral(integral,distance_m-actual,dt,abs(load_N))
        target=actual+float(np.clip(command+integral/500.-actual,-.0005,.0005))
        peak_lead=max(peak_lead,abs(target-actual))
        actual+=(500.*(target-actual)-load_N)/100.*dt
        trace.append((t,actual,command,integral))
        if abs(distance_m-actual)<.00004:
            if settled is None:settled=t
            if t-settled>=.3:
                return dict(passed=True,time_s=t,error_m=distance_m-actual,peak_lead_m=peak_lead,integral_N=integral)
        else:settled=None
    return dict(passed=False,time_s=t,error_m=distance_m-actual,peak_lead_m=peak_lead,integral_N=integral)


def main():
    rows=[]
    for load in [-.06,-.03,0.,.03,.06]:
        for distance in [-.012,.012]:
            original=fixture(load_N=load,distance_m=distance,compensate=False)
            candidate=fixture(load_N=load,distance_m=distance,compensate=True)
            assert candidate['passed'],(load,distance,candidate)
            assert candidate['peak_lead_m']<=.0005+1e-12
            assert abs(candidate['integral_N'])<=.15
            rows.append(dict(load_N=load,distance_m=distance,original=original,candidate=candidate))
    assert update_integral(.02,.001,.02,1.)==.02
    assert update_integral(.149,.01,1.,0.)==.15
    print(json.dumps(dict(ideal_fixture_only=True,production_policy_changed=False,results=rows),indent=2))


if __name__=='__main__':main()
