"""Recorded stationary sensor audit; no ground-truth state or policy tuning."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tactile_geometry import moment_at_tip
from wrench_noise import tip_axis_noise_sigma,moment_guard_estimate

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    policy=json.loads((a.run_dir/'policy.json').read_text());trace=policy['trace']
    stationary=[r for r in trace if r['phase']=='terminal_dwell']
    if not stationary:raise ValueError('No terminal stationary command phase')
    end=stationary[-1]['sim_s'];start=max(stationary[0]['sim_s']+2,end-4)
    raw=json.load(gzip.open(a.run_dir/'wrench.json.gz','rt'))
    stamp=lambda m:m['header']['stamp']['sec']+m['header']['stamp']['nanosec']/1e9
    raw=[m for m in raw if start<=stamp(m)<=end]
    axis=Rotation.from_quat([.1756266707,-.0115567892,-.0248599222,.9840750466]).as_matrix()[:,2]
    offset=np.array([-.0017771781,-.0188744563,.054722198])+[0,0,.172]
    values=np.array([moment_at_tip([m['wrench']['force'][k] for k in ('x','y','z')],
        [m['wrench']['torque'][k] for k in ('x','y','z')],offset)@axis for m in raw])
    times=np.array([stamp(m) for m in raw]);slope,intercept=np.polyfit(times-times[0],values,1)
    residual=values-(slope*(times-times[0])+intercept);tare=values[:100]
    source=tip_axis_noise_sigma(axis,offset)
    estimates=[moment_guard_estimate(values[i-49:i+1],tare,source) for i in range(149,len(values))]
    result={'purpose':'recorded_sensor_noise_diagnostic_not_an_insertion_trial','run':policy['run'],
        'stationary_command_window_sim_s':[start,end],'sample_count':len(values),
        'source_predicted_tip_axis_sigma_Nm':source,'measured_detrended_sigma_Nm':float(residual.std(ddof=1)),
        'measured_moment_drift_Nm_s':float(slope),'detrended_lag1_correlation':float(np.corrcoef(residual[:-1],residual[1:])[0,1]),
        'raw_0p08_exceedances_after_first100_tare':int(np.count_nonzero(abs(values[100:]-tare.mean())>.08)),
        'raw_samples_after_tare':max(0,len(values)-100),
        'proposed_guard_estimates':len(estimates),'proposed_guard_stops':sum(e['upper_abs_bound_Nm']>.08 for e in estimates),
        'recorded_trace':[{'sim_s':float(t),'tared_tip_axis_moment_Nm':float(v-tare.mean())} for t,v in zip(times,values)],
        'retrospective_estimator_trace':[{'sim_s':float(times[149+i]),**e} for i,e in enumerate(estimates)],
        'caveat':'No simulator truth is used. Stationary command does not prove zero physical moment; trend and autocorrelation are reported. Proposed guard introduces a 1 s window and is not a raw-sample bound.'}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if not k.endswith('trace')},indent=2))

if __name__=='__main__':main()
