"""Independent wrist-force measurement from the evaluator's raw FT stream."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def audit_raw_force(raw, first_command, action_end, expected_hz, physics_step, force_ceiling_N=8.):
    """Reject absent/gapped sensor data; never interpolate over an impact."""
    if force_ceiling_N not in (8.,20.):raise ValueError('Unknown prospective force protocol')
    if not raw:return {'qualified':False,'reason':'missing_raw_force_samples'}
    t=np.array([m['header']['stamp']['sec']+m['header']['stamp']['nanosec']/1e9 for m in raw])
    f=np.array([[m['wrench']['force'][k] for k in 'xyz'] for m in raw])
    prior=(t<first_command)&(t>=first_command-.1)
    active=(t>=first_command)&(t<=action_end)
    if prior.sum()<10 or active.sum()<2 or not np.isfinite(f).all():
        return {'qualified':False,'reason':'missing_or_nonfinite_raw_force_coverage'}
    baseline=np.median(f[prior],axis=0);norm=np.linalg.norm(f[active]-baseline,axis=1)
    selected=t[active];dt=np.diff(selected)
    period=1/expected_hz
    covered=bool(np.all(dt>0) and dt.max()<=1.5*max(period,physics_step)+1e-8
        and selected[0]-first_command<=1.5*period and action_end-selected[-1]<=1.5*period)
    peak=float(norm.max());gentle=peak<=force_ceiling_N
    return dict(qualified=covered and gentle,reason='qualified' if covered and gentle else
        ('raw_force_coverage_gap' if not covered else f'raw_force_exceeded_{force_ceiling_N:g}N'),
        force_ceiling_N=force_ceiling_N,samples_over_ceiling=int((norm>force_ceiling_N).sum()),
        raw_samples=len(raw),action_samples=int(active.sum()),expected_hz=expected_hz,
        observed_mean_hz=float((len(selected)-1)/(selected[-1]-selected[0])),
        maximum_action_gap_s=float(dt.max()),complete_action_coverage=covered,
        baseline_N=baseline.tolist(),peak_tared_force_N=peak,p95_tared_force_N=float(np.percentile(norm,95)),
        peak_sim_s=float(selected[np.argmax(norm)]),samples_over_8N=int((norm>8).sum()),
        sensor_frame_ids=sorted({m['header']['frame_id'] for m in raw}),
        force_norm_requires_no_world_or_object_pose=True,
        scope='Sampled physical sensor peak; no interpolation, no object/contact truth inputs')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    policy=json.loads((a.run_dir/'policy.json').read_text());raw=json.load(gzip.open(a.run_dir/'wrench.json.gz','rt'))
    times=np.array([m['header']['stamp']['sec']+m['header']['stamp']['nanosec']/1e9 for m in raw])
    values=np.array([[m['wrench']['force'][k] for k in ('x','y','z')] for m in raw])
    if not len(values) or not np.isfinite(values).all():raise ValueError('Missing/nonfinite measured force')
    trace=policy.get('trace',[]);t0=trace[0]['sim_s'] if trace else policy.get('action_start_sim_s',times[0])
    prior=(times<t0)&(times>=t0-.1)
    baseline=np.median(values[prior] if np.any(prior) else values[:min(10,len(values))],axis=0)
    valid=times>=t0;norm=np.linalg.norm(values[valid]-baseline,axis=1)
    result={'baseline_N':baseline.tolist(),'baseline_window':'last 0.1 sim s before first motion command; first ten samples if unavailable',
        'raw_samples':len(values),'action_samples':int(np.count_nonzero(valid)),
        'peak_independent_tared_force_N':float(norm.max()),'p95_independent_tared_force_N':float(np.percentile(norm,95)),
        'policy_peak_tared_force_N':policy.get('peak_tared_force_N'),'motion_commands_observed':bool(trace)}
    result['action_window_start_source']=('first_motion_command' if trace else
        'recorded_policy_action_start' if 'action_start_sim_s' in policy else 'legacy_first_50hz_sample')
    spec=json.loads((a.run_dir.parent/'scene.json').read_text())
    raw_path=a.run_dir/'wrench_raw.json.gz'
    if spec.get('high_rate_wrist_audit_required'):
        high_raw=json.load(gzip.open(raw_path,'rt')) if raw_path.exists() else []
        truth=json.load(gzip.open(a.run_dir/'truth_trace.json.gz','rt'))
        end=truth[-1]['sim_time_s'] if truth else float(times[-1])
        result['high_rate_wrist_audit']=audit_raw_force(high_raw,t0,end,
            spec['wrist_force_expected_rate_hz'],spec['timestep_s'],spec.get('evaluation_force_ceiling_N',8.))
    (a.run_dir/'wrench_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
