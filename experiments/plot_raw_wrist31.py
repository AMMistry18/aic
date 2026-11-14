"""Plot original raw measurements; never revises a trial's stored verdict."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--stem',default='raw_force31_comparison');a=parser.parse_args()
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(9,6),layout='constrained');records=[]
    for run in a.runs:
        p=ROOT/'.artifacts/lastinch_runs'/run/run
        q=json.loads((p/'policy.json').read_text(encoding='utf-8'))
        w=json.loads((p/'wrench_analysis.json').read_text(encoding='utf-8'));audit=w['high_rate_wrist_audit']
        v=json.load(gzip.open(p/'verification_trace.json.gz','rt'))
        t=np.array([r['sim_time_s'] for r in v]);d=np.array([r['depth_m'] for r in v])
        reached=np.flatnonzero(d>=.0155);seat=t[reached[0]] if len(reached) else t[-1]
        raw=json.load(gzip.open(p/'wrench_raw.json.gz','rt'))
        wt=np.array([r['header']['stamp']['sec']+r['header']['stamp']['nanosec']/1e9 for r in raw])
        values=np.array([[r['wrench']['force'][k] for k in 'xyz'] for r in raw])
        force=np.linalg.norm(values-audit['baseline_N'],axis=1)
        active=(wt>=q['trace'][0]['sim_s'])&(wt<=t[-1]);unique=np.unique(wt[active]);gaps=np.diff(unique)
        label=('Braked' if '_brake_' in q['method'] else 'Unbraked')+f" v{q['harness_revision']}"
        axes[0].plot(t-seat,d*1000,label=label)
        # All samples retained, including duplicate publishers. No decimation
        # or averaging that could hide a sub-millisecond impulse.
        axes[1].plot(wt[active]-seat,force[active],lw=.6,label=f'{label}: all raw samples')
        records.append(dict(run=run,stored_verdict=json.loads((p/'verified.json').read_text(encoding='utf-8'))['reason'],
            sampled_50hz_peak_N=w['peak_independent_tared_force_N'],raw_peak_N=audit['peak_tared_force_N'],
            raw_records=int(active.sum()),distinct_timestamps=len(unique),
            distinct_hz=float((len(unique)-1)/(unique[-1]-unique[0])),maximum_distinct_gap_s=float(gaps.max()),
            source_sha256={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in
                ['policy.json','verified.json','wrench_raw.json.gz','wrench_analysis.json','verification_trace.json.gz']}))
    axes[0].axhspan(15.39,15.89,color='green',alpha=.1);axes[0].set_ylabel('Diagnostic depth (mm)');axes[0].set_ylim(0,16)
    axes[1].axhline(8,color='red',ls='--',label='Unchanged 8 N audit limit')
    axes[1].set_ylabel('Raw tared wrist force (N)');axes[1].set_xlabel('Simulation seconds relative to first 15.5 mm depth')
    axes[1].set_xlim(-2,1.5)
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    out=ROOT/'experiments/evidence/original_repair/force_audit31'
    for ext in ['png','svg']:fig.savefig(out/f'{a.stem}.{ext}',dpi=180)
    (out/f'{a.stem}.json').write_text(json.dumps(dict(diagnostic_only_stored_verdicts_unchanged=True,runs=records),indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='source_sha256'} for r in records],indent=2))


if __name__=='__main__':main()
