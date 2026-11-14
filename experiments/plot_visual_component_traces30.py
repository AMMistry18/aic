"""Actual independent traces; truth is diagnostic only, never a policy input."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='dev_3000');p.add_argument('--label',default='component30v1');a=p.parse_args()
    protocol=json.loads((ROOT/'experiments/configs/component_ablation30.json').read_text());fig,axes=plt.subplots(3,1,figsize=(7.1,5.8),sharex=True,layout='constrained');summaries=[]
    labels=['CAD body + lateral PI','CAD body; no lateral PI','Original weighted keypoints + PI; no body correction']
    for method,label in zip(protocol['methods'],labels):
        name=f'{a.case}_{a.label}_{method}';r=ROOT/'.artifacts/lastinch_runs'/name/name
        if not (r/'verified.json').exists():continue
        q=json.loads((r/'policy.json').read_text());v=json.loads((r/'verified.json').read_text());wa=json.loads((r/'wrench_analysis.json').read_text());trace=json.load(gzip.open(r/'verification_trace.json.gz','rt'));w=json.load(gzip.open(r/'wrench.json.gz','rt'))
        t0=q['trace'][0]['sim_s'];t=np.array([x['sim_time_s'] for x in trace])-t0
        axes[0].plot(t,[x['depth_m']*1000 for x in trace],label=label,lw=1)
        axes[1].plot(t,[x['lateral_error_m']*1000 for x in trace],label=label,lw=1)
        wt=np.array([m['header']['stamp']['sec']+m['header']['stamp']['nanosec']*1e-9 for m in w]);force=np.array([[m['wrench']['force'][k] for k in 'xyz'] for m in w]);norm=np.linalg.norm(force-wa['baseline_N'],axis=1)
        keep=(wt>=t0)&(wt<=q['trace'][-1]['sim_s']+.05);axes[2].plot(wt[keep]-t0,norm[keep],label=label,lw=.9)
        peak=int(np.argmax(np.where(keep,norm,-np.inf)));over=np.flatnonzero(keep&(norm>8));stamps=wt[keep];forces=norm[keep]
        summaries.append(dict(run=name,verified=v['verified_success'],force_peak_N=float(norm[peak]),peak_sim_s=float(wt[peak]),last_policy_command_sim_s=q['trace'][-1]['sim_s'],
            first_over_8N_sim_s=float(wt[over[0]]) if len(over) else None,over_8N_samples=len(over),
            force_samples_nominal_sim_interval_s=float(np.median(np.diff(stamps))),
            over_8N_sample_span_s=float(wt[over[-1]]-wt[over[0]]) if len(over) else 0,
            note='Sparse 50Hz sampled load; peak can miss intersample transients. 8N is pre-command abort threshold, not physical force cap.'))
    axes[0].axhspan(15.39,15.89,color='gray',alpha=.15);axes[0].axhline(15.64,color='black',ls=':',lw=.8)
    axes[1].axhline(.25,color='black',ls=':',lw=.8);axes[2].axhline(8,color='black',ls=':',lw=.8)
    for ax,ylabel in zip(axes,['Depth (mm)','Lateral error (mm)','Wrist load (N)']):ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.set_xlim(left=0)
    axes[0].legend(fontsize=7.5,loc='lower right');axes[2].set_xlabel('Simulation time since first policy motion command (s)')
    fig.suptitle(f'{a.case}: actual conditional-insertion component traces',fontsize=11)
    out=ROOT/'experiments/evidence/original_repair';fig.savefig(out/f'{a.case}_{a.label}_traces.png',dpi=200);fig.savefig(out/f'{a.case}_{a.label}_traces.svg');plt.close(fig)
    (out/f'{a.case}_{a.label}_force_audit.json').write_text(json.dumps(summaries,indent=2)+'\n');print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
