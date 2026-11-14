"""Post-trial diagnostic comparison; simulator truth never enters the controller."""
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
BASE='dev_41004_translation30v1_original_perception_rim_position_body_track_noangular_handoff'


def read(run):
    p=ROOT/'.artifacts/lastinch_runs'/run/run
    q=json.loads((p/'policy.json').read_text())
    v=json.load(gzip.open(p/'verification_trace.json.gz','rt'))
    w=json.load(gzip.open(p/'wrench.json.gz','rt'))
    summary=json.loads((p/'wrench_analysis.json').read_text())
    t=np.array([x['sim_time_s'] for x in v]);depth=np.array([x['depth_m'] for x in v])
    wt=np.array([x['header']['stamp']['sec']+x['header']['stamp']['nanosec']/1e9 for x in w])
    force=np.linalg.norm(np.array([[x['wrench']['force'][k] for k in 'xyz'] for x in w])-summary['baseline_N'],axis=1)
    active=wt>=q['trace'][0]['sim_s'];indices=np.flatnonzero(active);peak=indices[np.argmax(force[active])]
    reached=np.flatnonzero(depth>=.0155)
    seat_t=float(t[reached[0]]) if len(reached) else float(t[-1])
    pre=(t>=seat_t-.12)&(t<=seat_t-.02)
    speed=float(np.polyfit(t[pre]-seat_t,depth[pre]*1000,1)[0]) if pre.sum()>2 else None
    record=dict(run=run,verified=json.loads((p/'verified.json').read_text()),
        peak_independent_force_N=float(force[peak]),peak_sim_s=float(wt[peak]),
        first_depth_15_5_mm_sim_s=seat_t,pre_seat_axial_speed_mm_s_diagnostic=speed,
        speed_window_s_relative_to_first_15_5_mm=[-.12,-.02],
        simulator_truth_used_only_for_diagnostics=True,
        policy_method=q['method'],harness_revision=q['harness_revision'],
        policy_parameters=q['research_controller_parameters'],
        source_sha256={f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in
            ['policy.json','verified.json','verification_trace.json.gz','wrench.json.gz']})
    return record,t,depth,wt,force,seat_t


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repaired-run');a=parser.parse_args()
    runs=[BASE]+([a.repaired_run] if a.repaired_run else [])
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained');rows=[]
    for index,run in enumerate(runs):
        row,t,d,wt,f,seat_t=read(run);rows.append(row)
        label='Baseline: shoulder impact' if index==0 else 'Terminal braking repair'
        axes[0].plot(t-seat_t,d*1000,label=label)
        axes[1].plot(t-seat_t,np.gradient(d*1000,t),label=label)
        axes[2].plot(wt-seat_t,f,label=label)
    axes[0].axhspan(15.39,15.89,color='green',alpha=.1,label='Unchanged depth tolerance')
    axes[0].set_ylabel('Independent depth (mm)');axes[0].legend(fontsize=8)
    axes[0].set_ylim(0,16.1)
    axes[1].set_ylabel('Axial velocity (mm/s)')
    axes[2].set_ylabel('Tared wrist force (N)');axes[2].axhline(8,color='red',ls='--',label='Pre-command abort check')
    axes[2].legend(fontsize=8);axes[2].set_xlabel('Simulation seconds relative to first 15.5 mm depth')
    axes[2].set_xlim(-2,1.5)
    for ax in axes:ax.grid(alpha=.25)
    out=ROOT/'experiments/evidence/original_repair';stem='terminal_braking31_comparison' if a.repaired_run else 'terminal_impact30_diagnosis'
    for suffix in ['png','svg']:fig.savefig(out/f'{stem}.{suffix}',dpi=180)
    (out/f'{stem}.json').write_text(json.dumps(dict(purpose='Development failure diagnosis, not independent final reliability',runs=rows),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k not in ['verified','source_sha256','policy_parameters']} for row in rows],indent=2))


if __name__=='__main__':main()
