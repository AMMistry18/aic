"""Recorded development repeat: terminal tracking defect and bounded correction."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    runs=[('v5: lateral registration','dev_3003_ftactile_v5_translation_force_tactile_translation','#b54b27'),
          ('v6: terminal axial compensation','dev_3003_ftactile_v6_force_tactile_translation','#126b89')]
    fig,axes=plt.subplots(2,1,figsize=(6.8,4),sharex=True,layout='constrained')
    for label,run,color in runs:
        folder=a.evidence/run;policy=json.load(gzip.open(folder/'policy.json.gz','rt'))
        rows=policy['trace'];start=next(r['sim_s'] for r in rows if r['phase']=='terminal_dwell')
        truth=json.load(gzip.open(folder/'verification_trace.json.gz','rt'))
        truth=[r for r in truth if -2<=r['sim_time_s']-start<=9]
        axes[0].plot([r['sim_time_s']-start for r in truth],[r['depth_m']*1000 for r in truth],color=color,label=label,lw=1)
        selected=[r for r in rows if -2<=r['sim_s']-start<=9]
        axes[1].plot([r['sim_s']-start for r in selected],[r.get('terminal_axial_compensation_N',0) for r in selected],color=color,lw=1)
    axes[0].axhspan(45.55,46.05,color='#228833',alpha=.12,label='Unchanged seat-depth tolerance')
    axes[0].set(ylabel='Depth (mm)',ylim=(45.1,46.1),title='Development seed 3003: terminal geometric seat only')
    axes[0].legend(fontsize=7,loc='upper left')
    axes[1].axhline(.25,color='black',ls='--',lw=.7,label='Bound from stiffness × lead cap')
    axes[1].set(xlabel='Time from terminal dwell start (simulation s)',ylabel='Axial support (N)',ylim=(-.01,.28),xlim=(0,9));axes[1].legend(fontsize=7)
    for ax in axes:ax.axvline(0,color='black',lw=.6);ax.grid(alpha=.2)
    fig.savefig(a.evidence/'axial_tracking_development.png',dpi=220);fig.savefig(a.evidence/'axial_tracking_development.pdf');plt.close(fig)

if __name__=='__main__':main()
