"""Plot measured development trajectories; no aggregation of policy versions."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(7.2,6.3),sharex=True,constrained_layout=True)
    quantities=[('depth_m',1000,'Insertion depth (mm)'),('lateral_error_m',1000,'Lateral error (mm)'),('angular_error_deg',1,'Angle error (deg)')]
    for name in a.runs:
        path=ROOT/'.artifacts/lastinch_runs'/name/name;policy=json.loads((path/'policy.json').read_text())
        data=json.load(gzip.open(path/'verification_trace.json.gz','rt'));start=policy['trace'][0]['sim_s']
        t=np.array([row['sim_time_s']-start for row in data]);selected=t>=0
        verdict=json.loads((path/'verified.json').read_text())
        label=name.removeprefix('sc_supported_').split('_original')[0]+(' (pass)' if verdict['verified_success'] else ' (fail)')
        for ax,(key,scale,ylabel) in zip(axes,quantities):
            values=np.array([row[key]*scale for row in data]);ax.plot(t[selected],values[selected],label=label,lw=1.2)
            ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    axes[0].axhspan(15.39,15.89,color='green',alpha=.15,label='Depth acceptance band')
    axes[0].axhline(15.64,color='green',lw=.7,ls='--')
    for ax in axes[1:]:ax.axhspan(0,.25,color='green',alpha=.15);ax.set_ylim(bottom=0)
    axes[-1].set_xlabel('Simulation seconds after first policy motion command')
    axes[0].legend(fontsize=8,ncol=2,loc='lower right')
    fig.suptitle('Independent measurements: separate development variants',fontsize=11)
    fig.savefig(a.out.with_suffix('.png'),dpi=200);fig.savefig(a.out.with_suffix('.svg'))
    print(a.out.with_suffix('.png'))


if __name__=='__main__':main()
