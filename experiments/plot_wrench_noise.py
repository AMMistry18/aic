"""Plot recorded sensor samples and retrospective estimator, never simulated results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.audit.read_text());raw=d['recorded_trace'];est=d['retrospective_estimator_trace'];first=raw[0]['sim_s']
    fig,ax=plt.subplots(figsize=(6.8,2.8),layout='constrained')
    ax.plot([r['sim_s']-first for r in raw],[r['tared_tip_axis_moment_Nm'] for r in raw],lw=.7,color='#888888',label='Recorded tared noisy samples')
    ax.plot([r['sim_s']-first for r in est],[r['upper_abs_bound_Nm'] for r in est],lw=1.5,color='#126b89',label='Retrospective 99.9% absolute bound')
    ax.axhline(.08,color='#b54b27',ls='--',lw=.8,label='Unchanged 0.08 Nm value');ax.axhline(-.08,color='#b54b27',ls='--',lw=.8)
    ax.axvspan(0,raw[99]['sim_s']-first,color='#228833',alpha=.08,label='100-sample tare')
    ax.set(xlabel='Recorded stationary-command window (simulation s)',ylabel='Tip-axis moment (Nm)',title='Source-noise diagnosis; retrospective estimator only')
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out.with_suffix('.png'),dpi=220);fig.savefig(a.out.with_suffix('.pdf'));plt.close(fig)

if __name__=='__main__':main()
