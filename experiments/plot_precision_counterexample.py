"""Plot measured isolated narrowphase results; no simulation success inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(5.5,2.9),layout='constrained')
    for name,label,color,marker in [('float32','Float32 Bullet 3.24','#b54b27','o'),('float64','Float64 Bullet 3.24','#126b89','s')]:
        d=json.loads((a.diagnostic/(name+'.json')).read_text());rows=d['scaling_diagnostic']
        ax.plot([r['uniform_scale'] for r in rows],[r['normalized_signed_distance_m']*1000 for r in rows],
            color=color,marker=marker,label=label,lw=1.2,ms=4)
    ax.axhline(0,color='black',lw=.8);ax.axvline(1,color='black',lw=.5,ls=':')
    ax.text(1.15,1,'Gap reported',fontsize=8);ax.text(1.15,-1,'Overlap reported',fontsize=8)
    ax.set(xscale='log',xticks=[1,10,100,1000],xticklabels=['1 (meters)','10','100','1000'],
        xlabel='Uniform query scale (collision margins scaled proportionally)',
        ylabel='Signed distance / scale (mm)',ylim=(-2.4,2.4),
        title='Recorded capsule / NIC-box pose: isolated collision query')
    ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2)
    fig.savefig(a.out/'precision_counterexample.png',dpi=220);fig.savefig(a.out/'precision_counterexample.pdf');plt.close(fig)


if __name__=='__main__':main()
