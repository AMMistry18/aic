"""Plot recorded geometry and independently tared force, never synthetic curves."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    root=a.evidence;fig,axes=plt.subplots(2,2,figsize=(7.1,4.6),layout='constrained')
    runs=[('SC','sc_supported_ecm_free_initial_pose_guarded',15.64,.25),
          ('SFP','sfp_supported_ecm_dt1_initial_pose_guarded',45.8,.25)]
    for col,(label,run,seat,tol) in enumerate(runs):
        folder=root/run;truth=json.load(gzip.open(folder/'truth_trace.json.gz','rt'))
        policy=json.load(gzip.open(folder/'policy.json.gz','rt'));t0=policy['trace'][0]['sim_s']
        times=np.array([r['sim_time_s'] for r in truth])-t0
        ax=axes[0,col];ax.plot(times,[r['depth_m']*1000 for r in truth],lw=1.2,color='#126b89')
        ax.axhspan(seat-tol,seat+tol,color='#228833',alpha=.2,label='Seat tolerance')
        ax.set(title=f'{label}: verified nominal held seat',ylabel='Insertion depth (mm)')
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
        raw=json.load(gzip.open(folder/'wrench.json.gz','rt'));analysis=json.loads((folder/'wrench_analysis.json').read_text())
        wt=np.array([m['header']['stamp']['sec']+m['header']['stamp']['nanosec']/1e9 for m in raw])-t0
        values=np.array([[m['wrench']['force'][k] for k in ('x','y','z')] for m in raw])
        norm=np.linalg.norm(values-np.array(analysis['baseline_N']),axis=1);valid=wt>=0
        ax=axes[1,col];ax.plot(wt[valid],norm[valid],lw=.8,color='#b54b27');ax.axhline(8,color='black',ls='--',lw=.7,label='8 N abort threshold')
        ax.set(xlabel='Time from first command (simulation s)',ylabel='Tared wrist force (N)',ylim=(0,8.5))
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(root/'nominal_depth_force.png',dpi=220);fig.savefig(root/'nominal_depth_force.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.6),layout='constrained')
    for row,(label,run,_,_) in enumerate(runs):
        for col,phase in enumerate(('before','after')):
            camera='center_camera' if label=='SFP' else 'target'
            ax=axes[row,col];ax.imshow(plt.imread(root/run/f'{camera}_{phase}.jpg'));ax.axis('off')
            ax.set_title(f'{label}: {phase} ({"wrist" if label=="SFP" else "fixed"} camera)',fontsize=10)
    fig.savefig(root/'nominal_camera_frames.png',dpi=200);fig.savefig(root/'nominal_camera_frames.pdf');plt.close(fig)
    print(root/'nominal_depth_force.png');print(root/'nominal_camera_frames.png')

if __name__=='__main__':main()
