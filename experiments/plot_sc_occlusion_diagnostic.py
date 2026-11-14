"""Scientific comparison of actual recorded camera pixels and extracted edges."""
import json
from pathlib import Path
import sys
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from diagnose_sc_rim_occlusion import white_only_edges, cable_aware_edges

ROOT=Path(__file__).resolve().parents[1]


def main():
    run='dev_3000_factor30v1_original_perception_rim_position_track_handoff'
    folder=ROOT/'.artifacts/lastinch_runs'/run/run/'rim_inputs/000'
    data=json.loads((folder/'inputs.json').read_text())
    fig,axes=plt.subplots(3,3,figsize=(9,8.7),layout='constrained')
    for row,v in enumerate(data['views']):
        image=cv2.imread(str(folder/(v['camera']+'.png')))
        P=np.array(v['P']);q=P@np.r_[data['initial_position'],1];uv=q[:2]/q[2]
        x,y=np.round(uv).astype(int);x0=max(0,x-90);y0=max(0,y-90)
        crop=np.s_[y0:y+91,x0:x+91]
        a,_=white_only_edges(image);b,mask=cable_aware_edges(image)
        axes[row,0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)[crop])
        axes[row,1].imshow(a[crop]<.1,cmap='gray',vmin=0,vmax=1)
        axes[row,2].imshow(b[crop]<.1,cmap='gray',vmin=0,vmax=1)
        axes[row,0].set_ylabel(v['camera'].replace('_',' '))
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],['Recorded camera crop','White-only mask: false cable edge','White + cable mask']):
        ax.set_title(title,fontsize=10)
    fig.suptitle('SC port rim: cable occlusion diagnostic (one recorded scene)\nSame Canny/blue filters and acceptance thresholds; no synthetic images',fontsize=11)
    out=ROOT/'experiments/evidence/original_repair'
    fig.savefig(out/'cable_occlusion_edges.png',dpi=180)
    fig.savefig(out/'cable_occlusion_edges.svg')


if __name__=='__main__':main()
