"""Matched target outcomes, measured force and explicit development scope."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'experiments/evidence/original_repair';data=json.loads((out/'component_screen_component30v1.json').read_text())
    methods=data['protocol']['methods'];methods=[methods[0],methods[2],methods[1]]
    cases=data['protocol']['case_ids'];values=np.full((5,3),2);labels={}
    for r in data['runs']:
        i=cases.index(r['case_id']);j=methods.index(r['method']);values[i,j]=int(r['verified_success']);labels[i,j]=('Pass' if r['verified_success'] else 'Fail')+'\n'+f"{r['peak_independent_force_N']:.2f} N"
    with plt.rc_context({'font.size':8,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(3.5,3.0));ax.imshow(values,cmap=ListedColormap(['#f3c7c7','#c9e5d1','#ededed']),vmin=0,vmax=2,aspect='auto')
        for i in range(5):
            for j in range(3):ax.text(j,i,labels.get((i,j),'Pending'),ha='center',va='center',fontsize=8)
        ax.set_xticks(range(3),['Body + PI','No body','No PI']);ax.xaxis.tick_top();ax.tick_params(length=0,pad=5)
        ax.set_yticks(range(5),[f'SC {i} ({3000+2*i})' for i in range(5)])
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True);ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
        for s in ax.spines.values():s.set_visible(False)
        fig.subplots_adjust(left=.27,right=.98,top=.79,bottom=.16)
        fig.suptitle('Conditional SC insertion: matched cases',fontsize=9,y=.98)
        fig.text(.5,.885,'Upstream angular alignment assumed',ha='center',fontsize=7.5)
        fig.text(.5,.07,'Numbers: independent peak tared wrist force.\nFive reused development cases; no reliability CI.',ha='center',fontsize=7)
        fig.savefig(out/'component30_outcomes.png',dpi=240);fig.savefig(out/'component30_outcomes.svg');plt.close(fig)


if __name__=='__main__':main()
