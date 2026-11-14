"""Actual frozen V25 development outcomes; no pooling with later repairs."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'.artifacts/repair34/validation34v1'
OUT=ROOT/'experiments/evidence/original_repair/validation34v1'


def main():
    summary=json.loads((DATA/'analysis.json').read_text())
    assert summary['completed']==21 and summary['status']=='development_complete_needs_analysis'
    OUT.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.85),gridspec_kw={'width_ratios':[1,1.3]})
    data=summary['by_connector'];names=['sc','sfp'];colors=['#2878ad','#d17b1e']
    for i,(name,color) in enumerate(zip(names,colors)):
        d=data[name];p=d['observed_success_fraction'];lo,hi=d['descriptive_wilson95']
        axes[0].bar(i,p*100,width=.53,color=color,zorder=2)
        axes[0].errorbar(i,p*100,yerr=np.array([[p-lo],[hi-p]])*100,fmt='none',color='black',capsize=4,zorder=3)
        axes[0].text(i,4,f"{d['verified']}/{d['completed']}",ha='center',va='bottom',color='white',weight='bold')
    axes[0].set(xticks=[0,1],xticklabels=['SC','SFP'],ylim=(0,105),ylabel='Verified complete trials (%)',title='(a) Development qualification')
    axes[0].yaxis.grid(True,color='.9',zorder=0);axes[0].set_axisbelow(True)
    labels=['SC confirmation timeout','SFP initial waypoint timeout','SFP action deadline','SFP noisy side-wall evidence']
    keys=['physical_seat_but_policy_confirmation_timeout','initial_waypoint_tracking_timeout',
          'action_deadline_before_completion','side_wall_force_difference_below_noise_margin']
    values=[data['sc']['failure_causes'].get(keys[0],0)]+[data['sfp']['failure_causes'].get(k,0) for k in keys[1:]]
    axes[1].barh(range(4),values,color=[colors[0]]+[colors[1]]*3,height=.6)
    axes[1].set(yticks=range(4),yticklabels=labels,xlim=(0,4.7),xticks=range(5),xlabel='Failed complete trials',title='(b) Preserved failure causes')
    axes[1].invert_yaxis()
    for i,n in enumerate(values):axes[1].text(n+.09,i,str(n),va='center')
    fig.text(.5,.015,'Frozen V25 development cases; 95% Wilson intervals. SC uses selected regressions, not a random population sample.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.065,1,1))
    for suffix in ['svg','png','pdf']:fig.savefig(OUT/f'outcomes.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    inputs={f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['progress.json','analysis.json','trials.csv']}
    (OUT/'provenance.json').write_text(json.dumps(dict(study='validation34v1',actual_trials_only=True,
        later_repair_trials_included=False,source_sha256=inputs),indent=2)+'\n')
    for file in ['analysis.json','trials.csv']:(OUT/file).write_bytes((DATA/file).read_bytes())
    print(str(OUT/'outcomes.png'))


if __name__=='__main__':main()
