"""Plot exported measured development outcomes, including invalid setup runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    data=json.loads((a.evidence/'results.json').read_text())
    rows=[r for r in data['trials'] if r['method']=='initial_pose_guarded' and r['seed']>=3000 and r['status']=='completed']
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    colors={'sc':'#126b89','sfp':'#b54b27'}
    for r in rows:
        eligible=r['eligible_for_policy_comparison'] is not False;color=colors[r['connector']] if eligible else '#888888'
        tolerance=.25 if r['connector']=='sc' else .1
        marker='o' if r['verified_success'] else 'x'
        x=r['terminal_lateral_max_mm']/tolerance;y=r['terminal_angle_max_deg']/tolerance
        axes[0].scatter(x,y,c=color,marker=marker,s=35)
        axes[0].annotate(str(r['seed']), (x,y),xytext=(3,3),textcoords='offset points',fontsize=6)
    ax=axes[0];ax.axvline(1,color='black',ls='--',lw=.7);ax.axhline(1,color='black',ls='--',lw=.7)
    ax.set(xlabel='Terminal lateral error / tolerance',ylabel='Terminal full angle / tolerance',title='Measured terminal pose')
    ax.margins(x=.12)
    ax.legend(handles=[Line2D([],[],color=colors['sc'],marker='o',ls='',label='SC'),
        Line2D([],[],color=colors['sfp'],marker='x',ls='',label='SFP'),
        Line2D([],[],color='#888888',marker='x',ls='',label='Invalid initial overlap')],fontsize=6)
    summaries=[s for s in data['seeded_development_summary'] if s['method']=='initial_pose_guarded'
               and s['development_revision']=='initial_development' and s['valid_trials']>0]
    for i,s in enumerate(summaries):
        y=s['success_rate'];lo,hi=s['wilson_95_ci']
        axes[1].errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=4,color=colors[s['connector']])
        axes[1].text(i,.98,f"{s['successes']}/{s['valid_trials']}",ha='center',va='top',fontsize=9)
    axes[1].set(xticks=range(len(summaries)),xticklabels=[s['connector'].upper() for s in summaries],ylim=(-.04,1.04),
        ylabel='Verified / eligible trials (95% Wilson)',title='Small development sample')
    axes[1].set_xlim(-.5,len(summaries)-.5)
    for i,r in enumerate(rows):
        eligible=r['eligible_for_policy_comparison'] is not False
        axes[2].scatter(i,r['peak_tared_force_N'],c=colors[r['connector']] if eligible else '#888888',s=30,marker='o' if eligible else 'x')
    axes[2].axhline(8,color='black',ls='--',lw=.7,label='8 N abort threshold')
    axes[2].set(xticks=range(len(rows)),xticklabels=[str(r['seed']) for r in rows],ylabel='Independent tared wrist peak (N)',title='All recorded baseline executions')
    axes[2].tick_params(axis='x',rotation=70,labelsize=6);axes[2].legend(fontsize=6)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(a.evidence/'seeded_baseline.png',dpi=220);fig.savefig(a.evidence/'seeded_baseline.pdf');plt.close(fig)

if __name__=='__main__':main()
