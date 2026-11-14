"""Measured selected development comparison; different revisions, no pooled rate."""
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'experiments/evidence/original_repair/tracking36';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,1,figsize=(7,5.3),sharex=True,layout='constrained')
provenance=[]
for label,revision,color in [('sampling36v1','V30: deadline failure','#b24c35'),('tracking36v1','V31: verified seat','#137e78')]:
    run=f'dev_45002_{label}_original_perception_sfp_contact_datum_handoff'
    folder=ROOT/'.artifacts/lastinch_runs'/run/run
    policy=json.loads((folder/'policy.json').read_text())
    samples=policy['sfp_contact_datum']['insertion_samples'];start=samples[0]['sim_s']
    truth=json.load(gzip.open(folder/'truth_trace.json.gz','rt'))
    selected=[r for r in truth if r['sim_time_s']>=start]
    t=np.array([r['sim_time_s']-start for r in selected]);depth=np.array([r['depth_m']*1000 for r in selected])
    axes[0].plot(t,depth,label=revision,color=color)
    axes[1].plot([s['sim_s']-start for s in samples],
                 [(s['command_depth_m']-s['estimated_depth_m'])*1000 for s in samples],color=color,label=revision)
    provenance.append(dict(run=run,independent_depth_evaluation_only=True,
                           final_recorded_depth_mm=float(depth[-1]),travel_recording_s=float(t[-1])))
axes[0].axhspan(45.55,46.05,color='grey',alpha=.15,label='Independent seat tolerance')
axes[0].axhline(45.8,color='grey',linestyle='--',linewidth=.8)
axes[0].set_ylabel('Independent depth (mm)');axes[0].legend(fontsize=8,loc='lower right')
axes[1].set_ylabel('Own estimated command lead (mm)');axes[1].set_xlabel('Simulation seconds since insertion travel began')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Same selected SFP development case, separate policy revisions',fontsize=11)
for suffix in ['png','svg','pdf']:fig.savefig(out/f'tracking.{suffix}',dpi=180)
(out/'provenance.json').write_text(json.dumps({'scope':'Selected development repair comparison, not a randomized causal study','runs':provenance},indent=2)+'\n')
print(json.dumps(provenance,indent=2))
