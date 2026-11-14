"""Measured missed-contact trace; a diagnostic figure, not an outcome revision."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    run='dev_45003_axial35v1_original_perception_sfp_contact_datum_handoff'
    source=ROOT/'.artifacts/lastinch_runs'/run/run/'policy.json'
    policy=json.loads(source.read_text());rows=policy['sfp_contact_datum']['axial_probe_samples']
    diagnostic=json.loads((ROOT/'.artifacts/repair36/lip_diagnosis.json').read_text())
    t=np.array([r['sim_s'] for r in rows]);t-=t[0]
    depth=np.array([r['measured_depth_m'] for r in rows])*1000
    command=np.array([r['applied_command_m'] for r in rows])*1000
    force=-np.array([r['force_port_N'][2] for r in rows])
    candidates=diagnostic['kinematically_blocked_but_reaction_rejected']
    start=candidates[0]['end_sim_s']-.83-rows[0]['sim_s'];end=candidates[-1]['end_sim_s']-rows[0]['sim_s']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(8,5.4),sharex=True,layout='constrained')
    axes[0].plot(t,command,label='Applied command',color='#d97706',lw=1.5)
    axes[0].plot(t,depth,label='Own-robot estimated tip',color='#007c91',lw=1.5)
    axes[0].set_ylabel('Estimated axial depth (mm)');axes[0].legend(loc='lower right')
    axes[0].set_title('A real lip contact missed by the single-press detector')
    axes[1].plot(t,force,color='#475569',lw=.65,label='Policy-observed compression')
    axes[1].set_ylabel('Compression (N)');axes[1].set_xlabel('Time since first probe sample (simulation s)')
    for ax in axes:
        ax.axvspan(start,end,alpha=.18,color='#c026d3')
        ax.grid(alpha=.15);ax.set_xlim(0,t[-1])
    axes[1].text(.37,.94,'Seven windows passed motion checks,\nbut reaction change was -0.208 to -0.110 N.\nRequired increase: +0.025 N.',
        transform=axes[1].transAxes,va='top',bbox=dict(facecolor='white',edgecolor='#cbd5e1',alpha=.95))
    fig.suptitle('SFP dev_45003, V28: failed insertion retained; no simulator pose inputs',fontsize=10)
    out=ROOT/'experiments/evidence/original_repair/lip36';out.mkdir(exist_ok=True)
    for extension in ['png','svg','pdf']:fig.savefig(out/('missed_contact.'+extension),dpi=180)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(run=run,policy_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sensor_scope='Policy-observed port-Z wrist compression, not the independent 2000 Hz vector-norm audit',
        smoothing='none',shading='union of windows that pass kinematic gates but fail reaction',
        simulator_truth_inputs=False),indent=2)+'\n',encoding='utf-8')
    print(out/'missed_contact.png')


if __name__=='__main__':main()
