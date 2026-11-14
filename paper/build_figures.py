"""Use actual recorded camera pixels and evaluator traces; no synthetic results."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/figures';OUT.mkdir(exist_ok=True)
BASE=ROOT/'experiments/recovery_20261112/verified_evidence/runs'
SC='dev_42000_recovery20261112v1_original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff'
SFP='dev_45005_recovery20261112v1_original_perception_sfp_contact_datum_handoff'
manifest={}
for name,run in [('sc',SC),('sfp',SFP)]:
    images=sorted((BASE/run/'sampled_cameras').glob('target_*.jpg'),key=lambda p:float(p.stem.split('_')[-1]))
    for phase,image in [('before',images[0]),('after',images[-1])]:
        target=OUT/(name+'_'+phase+'.jpg');shutil.copy2(image,target)
        manifest[target.name]=dict(source=str(image.relative_to(ROOT)),sha256=hashlib.sha256(image.read_bytes()).hexdigest(),pixel_transform='none')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
    'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(3.45,1.65),layout='constrained')
for ax,run,label,seat in zip(axes,[SC,SFP],['SC','SFP'],[15.64,45.8]):
    src=BASE/run/'verification_trace.json.gz'
    trace=json.load(gzip.open(src,'rt'));t0=trace[0]['sim_time_s']
    ax.plot([r['sim_time_s']-t0 for r in trace],[1000*r['depth_m'] for r in trace],color='#176B87',lw=1.)
    ax.axhline(seat,color='#A94527',ls='--',lw=.8)
    ax.set_title(label);ax.set_xlabel('Recorded sim time (s)');ax.set_ylabel('Depth (mm)')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    manifest[label+'_trace']=dict(source=str(src.relative_to(ROOT)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),rows=len(trace),purpose='Evaluator only; not a policy input')
fig.savefig(OUT/'depth_traces.pdf');fig.savefig(OUT/'depth_traces.png',dpi=240);plt.close(fig)
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2))
print('Copied four unmodified camera frames; plotted two measured depth traces.')
