"""Analyze complete frozen final observations; no model selection or tuning."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.artifacts/final_aligned31'


def wilson(success,total):
    if total==0:return None
    z=1.959963984540054;p=success/total;denom=1+z*z/total
    centre=(p+z*z/(2*total))/denom;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [0. if success==0 else max(0.,centre-half),1. if success==total else min(1.,centre+half)]


def distribution(values):
    values=[float(v) for v in values if v is not None]
    if not values:return None
    return dict(n=len(values),median=float(np.median(values)),q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),min=min(values),max=max(values))


def main():
    batch=json.loads((OUT/'progress.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());protocol=json.loads((ROOT/'experiments/configs/final_aligned31_protocol.json').read_text())
    if batch['status']!='complete' or len(batch['results'])!=150:raise ValueError('No final inference from incomplete trials')
    if len({r['run'] for r in batch['results']})!=150:raise ValueError('Duplicate final identity')
    for source,digest in batch['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:raise ValueError(f'Final source changed: {source}')
    methods=protocol['methods'];rows=[];summaries={}
    for r in batch['results']:
        row=dict(run=r['run'],case_id=r['case']['case_id'],target_index=r['case']['target_index'],method=r['method'],status=r['status'],verified_success=bool(r.get('verified_success',False)),
            failure_cause=None if r.get('verified_success') else r.get('reason',r.get('error')),
            peak_independent_force_N=r.get('peak_independent_tared_force_N'),wall_s=r.get('wall_s'),sim_s=r.get('sim_s'),
            max_global_penetration_m=r.get('max_global_penetration_m'),terminal_metrics=r.get('terminal_metrics'),
            nominal_gap_mm=r['case']['gap_mm'],nominal_lateral_norm_mm=float(np.linalg.norm(r['case']['lateral_mm'])),grasp_jitter_seed=r['case'].get('grasp_jitter_seed'))
        if r['status']=='completed':
            folder=ROOT/r['evidence_dir']/r['run'];trace=json.load(gzip.open(folder/'verification_trace.json.gz','rt'));w=json.loads((folder/'wrench_analysis.json').read_text());policy=json.loads((folder/'policy.json').read_text())
            configuration=policy['configuration_wall_s']
            row.update(first_recorded_actual_angle_deg=trace[0]['angular_error_deg'],first_recorded_actual_lateral_mm=trace[0]['lateral_error_m']*1000,p95_independent_force_N=w['p95_independent_tared_force_N'],configuration_wall_s=configuration,action_wall_s=row['wall_s']-configuration)
        rows.append(row)
    for m in methods:
        selected=[r for r in rows if r['method']==m];eligible=[r for r in selected if r['status']=='completed'];success=[r for r in eligible if r['verified_success']]
        if len(selected)!=50:raise ValueError('Every method must retain all fifty selected cases')
        summaries[m]=dict(selected=50,policy_eligible=len(eligible),environment_or_execution_failures=50-len(eligible),verified=len(success),
            all_selected_success_rate=len(success)/50,all_selected_wilson95=wilson(len(success),50),
            eligible_success_rate=len(success)/len(eligible) if eligible else None,eligible_wilson95=wilson(len(success),len(eligible)),
            successful_completion_sim_s=distribution([r['sim_s'] for r in success]),successful_completion_wall_s=distribution([r['wall_s'] for r in success]),
            successful_action_wall_s=distribution([r['action_wall_s'] for r in success]),configuration_wall_s=distribution([r['configuration_wall_s'] for r in eligible]),
            all_attempt_duration_sim_s=distribution([r['sim_s'] for r in eligible]),
            all_measured_peak_force_N=distribution([r['peak_independent_force_N'] for r in eligible]),
            all_measured_p95_force_N=distribution([r.get('p95_independent_force_N') for r in eligible]),
            first_recorded_actual_angle_deg=distribution([r.get('first_recorded_actual_angle_deg') for r in eligible]),
            successful_max_penetration_m=distribution([r['max_global_penetration_m'] for r in success]),
            failure_causes=dict(Counter(r['failure_cause'] for r in selected if not r['verified_success'])))
    by={(r['case_id'],r['method']):r for r in rows};common=[c['case_id'] for c in manifest['cases'] if all(by[c['case_id'],m]['status']=='completed' for m in methods)]
    paired=[]
    for control in methods[1:]:
        b=sum(by[c,methods[0]]['verified_success'] and not by[c,control]['verified_success'] for c in common)
        c=sum(not by[c,methods[0]]['verified_success'] and by[c,control]['verified_success'] for c in common);discordant=b+c
        exact=1. if discordant==0 else min(1.,2*sum(math.comb(discordant,k) for k in range(min(b,c)+1))/2**discordant)
        paired.append(dict(full_method=methods[0],control=control,common_eligible_cases=len(common),full_only_successes=b,control_only_successes=c,
            observed_paired_success_difference=(b-c)/len(common) if common else None,exact_two_sided_McNemar_p=exact,
            inference_note='Two prespecified comparisons; p-values unadjusted, not evidence for another task or unconstrained angular recovery'))
    summary=dict(purpose='Measured frozen FINAL conditional SC results',protocol=protocol,manifest_sha256=batch['manifest_sha256'],
        cases=50,trials=150,methods=summaries,paired_comparisons=paired,all_frozen_sources_unchanged=True,
        inference_scope='Only the declared random target/translation/grasp distribution under nominal upstream angular alignment; no unrestricted rotation, SFP or hardware claim',
        old_100000_series_used=False,failed_cases_replaced_or_tuned=False,rows=rows)
    summary['timing_definition']='wall_s includes harness configuration; action_wall_s subtracts recorded configuration_wall_s and matches the action sim_s window. None is an independently established hardware cycle time.'
    export=ROOT/'experiments/evidence/final_aligned31';export.mkdir(exist_ok=True)
    (export/'protocol.json').write_bytes((ROOT/'experiments/configs/final_aligned31_protocol.json').read_bytes())
    (export/'manifest.json').write_bytes((OUT/'manifest.json').read_bytes())
    (export/'source_sha256.json').write_text(json.dumps(batch['source_sha256'],indent=2)+'\n')
    (export/'execution_metadata.json').write_text(json.dumps({k:v for k,v in batch.items() if k not in ['results','source_sha256']},indent=2)+'\n')
    with zipfile.ZipFile(export/'frozen_source_without_model_binaries.zip','w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in batch['source_sha256']:
            if Path(source).suffix!='.pt':bundle.write(ROOT/source,source)
    (export/'README.md').write_text('Final conditional SC measurements. All selected attempts are retained in trials.csv/results.json. Raw per-trial cameras, policy/verifier/wrench/contact records, environment audits and model snapshots remain at .artifacts/lastinch_runs/<run>. Exact source/checkpoint hashes and executable non-model sources are included here; original model binaries are preserved locally and not duplicated in this export. The earlier 100000-series full-angle set is unused. This study does not establish arbitrary rotation recovery, SFP perception, release retention or hardware transfer.\n',encoding='utf-8')
    (export/'results.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'analysis.json').write_text(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2)+'\n')
    fields=['run','case_id','target_index','method','status','verified_success','failure_cause','nominal_gap_mm','nominal_lateral_norm_mm','grasp_jitter_seed','peak_independent_force_N','p95_independent_force_N','sim_s','wall_s','configuration_wall_s','action_wall_s','max_global_penetration_m','first_recorded_actual_angle_deg','first_recorded_actual_lateral_mm']
    with (export/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    labels=['Body + PI','No body','No PI'];rates=[summaries[m]['all_selected_success_rate'] for m in methods];ci=np.array([summaries[m]['all_selected_wilson95'] for m in methods])
    with plt.rc_context({'font.size':8,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(3.5,2.8),layout='constrained');ax.bar(labels,np.array(rates)*100,color=['#4c956c','#858585','#b6a188'])
        ax.errorbar(range(3),np.array(rates)*100,yerr=np.maximum(0,np.array([np.array(rates)-ci[:,0],ci[:,1]-rates]))*100,fmt='none',ecolor='black',capsize=3)
        for i,m in enumerate(methods):ax.text(i,rates[i]*100+3,f"{summaries[m]['verified']}/50",ha='center')
        ax.set_ylim(0,112);ax.set_ylabel('Verified success (%)');ax.set_title('Frozen conditional SC evaluation\nNominal upstream angular alignment',fontsize=9);ax.spines[['top','right']].set_visible(False)
        fig.savefig(export/'success_wilson95.png',dpi=240);fig.savefig(export/'success_wilson95.svg');plt.close(fig)
    text=['# Frozen conditional SC final measurements','',summary['inference_scope'],'',
        'All 150 selected attempts are retained. Intervals are Wilson 95% under the declared sampling distribution. Environment/execution failures are shown separately; all-selected rates also count them as failures.','',
        '| Method | Verified / selected | Wilson 95% | Policy eligible | Environment/execution failures |','|---|---:|---:|---:|---:|']
    for label,m in zip(labels,methods):
        s=summaries[m];interval=s['all_selected_wilson95'];text.append(f"| {label} | {s['verified']}/50 | {interval[0]*100:.1f}-{interval[1]*100:.1f}% | {s['policy_eligible']} | {s['environment_or_execution_failures']} |")
    text+=['','Measured force/time distributions, every failure cause, actual initial angles, paired differences and exact tests are in `../experiments/evidence/final_aligned31/results.json`.','',
        'The 8 N check is a pre-command abort threshold, not a physical force ceiling. No threshold or weight was tuned on these final cases. This report does not establish unrestricted visual rotation recovery, SFP perception, hardware transfer or algorithmic novelty.','',
        '![Measured success and Wilson intervals](../experiments/evidence/final_aligned31/success_wilson95.svg)']
    (ROOT/'paper/FINAL_ALIGNED31_RESULTS.md').write_text('\n'.join(text)+'\n')
    active=ROOT/'.artifacts/active_original_perception29.json';status=json.loads(active.read_text());status.update(status='frozen_aligned31_final_complete',live_insertion_world=False,current_run='150 frozen conditional SC trials completed',final_results='paper/FINAL_ALIGNED31_RESULTS.md',final_study=summary['methods'],old_100000_series_used=False,verified_unrestricted_visual_rotation_insertions=0);active.write_text(json.dumps(status,indent=2)+'\n')
    (OUT/'waiting.json').write_text(json.dumps(dict(status='final_complete',progress='progress.json',report='paper/FINAL_ALIGNED31_RESULTS.md'),indent=2)+'\n')
    (export/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(export)):hashlib.sha256(p.read_bytes()).hexdigest() for p in export.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'},indent=2)+'\n')
    print(json.dumps(dict(final_report='paper/FINAL_ALIGNED31_RESULTS.md',methods=summaries,paired=paired),indent=2))


if __name__=='__main__':main()
