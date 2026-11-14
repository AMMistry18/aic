"""Freeze an auditable manuscript snapshot and evidence-first writing contract."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
SOURCE=ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json'


def write_csv(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def wilson(k,n):
    z=1.959963984540054;p=k/n
    midpoint=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [midpoint-half,midpoint+half]


def main():
    results=PAPER/'results';results.mkdir(exist_ok=True)
    snap=results/'recovery_snapshot_20261112.json'
    if not snap.exists():
        raw=SOURCE.read_bytes();live=json.loads(raw)
        snap.write_text(json.dumps(dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            source=str(SOURCE.relative_to(ROOT)),ledger_sha256=hashlib.sha256(raw).hexdigest(),
            study_status=live['status'],current_case=live.get('current_case'),
            source_sha256=live['source_sha256'],results=live['results'],
            aborted_setups=live.get('aborted_setups',[]),
            interpretation='Frozen partial development snapshot; no held-out or population reliability claim'),indent=2))
    data=json.loads(snap.read_text());summary={}
    table=[]
    for kind in ['sc','sfp']:
        rows=[r for r in data['results'] if r.get('connector',r['case']['connector'])==kind]
        eligible=[r for r in rows if r['status']=='completed'];ok=[r for r in rows if r.get('verified_success')]
        cancelled=[a for a in data.get('aborted_setups',[]) if any(r['run'].startswith(a['case_id']+'_') for r in rows)]
        summary[kind]=dict(verified=len(ok),attempts=len(rows)+len(cancelled),policy_trials=len(eligible),
            policy_failures=len(eligible)-len(ok),setup_failures=len(rows)-len(eligible),
            cancelled_duplicate_setups=len(cancelled),
            median_success_sim_s=statistics.median(r['sim_s'] for r in ok),
            median_success_wall_s=statistics.median(r['wall_s'] for r in ok),
            max_raw_peak_N=max(r['high_rate_wrist_audit']['peak_tared_force_N'] for r in eligible),
            complete_2000Hz=all(r['high_rate_wrist_audit']['complete_action_coverage'] for r in eligible),
            descriptive_policy_trial_wilson95=wilson(len(ok),len(eligible)),
            interval_limitation='Development case selection and shared-seed pairs prevent a held-out population interpretation')
        for r in rows:
            a=r.get('high_rate_wrist_audit',{});m=r.get('terminal_metrics',{})
            table.append(dict(run=r['run'],connector=kind,case=r['run'].split('_recovery')[0],
                simulator_seed=r['case']['seed'],target=r['case']['target_index'],slot=r['case']['sfp_port'],
                status=r['status'],verified=bool(r.get('verified_success')),reason=r.get('reason',r.get('error','')),
                sim_s=r.get('sim_s',''),wall_s=r.get('wall_s',''),raw_peak_N=a.get('peak_tared_force_N',''),
                complete_2000Hz=a.get('complete_action_coverage',''),
                depth_min_mm=1000*m['depth_m']['min'] if m else '',
                depth_max_mm=1000*m['depth_m']['max'] if m else '',
                lateral_max_mm=1000*m['lateral_error_m']['max'] if m else '',
                angle_max_deg=m['angular_error_deg']['max'] if m else '',evidence=r['evidence_dir']))
        for event in cancelled:
            row={key:'' for key in table[0]}
            row.update(run=event['case_id']+'_'+event['label']+'_cancelled_setup',connector=kind,
                case=event['case_id'],status='cancelled_duplicate_setup',verified=False,
                reason=event['reason'],evidence=event['evidence_dir'])
            table.append(row)
    (results/'recovery_summary_20261112.json').write_text(json.dumps(summary,indent=2))
    write_csv(results/'recovery_trials_20261112.csv',list(table[0]),table)
    macros=['% Generated from the immutable partial development snapshot.']
    for kind,d in summary.items():
        for name,value in [('Verified',d['verified']),('Attempts',d['attempts']),('Eligible',d['policy_trials']),
             ('MedianSim',f"{d['median_success_sim_s']:.2f}"),('MedianWall',f"{d['median_success_wall_s']:.1f}"),
             ('PeakForce',f"{d['max_raw_peak_N']:.2f}"),
             ('Interval',f"{100*d['descriptive_policy_trial_wilson95'][0]:.1f}--{100*d['descriptive_policy_trial_wilson95'][1]:.1f}")]:
            macros.append('\\newcommand{\\'+kind.upper()+name+'}{'+str(value)+'}')
    (results/'recovery_macros.tex').write_text('\n'.join(macros)+'\n')
    claims=[
        ('C1','Conditional SC and SFP held-grasp insertion executes with nonprivileged observations','verified','EXP1','No unrestricted rotation or population robustness claim'),
        ('C2','Connector-specific relative datums address different axial observability','verified','EXP1;EXP2','Mechanism implemented; current causal advantage remains pending'),
        ('C3','Full-rate independent geometry/contact/force verification distinguishes policy completion from a seat','verified','EXP1','Simulation fidelity and sampled contact coverage limit interpretation'),
        ('C4','Current combined method generalizes over a fresh randomized population','planned','EXP4','No final cases opened'),
        ('C5','Greater force authority reduces complete cycle time','planned','EXP3','No speed43 physical results at snapshot'),
    ]
    write_csv(PAPER/'brief/evidence-matrix.csv',['claim_id','claim','result_status','experiment_id','boundary'],
              [dict(zip(['claim_id','claim','result_status','experiment_id','boundary'],row)) for row in claims])
    write_csv(PAPER/'notes/design/method-components.csv',
        ['component','inputs','outputs','contribution_status','ablation'],[
        dict(component='Weighted keypoints',inputs='Three RGB views and calibrated exposure transforms',outputs='Plug and port proposals',contribution_status='Reused original method',ablation='Single-view or equal-weight study not yet done'),
        dict(component='CAD refinement',inputs='RGB edges and public geometry',outputs='Validated lateral/full translation candidate',contribution_status='Engineering integration',ablation='Earlier SC no-body screen verified'),
        dict(component='Relative datum registration',inputs='Own estimated tip motion and tared wrist force',outputs='SC shoulder or SFP lip/sidewall correction',contribution_status='Candidate practical contribution',ablation='Discard correction while retaining probes; pending'),
        dict(component='Tracking compensation',inputs='Own TCP tracking error and force',outputs='Bounded PI/feedforward',contribution_status='Established control principle',ablation='Earlier no-PI screen verified; current removal pending'),
        dict(component='Independent evaluator',inputs='Evaluator-only poses contacts and 2000Hz force',outputs='Seat verdict and recorded failure',contribution_status='Auditable simulation evaluation',ablation='Never a policy input'),
    ])
    experiments=[
        ('EXP1','recovery replay','verified','Partial frozen snapshot only','results/recovery_snapshot_20261112.json'),
        ('EXP2','earlier SC component screen','verified','Different revision; 5/5 full versus 0/5 without body or PI','../experiments/evidence/original_repair/component_screen_component30v1.json'),
        ('EXP3','speed43 matched development','planned','Six paired cases after replay','../experiments/recovery_20261112/speed43_development/progress.json'),
        ('EXP4','fresh combined robustness','planned','50 cases per connector; no tuning on final cases','../experiments/configs/recovered_robustness43_protocol.json'),
        ('EXP5','current matched component removal','planned','Additional 80 paired trials require qualification','../experiments/configs/recovered_robustness43_protocol.json'),
    ]
    fields=['experiment_id','type','result_status','scope','source']
    write_csv(PAPER/'notes/design/experiment-matrix.csv',fields,[dict(zip(fields,r)) for r in experiments])
    fields=['method','role','selected','reason','citation']
    write_csv(PAPER/'notes/design/baselines.csv',fields,[dict(zip(fields,r)) for r in [
        ('Initial learned geometry','ablation anchor','yes','Existing matched SC no-body evidence; no full combined comparison','team'),
        ('Without tracking compensation','ablation anchor','yes','Isolate own-robot tracking effect','team'),
        ('Discard axial correction; retain probes','ablation anchor','planned','Isolate information benefit from probe motion','kim2022'),
        ('Insert-One','related work','no','Broader 6DoF real-world task; no comparable implementation executed','chang2024'),
        ('FORGE','related work','no','Force-aware learned sim-to-real; no matched benchmark executed','noseworthy2024'),
        ('SyPAS','related work','no','Different connector types and real-world servoing; numerical ranking inappropriate','sypas'),
    ]])
    (PAPER/'plan').mkdir(exist_ok=True)
    (PAPER/'plan/20261112-writing-contract.md').write_text('''# Four-page empirical writing contract

Authorization: the user explicitly requested writing the paper and provided the empirical-paper-writer skill. Existing experiment and novelty plans are the approved starting point; no repeat approval gate is needed.

Scope: conditional held-grasp last-inch SC and SFP insertion in Gazebo. Use a four-page body, with at most one additional reference page if needed. Preserve prior manuscript drafts. Write a new main.tex and ref.bib, compile, render and inspect.

Contribution: a practical allocation of pose components to visual geometry and connector-specific physical datums, supported by independent recorded verification. Do not claim invention of visual-force fusion, confidence weighting, synthetic training or active probing. Current combined-method causal improvement remains pending.

Outline: application/related work; visual/contact method; simulator and scoring; measured development outcomes/failure analysis; limitations and conclusion. One actual-camera figure, one structural pipeline diagram, scoring/results tables, and a compact earlier-ablation comparison. New results use the immutable snapshot, not changing live progress.

Skill adaptations: the user specifies four pages, overriding the skill's 6-10-page and 25-citation defaults. Use a compact set of directly relevant verified primary references. Existing runtime and authorization permit ongoing experiment execution; do not stop active jobs under the skill's generic user-run handoff rule. Design changes and unfinished experiments stay explicitly planned. No manuscript submission is authorized.
''',encoding='utf-8')
    (PAPER/'paper.config.yaml').write_text('''venue: IEEE Robotics and Automation Practice
page_target_body: 4
page_limit_references: 1
runtime:
  python: C:/Python314/python.exe
  simulator_container: research-aic-eval
  perception_python: /opt/research-perception/bin/python
evidence_snapshot: results/recovery_snapshot_20261112.json
study_status: partial_development
final_trials_started: false
''')
    fields=['ID','Phase','Title','Section_Path','Claim_ID','Evidence_Type','Experiment_ID','Result_Status','Description','Source_Policy','Target_Citations','Visualization','Acceptance','Status','Verified_Citations','Depends_On','Must_Verify','Notes']
    rows=[]
    def issue(id,phase,title,claim='C1',evidence='mixed',exp='EXP1',status='TODO',result='verified',dep='R1',section='N/A',source='N/A',target=0):
        rows.append(dict(zip(fields,[id,phase,title,section,claim,evidence,exp,result,title,source,target,'Actual camera figure and explicit result tables','Evidence cited and bounded claims checked',status,0,dep,'yes','Partial development draft; preserve pending results'])))
    issue('R1','Research','Contract and immutable evidence inventory',status='DONE',dep='N/A')
    issue('R2','Research','Verify primary references and closest prior work',status='DOING',evidence='citation')
    issue('E1','Experiment','Freeze completed development snapshot',status='DONE')
    issue('E2','Experiment','Current combined ablations and held-out evaluation',claim='C4',exp='EXP4;EXP5',result='planned')
    for n,title in enumerate(['Application and scope','Visual and contact registration','Testbed and independent verification','Measured development results','Discussion and conclusion'],1):
        issue('W'+str(n),'Writing',title,section=title,source='standard',target=1,dep='R1;R2;E1')
    issue('V1','Visual','Actual mesh camera sequence and pipeline diagram',section='Application and scope',evidence='figure')
    issue('Q1','QA','Citation compile page count and visual layout audit',dep='W1;W2;W3;W4;W5;V1')
    path=PAPER/'issues/20261112-sc-sfp-draft.csv'
    if not path.exists():write_csv(path,fields,rows)
    print(json.dumps(dict(snapshot=str(snap),summary=summary),indent=2))


if __name__=='__main__':main()
