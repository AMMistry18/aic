"""Export executed evidence only; aggregate seeded development separately."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-half),min(1,center+half)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--indexes',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'experiments/evidence/insertion_development');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);records=[];groups=defaultdict(list);seen=set()
    for index_path in a.indexes:
        index=json.loads(index_path.read_text())
        cohort=index.get('development_revision','initial_development')
        for result in index['results']:
            run=result['run'];source=ROOT/result['evidence_dir'];dest=a.out/run;dest.mkdir(exist_ok=True)
            inner=source/run
            for name in ('scene.json','controller_bootstrap.json','physics_process_audit.json'):
                if (source/name).exists():shutil.copy2(source/name,dest/name)
            if (source/'source_snapshot').exists():shutil.copytree(source/'source_snapshot',dest/'source_snapshot',dirs_exist_ok=True)
            if (source/'cameras').exists():shutil.copytree(source/'cameras',dest/'setup_cameras',dirs_exist_ok=True)
            if (source/'sfp_contact_positive_control.json').exists():shutil.copy2(source/'sfp_contact_positive_control.json',dest/'sfp_contact_positive_control.json')
            for name in ('world.sdf','controller_hold_calibrated.json'):
                if (source/name).exists():
                    with gzip.open(dest/(name+'.gz'),'wb') as f:f.write((source/name).read_bytes())
            if inner.exists():
                for item in inner.iterdir():
                    if item.suffix in ('.json','.gz'):
                        if item.name=='policy.json':
                            with gzip.open(dest/'policy.json.gz','wb') as f:f.write(item.read_bytes())
                        else:shutil.copy2(item,dest/item.name)
                for camera in ('overview','target','board','left_camera','center_camera','right_camera'):
                    images=sorted(inner.glob(camera+'_*.jpg'),key=lambda f:float(f.stem.rsplit('_',1)[1]))
                    if not images:continue
                    for label,i in (('before',0),('during',len(images)//2),('after',len(images)-1)):
                        shutil.copy2(images[i],dest/f'{camera}_{label}.jpg')
                for item in inner.glob('vision_*.png'):shutil.copy2(item,dest/item.name)
            if result.get('verification_file'):
                if (dest/'verified.json').exists():shutil.copy2(dest/'verified.json',dest/'legacy_verified.json')
                shutil.copy2(ROOT/result['verification_file'],dest/'verified.json')
                shutil.copy2(ROOT/result['verification_trace_file'],dest/'verification_trace.json.gz')
            if index.get('positive_control_proof'):
                if (dest/'sfp_contact_positive_control.json').exists():
                    shutil.copy2(dest/'sfp_contact_positive_control.json',dest/'legacy_sfp_contact_positive_control.json')
                shutil.copy2(ROOT/index['positive_control_proof'],dest/'sfp_contact_positive_control.json')
            for label,path in result.get('diagnostic_files',{}).items():
                if Path(label).name!=label:raise ValueError('Diagnostic label must be a filename')
                (dest/'diagnostics').mkdir(exist_ok=True)
                shutil.copy2(ROOT/path,dest/'diagnostics'/label)
            (dest/'trial_index.json').write_text(json.dumps({'trial':result,'batch':{k:v for k,v in index.items() if k!='results'}},indent=2)+'\n')
            force=result.get('peak_independent_tared_force_N')
            wrench=json.loads((inner/'wrench_analysis.json').read_text()) if (inner/'wrench_analysis.json').exists() else {}
            if force is None and (inner/'wrench_analysis.json').exists():
                force=json.loads((inner/'wrench_analysis.json').read_text())['peak_independent_tared_force_N']
            row={'run':run,'connector':result['case']['connector'],'method':result['method'],'development_revision':cohort,'seed':result['case']['seed'],
                'status':result['status'],'verified_success':result.get('verified_success',False),'reason':result.get('reason',result.get('error')),
                'eligible_for_policy_comparison':result.get('eligible_for_policy_comparison'),
                'all_recorded_checks_passed':result.get('all_recorded_checks_passed'),
                'independent_source_overlap_max_mm':result.get('independent_source_overlap_m')*1000 if result.get('independent_source_overlap_m') is not None else None,
                'verification_semantics_revision':result.get('verification_semantics_revision'),
                'wall_s':result.get('wall_s'),'sim_s':result.get('sim_s'),
                'peak_tared_force_N':force if force is not None else result.get('peak_policy_tared_force_N'),
                'p95_tared_force_N':wrench.get('p95_independent_tared_force_N')}
            terminal=result.get('terminal_metrics',{})
            for key,field,scale in [('terminal_depth_min_mm','depth_m',1000),('terminal_depth_max_mm','depth_m',1000),
                ('terminal_lateral_max_mm','lateral_error_m',1000),('terminal_angle_max_deg','angular_error_deg',1),
                ('terminal_penetration_max_mm','max_penetration_m',1000)]:
                value=terminal.get(field,{}).get('min' if key=='terminal_depth_min_mm' else 'max')
                row[key]=value*scale if value is not None else None
            row['whole_trial_penetration_max_mm']=result.get('max_global_penetration_m',0)*1000 if terminal else None
            row['recorded_whole_trial_penetration_within_0p1mm']=bool(result.get('max_global_penetration_m',float('inf'))<=.0001) if terminal else None
            if (inner/'truth_trace.json.gz').exists() and (inner/'policy.json').exists():
                trace=json.load(gzip.open(inner/'truth_trace.json.gz','rt'))
                policy=json.loads((inner/'policy.json').read_text());start=policy.get('trace',[{}])[0].get('sim_s') if policy.get('trace') else None
                first=min(trace,key=lambda r:abs(r['sim_time_s']-start)) if trace and start is not None else None
            else:first=None
            row.update(initial_measured_gap_mm=-first['depth_m']*1000 if first else None,
                initial_measured_lateral_mm=first['lateral_error_m']*1000 if first else None,
                initial_measured_full_angle_deg=first['angular_error_deg'] if first else None,
                initial_truth_command_stamp_difference_s=first['sim_time_s']-start if first else None)
            records.append(row)
            if result['case']['case_id'].startswith('dev_'):
                key=(row['connector'],row['method'],cohort,row['seed'])
                if row['status']=='completed' and row['eligible_for_policy_comparison'] is not False:
                    if key in seen:raise ValueError(f'Repeated development seed would inflate independent sample size: {key}')
                    seen.add(key)
                groups[(row['connector'],row['method'],cohort)].append(row)
    summaries=[]
    for (connector,method,cohort),rows in sorted(groups.items()):
        completed=[r for r in rows if r['status']=='completed']
        valid=[r for r in rows if r['status']=='completed' and r['eligible_for_policy_comparison'] is not False];success=[r for r in valid if r['verified_success']]
        full_available=bool(valid) and all(r['all_recorded_checks_passed'] is not None for r in valid)
        full_successes=sum(r['all_recorded_checks_passed'] is True for r in valid) if full_available else None
        numeric=lambda key:[r[key] for r in valid if isinstance(r.get(key),(int,float))]
        summaries.append({'connector':connector,'method':method,'development_revision':cohort,'attempted':len(rows),'valid_trials':len(valid),
            'completed_executions_before_initial_contact_screen':len(completed),
            'unfiltered_geometric_successes':sum(r['verified_success'] for r in completed),
            'unfiltered_geometric_success_wilson_95_ci':wilson(sum(r['verified_success'] for r in completed),len(completed)),
            'environment_failures':sum(r['status']!='completed' for r in rows),
            'initial_contact_integrity_exclusions':sum(r['status']=='completed' and r['eligible_for_policy_comparison'] is False for r in rows),
            'successes':len(success),
            'success_rate':len(success)/len(valid) if valid else None,'wilson_95_ci':wilson(len(success),len(valid)),
            'full_recorded_successes':full_successes,
            'full_recorded_success_wilson_95_ci':wilson(full_successes,len(valid)) if full_available else None,
            'geometric_successes_passing_recorded_whole_trial_penetration_screen':sum(r['recorded_whole_trial_penetration_within_0p1mm'] is True for r in success),
            'recorded_whole_trial_penetration_screen_success_wilson_95_ci':wilson(sum(r['recorded_whole_trial_penetration_within_0p1mm'] is True for r in success),len(valid)),
            'median_wall_s':statistics.median(numeric('wall_s')) if numeric('wall_s') else None,
            'median_sim_s':statistics.median(numeric('sim_s')) if numeric('sim_s') else None,
            'median_success_sim_s':statistics.median([r['sim_s'] for r in success]) if success else None,
            'median_success_wall_s':statistics.median([r['wall_s'] for r in success]) if success else None,
            'median_peak_force_N':statistics.median(numeric('peak_tared_force_N')) if numeric('peak_tared_force_N') else None,
            'max_peak_force_N':max(numeric('peak_tared_force_N'),default=None),
            'failure_causes':dict(Counter(r['reason'] for r in valid if not r['verified_success'])),
            'environment_failure_causes':dict(Counter(r['reason'] for r in rows if r['status']!='completed'))})
    (a.out/'results.json').write_text(json.dumps({'trials':records,'seeded_development_summary':summaries,
        'interval_method':'Wilson 95%, independent seeded cases per method; small development sample, not final reliability',
        'heldout_final_trials_executed':0},indent=2)+'\n')
    with (a.out/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['run']);writer.writeheader();writer.writerows(records)
    lines=['# Executed insertion evidence','','Only executed trials are exported. Nominal diagnostics and seeded development are separate.',
        'All trials use a prescribed grasp; no pickup, grasp stability, release retention or electrical mating is established.','',
        '| Connector | Method / development revision | Success / valid trials | 95% Wilson CI | Median sim / wall s | Max peak force N |',
        '|---|---|---:|---|---|---:|']
    for s in summaries:
        ci=s['wilson_95_ci'];interval='n/a' if ci is None else f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]%"
        timing='n/a' if s['median_sim_s'] is None else f"{s['median_sim_s']:.2f} / {s['median_wall_s']:.2f}"
        force='n/a' if s['max_peak_force_N'] is None else f"{s['max_peak_force_N']:.2f}"
        lines.append(f"| {s['connector']} | {s['method']} / {s['development_revision']} | {s['successes']} / {s['valid_trials']} | {interval} | {timing} | {force} |")
    lines+=['','Contact interpretation: the pinned Bullet plugin reports signed distance; negative values are penetration. Revision 3 additionally recognizes audited nonzero normal solver impulses as active contact, including numerical positive gaps. Positive-gap cached manifolds with zero impulse remain insufficient. Per-trial revisions and any original verdicts are retained. All numeric seating gates remain unchanged.',
        'The raw traces retain transient/global and requested-connector penetration separately. Terminal seating gates do not erase earlier contact fidelity limitations.',
        'Times are policy elapsed time to success or failure, excluding world preparation and hold calibration. Successful completion times are separately recorded in results.json. Failed-trial time is not a successful cycle time.',
        'The source-world hash and policy/profile source hashes are preserved with each trial. No held-out final case was used for development.','']
    lines+=['Development revisions are separate cohorts. Repeated seeds across revisions are debugging comparisons, not additional independent samples or pooled reliability evidence.','']
    lines+=['Policy-comparison denominators exclude positively observed invalid initial contacts when an explicit post-hoc screen is supplied. All executions, excluded forces and original verdicts remain in trials.csv and trial_index.json. The short legacy pre-command screen does not certify the new full setup hold gate.','']
    lines+=['Terminal geometric seats and whole-trial contact fidelity are distinct. The additional recorded whole-trial penetration screen requires every recorded global contact distance to imply <=0.1 mm penetration. It is sampled evidence, not a continuous physical guarantee. Passing terminal seating does not override a failed whole-trial screen.','']
    (a.out/'README.md').write_text('\n'.join(lines))
    manifest={str(f.relative_to(a.out)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(a.out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.json'}
    (a.out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'exported_trials':len(records),'seeded_groups':summaries},indent=2))

if __name__=='__main__':main()
