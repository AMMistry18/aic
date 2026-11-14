"""Analyze completed frozen development cases, never infer pending results."""
import csv
import json
import math
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair34/validation34v1'


def wilson(s,n):
    if not n:return None
    z=1.959963984540054;den=1+z*z/n;center=(s/n+z*z/(2*n))/den
    half=z*math.sqrt(s/n*(1-s/n)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def failure_category(result,policy,datum):
    if result.get('verified_success'):return None
    if result.get('status')!='completed':return 'environment_or_execution_failure'
    audit=result.get('high_rate_wrist_audit',{})
    if not audit.get('complete_action_coverage',False):return 'incomplete_raw_force_coverage'
    if (audit.get('peak_tared_force_N') or 0)>8:return 'raw_force_exceeded_8N'
    terminal=policy.get('observable_terminal_diagnostic') or {}
    if result.get('geometry_verifier',{}).get('verified_success') and terminal.get('reason')=='terminal_timeout':
        return 'physical_seat_but_policy_confirmation_timeout'
    feedback=' '.join(str(x.get('message','')) for x in policy.get('feedback',[])).lower()
    if 'timed out' in feedback or 'action_deadline' in str(policy.get('reason','')):
        return 'action_deadline_before_completion'
    reason=str(datum.get('reason',''))
    if 'waypoint tracking timeout' in reason:return 'initial_waypoint_tracking_timeout'
    if 'lateral contact unobservable' in reason:
        records=datum.get('lateral_registration') or []
        if records and records[-1].get('force_difference_N',float('inf'))<=records[-1].get('force_difference_3sem_N',0):
            return 'side_wall_force_difference_below_noise_margin'
        return 'unobservable_lateral_contact'
    return policy.get('sfp_fixture_failure') or terminal.get('reason') or reason or policy.get('reason') or result.get('reason')


def main():
    state=json.loads((OUT/'progress.json').read_text());rows=[]
    for result in state['results']:
        run=result['run'];folder=ROOT/result['evidence_dir']/run
        policy=json.loads((folder/'policy.json').read_text()) if (folder/'policy.json').exists() else {}
        audit=result.get('high_rate_wrist_audit',{})
        datum=policy.get('sfp_contact_datum') or policy.get('axial_contact_registration') or {}
        rows.append(dict(run=run,connector=result['case']['connector'],case_id=result['case']['case_id'],
            status=result['status'],verified_success=bool(result.get('verified_success',False)),
            reason=result.get('reason',result.get('error')),policy_reason=policy.get('reason'),
            component_reason=datum.get('reason'),failure_category=failure_category(result,policy,datum),
            geometry_only_pass=bool(result.get('geometry_verifier',{}).get('verified_success',result.get('verified_success',False))),
            wall_s=result.get('wall_s',policy.get('wall_s')),
            sim_s=result.get('sim_s',policy.get('sim_s')),raw_peak_N=audit.get('peak_tared_force_N'),
            raw_p95_N=audit.get('p95_tared_force_N'),raw_force_qualified=audit.get('qualified',False),
            raw_coverage_complete=audit.get('complete_action_coverage',False)))
    summary=dict(status=state['status'],planned=len(state.get('planned_cases',[])),completed=len(rows),
        held_out_final_trials_run=0,paper_ready=False,
        interval_warning='Exploratory Wilson intervals conditional on these development cases; selected SC regressions are not a random population sample.',
        timing_warning='Local headless simulation wall times, not real-robot throughput; all-case and successful-only times kept separate.',by_connector={})
    for kind in ['sc','sfp']:
        cases=[r for r in rows if r['connector']==kind];success=[r for r in cases if r['verified_success']]
        failure={}
        for r in cases:
            if not r['verified_success']:
                key=r['failure_category'];failure[key]=failure.get(key,0)+1
        def med(key,items):
            values=[r[key] for r in items if r[key] is not None];return median(values) if values else None
        peaks=[r['raw_peak_N'] for r in cases if r['raw_peak_N'] is not None]
        summary['by_connector'][kind]=dict(completed=len(cases),verified=len(success),
            observed_success_fraction=len(success)/len(cases) if cases else None,
            descriptive_wilson95=wilson(len(success),len(cases)),
            all_attempts_median_wall_s=med('wall_s',cases),all_attempts_median_sim_s=med('sim_s',cases),
            successful_only_median_wall_s=med('wall_s',success),successful_only_median_sim_s=med('sim_s',success),
            maximum_raw_peak_N=max(peaks) if peaks else None,
            incomplete_raw_force_records=sum(not r['raw_coverage_complete'] for r in cases),failure_causes=failure)
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    if rows:
        with (OUT/'trials.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
