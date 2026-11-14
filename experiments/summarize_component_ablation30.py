"""Read-only partial/final summary of the predeclared component comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='component30v1');a=p.parse_args()
    protocol=json.loads((ROOT/'experiments/configs/component_ablation30.json').read_text());rows=[];maps=[]
    for case in protocol['case_ids']:
        for method in protocol['methods']:
            name=f'{case}_{a.label}_{method}';archive=ROOT/'.artifacts/lastinch_runs'/name;r=archive/name
            if not (r/'verified.json').exists():continue
            policy=json.loads((r/'policy.json').read_text());v=json.loads((r/'verified.json').read_text());w=json.loads((r/'wrench_analysis.json').read_text())
            if not policy['angular_screen_diagnostic_ablation'] or policy['counts_as_guarded_production_policy']:raise ValueError('Ablation labeling contract violated')
            log=(ROOT/'.artifacts/container'/(name+'_policy.log')).read_text();timing=re.findall(r'SC_TIMING_SUMMARY outcome=(\S+) reason=(\S+)',log)
            cause=v['reason']
            if not v['verified_success'] and timing:cause+=' / '+timing[-1][1]
            rows.append(dict(case_id=case,run=name,method=method,verified_success=v['verified_success'],failure_cause=None if v['verified_success'] else cause,
                terminal_metrics=v['terminal_metrics'],geometry_verifier_reason=v.get('geometry_verifier',v)['reason'],
                terminal_violation_counts=v.get('terminal_violation_counts'),peak_independent_force_N=w['peak_independent_tared_force_N'],
                p95_independent_force_N=w['p95_independent_tared_force_N'],
                peak_policy_force_N=policy['peak_tared_force_N'],max_global_penetration_m=v['max_global_penetration_m'],wall_s=policy['wall_s'],sim_s=policy['sim_s'],
                body_refinement_enabled=policy['body_translation_refinement_enabled'],tracking_enabled=policy['lateral_tracking_compensation_enabled'],
                counts_as_guarded_production_policy=False))
            maps.append({str(f.relative_to(archive/'source_snapshot')).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in (archive/'source_snapshot').rglob('*') if f.is_file()})
    summary={}
    for method in protocol['methods']:
        completed=[r for r in rows if r['method']==method];success=[r for r in completed if r['verified_success']]
        summary[method]=dict(planned=5,completed=len(completed),verified=len(success),
            independent_force_peak_range_N=[min(r['peak_independent_force_N'] for r in completed),max(r['peak_independent_force_N'] for r in completed)] if completed else None,
            median_completion_sim_s=float(np.median([r['sim_s'] for r in success])) if success else None,
            median_attempt_sim_s=float(np.median([r['sim_s'] for r in completed])) if completed else None,
            median_attempt_wall_s=float(np.median([r['wall_s'] for r in completed])) if completed else None,
            failures=[dict(case_id=r['case_id'],cause=r['failure_cause']) for r in completed if not r['verified_success']])
    output=dict(protocol=protocol,planned=15,completed=len(rows),complete=len(rows)==15,
        identical_source_checkpoint_maps=bool(maps) and all(m==maps[0] for m in maps),
        final_100000_series_used=False,confidence_interval=None,
        inference_limit='Fixed reused development cases, conditional upstream alignment; no generalized reliability CI',
        method_summary=summary,runs=rows)
    (ROOT/'experiments/evidence/original_repair'/f'component_screen_{a.label}.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['runs','protocol']},indent=2))


if __name__=='__main__':main()
