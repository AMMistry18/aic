"""Summarize measured short development work separately from frozen final44."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/parallel_dev46'


def main():
    freeze=json.loads((ROOT/'experiments/full_evaluation44/source_freeze.json').read_text())['source_sha256']
    mismatches=[p for p,h in freeze.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if mismatches:raise RuntimeError('Frozen source changed: '+str(mismatches))
    angular=json.loads((OUT/'angular_reprojection_results.json').read_text())['rows']
    body=json.loads((OUT/'body_pose_results.json').read_text())['rows']
    result=dict(scope='Development only, not additional insertion results',
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),final_sources_unchanged=True,
        frozen_source_count=len(freeze),final_cases_used_for_development=0,final_policy_modified=False,
        angular={},body_six_dof=dict(attempted=len(body),accepted=sum(r.get('fit',{}).get('accepted',False) for r in body)))
    for kind,tolerance in [('sc',.25),('sfp',.1)]:
        rows=[r for r in angular if r['connector']==kind]
        before=[r['original_errors']['angle_deg'] for r in rows];after=[r['joint_errors']['angle_deg'] for r in rows]
        result['angular'][kind]=dict(cases=len(rows),median_original_angle_error_deg=statistics.median(before),
            median_joint_angle_error_deg=statistics.median(after),improved=sum(b<a for a,b in zip(before,after)),
            joint_within_terminal_angle_tolerance=sum(v<=tolerance for v in after),terminal_tolerance_deg=tolerance,
            production_accepted=0,static_capture_sync_rejections_preserved=True)
    path=OUT/'cropped_device_benchmark.json'
    if path.exists():
        benchmark=json.loads(path.read_text());result['cropped_pipeline']={}
        for kind in ['sc','sfp']:
            rows=[r for r in benchmark['rows'] if r['connector']==kind]
            if not rows:continue
            differences=[r['comparison'] for r in rows if 'position_difference_m' in r['comparison']]
            result['cropped_pipeline'][kind]=dict(cases=len(rows),
                median_cpu_wall_s=statistics.median(r['measurements']['cpu']['median_wall_s'] for r in rows),
                median_gpu_wall_s=statistics.median(r['measurements']['0']['median_wall_s'] for r in rows),
                all_acceptance_agrees=all(r['comparison']['acceptance_agrees'] for r in rows),
                all_rejection_agrees=all(r['comparison']['rejection_agrees'] for r in rows),
                maximum_position_difference_um=max((r['position_difference_m']*1e6 for r in differences),default=None),
                maximum_angle_difference_deg=max((r['rotation_difference_deg'] for r in differences),default=None))
        result['cropped_pipeline_scope']='Plug detector crops plus original weighted fusion/gates; excludes port/body/rim fitting, ROS waiting and motion; CPU quota 1, concurrent final study'
    callback=OUT/'thread_callback_verification.json'
    if callback.exists():result['thread_candidate']=json.loads(callback.read_text())
    integration=OUT/'thread_candidate_integration.json'
    if integration.exists():result['thread_candidate_integration']=json.loads(integration.read_text())
    progress=json.loads((OUT/'progress.json').read_text())
    result['physical_short_probe']=dict(completed=0,environment_or_cancelled_attempts=len(progress['results']),
        deferred=True,reason='Concurrent full-scene renderers threaten final raw sensor coverage; offline diagnostics continue')
    final=json.loads((ROOT/'experiments/full_evaluation44/progress.json').read_text())
    result['final_accounting_checkpoint']=dict(status=final['status'],attempts=len(final['results']),summary=final['summary'],
        does_not_select_development_candidates=True,
        resource_contention_note='Separate development Gazebo renderers ran concurrently during part of final44; wall timings are shared-laptop timings. SC700045 had a 0.3605 s raw force gap and remains failed. Causation has not been established; concurrent development rendering stopped.')
    (OUT/'measured_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='final_accounting_checkpoint'},indent=2))


if __name__=='__main__':main()
