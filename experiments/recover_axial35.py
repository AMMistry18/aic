"""Preserve and complete offline verification after the host worker exited."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    status=ROOT/'.artifacts/repair35/axial35v1/progress.json'
    state=json.loads(status.read_text());run=state['current_run']
    assert state['status']=='executing_development_case' and len(state['results'])==1
    for path,h in state['source_and_checkpoint_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    source=(ROOT/'.artifacts/kilted_handoff').resolve()
    destination=(ROOT/'.artifacts/lastinch_runs'/run).resolve()
    assert source.is_relative_to(ROOT/'.artifacts') and destination.is_relative_to(ROOT/'.artifacts')
    assert not destination.exists()
    result=json.loads((source/run/'verified.json').read_text())
    assert result['high_rate_wrist_audit']['complete_action_coverage']
    assert not json.loads((source/run/'status.json').read_text())['active']
    manifest=json.loads((ROOT/'experiments/configs/sfp_contact34_development.json').read_text())
    case=next(c.copy() for c in manifest['cases'] if c['case_id']=='dev_45003')
    case['case_id']='dev_45003_axial35v1'
    source.rename(destination)
    result.update(status='completed',case=case,method='original_perception_sfp_contact_datum_handoff',
        evidence_dir=str(destination.relative_to(ROOT)),
        recovered_after_host_worker_exit=True,
        recovery_note='Policy and recorder finished, host worker absent, Docker server left running; transport stdout truncated. Source hashes unchanged; complete recorded force and original policy failure retained.',
        recovery_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    result['peak_independent_tared_force_N']=json.loads((destination/run/'wrench_analysis.json').read_text())['peak_independent_tared_force_N']
    state['results'].append(result)
    state.update(status='development_complete_recovered_needs_analysis',current_run=None,paper_ready=False)
    status.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (status.parent/'dev_45003_recovered_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'.artifacts/active_original_perception29.json';a=json.loads(p.read_text())
    a.update(status='axial35_development_complete_recovered_needs_analysis',current_run=None,
             live_insertion_world=False,axial35_completed=2,paper_ready=False,held_out_final_trials_run=0)
    p.write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(run=run,verified_success=result['verified_success'],recovered=True,
                         raw_peak_N=result['high_rate_wrist_audit']['peak_tared_force_N'])))


if __name__=='__main__':main()
