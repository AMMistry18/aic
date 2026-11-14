"""Read-only trial observation; only the user-facing status file is updated."""
import datetime
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]


def read(path):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return None


def main():
    while True:
        phase=read(ROOT/'.artifacts/repair33/progress.json') or {}
        batch=read(ROOT/'.artifacts/seeded_evaluation_progress.json') or {}
        results=[r for r in batch.get('results',[]) if '_axial33v1_' in r.get('run','')]
        active=ROOT/'.artifacts/active_original_perception29.json';state=read(active) or {}
        state.update(status='repair33_'+phase.get('status','unknown'),paper_scope=['sc','sfp'],
                     held_out_final_trials_run=0,repair33_progress='.artifacts/repair33/progress.json')
        state['sc_axial33v1']=dict(completed=len(results),verified=sum(r.get('verified_success',False) for r in results),planned=3,
                                   scope='known development regressions, not robustness')
        state['sfp_visual33']=dict(static_training_scenes=20,static_development_scenes=6,
            adapted_models_complete=all((ROOT/'.artifacts/sfp_visual33/training'/n/'training_result.json').exists() for n in ['nic33_v1','plug33_v1']),
            physical_robustness_qualified=False)
        state['current_run']=phase.get('status')
        if results:
            last=results[-1];state['last_run']=last['run'];state['last_result']=dict(verified_success=last.get('verified_success'),reason=last.get('reason',last.get('error')))
            folder=ROOT/last['evidence_dir']/last['run'];v=read(folder/'verified.json') or {}
            state['last_result']['raw_peak_N']=v.get('high_rate_wrist_audit',{}).get('peak_tared_force_N')
            state['current_run']=f"SC axial repair {state['sc_axial33v1']['verified']}/{len(results)} verified, 3 planned; SFP precision remains unqualified"
            state['latest_attempt_video']='experiments/evidence/original_repair/video/axial33_timeout_repair/attempt.mp4'
        state['progress_observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp=active.with_suffix('.repair33.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');tmp.replace(active)
        if phase.get('status') in ('development_complete_needs_inspection','queue_failed'):return
        time.sleep(15)


if __name__=='__main__':main()
