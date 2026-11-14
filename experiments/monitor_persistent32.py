"""Read-only experiment observation and measured progress; no policy mutation."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
ACTIVE=ROOT/'.artifacts/active_original_perception29.json'
FINAL=ROOT/'.artifacts/final_aligned32'


def read(path):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return None


def main():
    previous_count=-1
    while True:
        state=read(ACTIVE) or {};waiting=read(FINAL/'waiting.json') or {};final=read(FINAL/'progress.json')
        archives=list((ROOT/'.artifacts/lastinch_runs').glob('*_persistent32v1_*'))
        complete=[p for p in archives if (p/p.name/'verified.json').exists()]
        if len(complete)!=previous_count:
            subprocess.run([sys.executable,str(ROOT/'experiments/summarize_persistent32.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            previous_count=len(complete)
        summary=read(ROOT/'experiments/evidence/original_repair/persistent_screen32v1.json') or {}
        state['persistent32v1']={k:v for k,v in summary.items() if k not in ['runs']}
        state['latest_attempt_video']='experiments/evidence/original_repair/video/persistent32_target0/attempt.mp4'
        state['queued_final_study']=dict(planned_cases=50,planned_trials=200,status=waiting.get('status'),status_file='.artifacts/final_aligned32/waiting.json',worker_lock_present=(FINAL/'runner.lock').exists())
        state['condition']='Upstream angular alignment assumed; original learned plug weights; RGB grasp translation; no object/target truth in policy'
        if final:
            state.update(status='frozen_aligned32_final_'+final['status'],
                current_run=f"Frozen final: {len(final['results'])}/200 completed trials",
                held_out_final_trials_run=len(final['results']),
                final_progress='.artifacts/final_aligned32/progress.json')
            if final['results']:
                last=final['results'][-1];state.update(last_run=last['run'],last_result=last.get('reason',last.get('error')))
        else:
            state.update(status='persistent32v1_development_running',held_out_final_trials_run=0,
                current_run=f"Qualification {summary.get('verified',0)}/{summary.get('completed',0)} completed verified; 11 planned, including 2 known repairs and 9 new development cases")
            if summary.get('runs'):
                last=summary['runs'][-1];state.update(last_run=last['run'],last_result=dict(reason=last['reason'],raw_peak_N=last['raw_peak_N'],complete_force_coverage=last['complete_force_coverage']))
        failed=waiting.get('status')=='final_not_started_qualification_failed'
        done=waiting.get('status')=='final_complete'
        if failed:state.update(status='persistent32v1_qualification_failed_final_not_started',live_insertion_world=False,final_gate_failures=waiting.get('failures'))
        elif done:state.update(live_insertion_world=False)
        else:state['live_insertion_world']=True
        state['progress_observed_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        temporary=ACTIVE.with_suffix('.monitor.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');temporary.replace(ACTIVE)
        if failed or done:return
        if not (FINAL/'runner.lock').exists():
            state.update(status='persistent32_observer_stopped_worker_not_running',live_insertion_world=False)
            ACTIVE.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');return
        time.sleep(30)


if __name__=='__main__':main()
