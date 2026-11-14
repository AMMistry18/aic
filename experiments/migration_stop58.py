"""Stop only this research queue at a fully recorded trial boundary."""
import json
import subprocess
import time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/migration58';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    crop_lease=ROOT/'experiments/runtime_qualification47_v57/slot_lease.json'
    lease=read(crop_lease)
    if lease['state']!='waiting_for_completed_trial':raise RuntimeError('Crop slot active; inspect before stopping')
    # Cancel waiting offline workers before acquiring the same boundary.
    for pid,needle in [(51740,'development_crop_slot57.py'),(55328,'continue_crop57.py'),(67816,'--watchdog')]:
        try:p=psutil.Process(pid)
        except psutil.NoSuchProcess:continue
        if p.name().lower()!='python.exe' or not any(needle in s for s in p.cmdline()):raise RuntimeError('Worker identity changed')
        p.terminate();p.wait(10)
    lease.update(state='aborted_before_slot',reason='user_requested_machine_migration',stopped_unix=time.time())
    crop_lease.write_text(json.dumps(lease,indent=2))
    for path in [ROOT/'experiments/crop_followup57/progress.json',ROOT/'experiments/runtime_qualification47_v57/progress.json']:
        state=read(path);state.update(status='cancelled_before_execution_for_machine_migration');path.write_text(json.dumps(state,indent=2))
    p=psutil.Process(45828)
    if abs(p.create_time()-1794581613.7174058)>.01 or not any('resume_full_evaluation44.py' in s for s in p.cmdline()):raise RuntimeError('Coordinator identity changed')
    report=dict(status='waiting_for_completed_trial',started_unix=time.time())
    (OUT/'stop_progress.json').write_text(json.dumps(report,indent=2))
    while True:
        state=read(ROOT/'experiments/full_evaluation44/progress.json')
        if state.get('current_run') is None:
            p.suspend();time.sleep(2)
            state=read(ROOT/'experiments/full_evaluation44/progress.json')
            running=subprocess.run(['docker','inspect','research-aic-eval','--format','{{.State.Running}}'],capture_output=True,text=True,check=True,timeout=20).stdout.strip()
            if state.get('current_run') is None and running=='false':
                p.terminate();p.wait(15)
                (OUT/'progress_before_migration.json').write_text(json.dumps(state,indent=2))
                state.update(status='stopped_requires_inspection',error='User requested machine migration at completed trial boundary; no selected case interrupted or replaced')
                (ROOT/'experiments/full_evaluation44/progress.json').write_text(json.dumps(state,indent=2))
                report.update(status='stopped_at_completed_trial_boundary',completed=len(state['results']),ended_unix=time.time(),current_run=None)
                (OUT/'stop_progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
            p.resume()
        if not p.is_running():raise RuntimeError('Coordinator exited; inspect ledger')
        time.sleep(.08)
if __name__=='__main__':main()
