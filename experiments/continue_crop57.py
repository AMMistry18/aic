"""Run a bounded perception diagnostic in an owned slot after motion qualification."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/crop_followup57'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'progress.json').exists():raise FileExistsError('Preserve attempt')
    state=dict(status='waiting_for_motion_qualification',final_cases_used=0,policy_changed=False)
    def save():
        state['updated_unix']=time.time();p=OUT/'progress.tmp'
        p.write_text(json.dumps(state,indent=2));p.replace(OUT/'progress.json')
    save()
    try:
        deadline=time.monotonic()+7200
        while True:
            previous=read(ROOT/'experiments/registration_followup55/progress.json')
            if previous['status'] in ('completed_requires_review','stopped_requires_inspection'):
                lease=ROOT/'experiments/runtime_qualification47_v7/slot_lease.json'
                if not lease.exists() or read(lease)['state'] in ('released','aborted_before_slot'):break
            if time.monotonic()>deadline:raise TimeoutError('Prior queue incomplete')
            time.sleep(5)
        source=(ROOT/'experiments/development_slot47.py').read_text()
        source=source.replace("final['status']=='primary_final_trials'",
            "final['status'] in ('primary_final_trials','paired_final_ablations')")
        start=source.index('        for kind,case,manifest,method in [')
        end=source.index("        unchanged();report.update(status='complete_requires_review'",start)
        offline="""        subprocess.run(['docker','start',CONTAINER],check=True,capture_output=True,timeout=30)
        with (OUT/'crop_inference.log').open('wb') as log:
            subprocess.run(['docker','exec','-w','/research',CONTAINER,'python3',
                'experiments/audit_sfp_crop57.py'],stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=600)
        report['offline_result']='experiments/crop_development57/results.json'
"""
        source=source[:start]+offline+source[end:]
        scheduler=ROOT/'experiments/development_crop_slot57.py'
        if scheduler.exists():raise FileExistsError('Preserve scheduler source')
        scheduler.write_text(source)
        state['status']='waiting_for_serialized_perception_slot';save()
        with (OUT/'slot.log').open('wb') as log:
            subprocess.run([sys.executable,str(scheduler),'--revision','57'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        state['status']='complete_requires_review';save()
    except Exception as exc:
        state.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise
if __name__=='__main__':main()
