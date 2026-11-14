"""Finite, serialized physical qualification after exact-pruning qualification."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'.artifacts/dev46_workspace'
OUT=ROOT/'experiments/registration_followup55'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'progress.json').exists():raise FileExistsError('Preserve previous attempt')
    state=dict(status='waiting_for_pruning_physical_qualification',final_cases_used_for_tuning=0,
        prospective_case='dev_45005',comparison='v6 vs v7; lateral travel limits only',
        force_and_geometry_thresholds_changed=False)
    def save():
        state['updated_unix']=time.time();p=OUT/'progress.tmp'
        p.write_text(json.dumps(state,indent=2));p.replace(OUT/'progress.json')
    freeze=read(ROOT/'experiments/full_evaluation44/source_freeze.json')['source_sha256']
    def unchanged():
        if any(sha(ROOT/p)!=h for p,h in freeze.items()):raise RuntimeError('Frozen source changed')
    save()
    try:
        deadline=time.monotonic()+3600
        while True:
            previous=read(ROOT/'experiments/pruning_followup54/progress.json')
            if previous['status']=='stopped_requires_inspection':raise RuntimeError('Pruning qualification stopped')
            if previous['status']=='completed_requires_review':
                if not previous['result']['verified_success']:raise RuntimeError('Pruning did not verify')
                if read(ROOT/'experiments/runtime_qualification47_v6/slot_lease.json')['state']=='released':break
            if time.monotonic()>deadline:raise TimeoutError('Previous qualification incomplete')
            time.sleep(5)
        proof=read(ROOT/'experiments/registration_development55/synthetic_replay.json')
        candidate=ROOT/'experiments/candidates/registration55/contact_registration.py'
        target=DEV/'aic_model/aic_model/insertion/contact_registration.py'
        if not proof['checks_passed'] or sha(candidate)!=proof['candidate_sha256'] or sha(target)!=proof['parent_sha256']:
            raise RuntimeError('Candidate replay or parent identity mismatch')
        unchanged();shutil.copy2(target,OUT/'baseline_contact_registration.py');shutil.copy2(candidate,target)
        # Reuse the established lease/watchdog. Both primary and ablation
        # trial boundaries have the same stopped-container ownership contract.
        slot=(ROOT/'experiments/development_slot47.py').read_text()
        anchor="final['status']=='primary_final_trials'"
        if slot.count(anchor)!=1:raise RuntimeError('Unexpected scheduler')
        slot=slot.replace(anchor,"final['status'] in ('primary_final_trials','paired_final_ablations')")
        scheduler=ROOT/'experiments/development_slot55.py'
        if scheduler.exists():raise FileExistsError('Preserve scheduler revision')
        scheduler.write_text(slot)
        state.update(status='waiting_for_serialized_physical_test',candidate_sha256=sha(candidate));save()
        with (OUT/'physical.log').open('wb') as log:
            subprocess.run([sys.executable,str(scheduler),'--revision','7','--only','sfp'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3300)
        result=read(ROOT/'experiments/runtime_qualification47_v7/progress.json')['results'][0]
        unchanged();state.update(status='completed_requires_review',result=result,final_sources_unchanged=True);save()
    except Exception as exc:
        state.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise
if __name__=='__main__':main()
