"""Finite dev-only axial-search qualification after the isolated SC cache pair.

The prospective comparison is dev_45005 against runtime47v3. No final case,
policy, manifest or acceptance threshold is used for selecting this change.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];DEV=ROOT/'.artifacts/dev46_workspace'
OUT=ROOT/'experiments/lip_followup51'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'progress.json').exists():raise FileExistsError('Preserve first qualification')
    state=dict(status='waiting_for_SC_cache_slot',final_policy_modified=False,
               prospective_case='dev_45005',baseline='runtime47v3',final_cases_used_for_tuning=0)
    def save():
        state['updated_unix']=time.time();p=OUT/'progress.tmp'
        p.write_text(json.dumps(state,indent=2),encoding='utf-8');p.replace(OUT/'progress.json')
    freeze=read(ROOT/'experiments/full_evaluation44/source_freeze.json')['source_sha256']
    def unchanged():
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in freeze.items()):
            raise RuntimeError('Frozen final source changed')
    save()
    try:
        deadline=time.monotonic()+5400
        while True:
            status=read(ROOT/'experiments/rim_followup49_v2/progress.json')
            if status['status']=='stopped_requires_inspection':raise RuntimeError('Prior physical qualification needs review')
            if status['status']=='completed_development_pair_requires_review':
                if read(ROOT/'experiments/runtime_qualification47_v4/slot_lease.json')['state']=='released':break
            if time.monotonic()>deadline:raise TimeoutError('Prior SC slot incomplete; no change installed')
            time.sleep(5)
        test=read(ROOT/'experiments/lip_development51/synthetic_replay.json')
        source=ROOT/'experiments/candidates/lip51/sfp_contact_datum.py'
        contract=read(source.with_name('source_contract.json'))
        if not test['checks_passed'] or test['summary']['candidate']['accepted']!=32:
            raise RuntimeError('Axial command/negative controls must pass before physical use')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=contract['candidate_sha256']:
            raise RuntimeError('Candidate changed after its prospective specification')
        target=DEV/'aic_model/aic_model/insertion/sfp_contact_datum.py'
        if hashlib.sha256(target.read_bytes()).hexdigest()!=contract['parent_development_sha256']:
            raise RuntimeError('Unexpected development parent')
        unchanged();shutil.copy2(target,OUT/'baseline_sfp_contact_datum.py');shutil.copy2(source,target)
        state.update(status='physical_lip_search_qualification',candidate_sha256=contract['candidate_sha256']);save()
        with (OUT/'physical.log').open('wb') as log:
            subprocess.run([sys.executable,'experiments/development_slot47.py','--revision','5','--only','sfp'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3300)
        candidate=read(ROOT/'experiments/runtime_qualification47_v5/progress.json')['results'][0]
        baseline=read(ROOT/'experiments/runtime_qualification47_v3/progress.json')['results'][0]
        comparison=[]
        for name,row in [('measured_start_baseline',baseline),('faster_bounded_lip_search',candidate)]:
            folder=DEV/row['evidence_dir'].replace('\\','/')/row['run'];policy=read(folder/'policy.json')
            log=(DEV/'.artifacts/container'/(row['run']+'_policy.log')).read_text(errors='replace')
            phases={m[1]:float(m[2]) for m in re.finditer(r'DATUM_PHASE (\S+) sim_s=([\d.]+)',log)}
            comparison.append(dict(condition=name,run=row['run'],verified_success=bool(row.get('verified_success')),
                lip_phase_sim_s=phases['aligned_entry']-phases['entrance_lip_probe_+0.002m'],
                action_sim_s=policy['sim_s'],record_wall_s=policy['wall_s'],
                raw_peak_N=row['high_rate_wrist_audit']['peak_tared_force_N'],
                force_coverage=row['high_rate_wrist_audit']['complete_action_coverage']))
        unchanged();state.update(status='completed_development_pair_requires_review',comparison=comparison,
            final_sources_unchanged=True,not_a_randomized_success_rate=True);save()
        if not candidate.get('verified_success'):raise RuntimeError('Faster search did not verify physically')
    except Exception as exc:
        state.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
