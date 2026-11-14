"""Finite development continuation: image pruning check, then physical motion pair.

Requires the current runtime SC/SFP qualification to pass. Uses old development
cases only, serializes complete Gazebo scenes, and never edits final sources.
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
OUT=ROOT/'experiments/efficiency_followup48';CONTAINER='research-aic-perception30'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'progress.json').exists():raise FileExistsError('Preserve first continuation')
    state=dict(status='waiting_for_full_runtime_qualification',final_policy_modified=False)
    def save():
        state['updated_unix']=time.time();temp=OUT/'progress.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(OUT/'progress.json')
    def call(args,name,timeout=300):
        with (OUT/(name+'.log')).open('wb') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    freeze=read(ROOT/'experiments/full_evaluation44/source_freeze.json')['source_sha256']
    def unchanged():
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in freeze.items()):raise RuntimeError('Frozen final source changed')
    save()
    try:
        deadline=time.monotonic()+3600
        while True:
            path=ROOT/'experiments/runtime_qualification47_v2/progress.json'
            status=read(path)
            if status['status']=='stopped_requires_inspection':raise RuntimeError('Runtime qualification stopped; inspect before changing motion')
            if status['status']=='complete_requires_review':
                lease=read(path.with_name('slot_lease.json'))
                if lease['state']=='released':break
            if time.monotonic()>deadline:raise TimeoutError('Runtime qualification incomplete; no motion change applied')
            time.sleep(5)
        if len(status['results'])!=2 or not all(r.get('verified_success') for r in status['results']):raise RuntimeError('Both runtime cases must verify')
        unchanged();state['status']='offline_NIC_association_cache';save()
        call(['docker','update','--cpus','1','--memory','3g','--memory-swap','4g',CONTAINER],'offline_limit')
        call(['docker','start',CONTAINER],'offline_start')
        try:
            call(['docker','exec','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',CONTAINER,'nice','-n','19',
                  '/opt/research-perception/bin/python','/research/experiments/cache_fixture_detections48.py'],'cache',600)
        finally:call(['docker','stop','--timeout','5',CONTAINER],'offline_stop',60)
        if len(read(ROOT/'experiments/perception_efficiency48/detections.json')['rows'])!=4:raise RuntimeError('Incomplete cache')
        state['status']='original_fixture_fit_parity_benchmark';save()
        call([sys.executable,'experiments/benchmark_fixture_pruning48.py'],'fixture_benchmark',600)
        benchmark=read(ROOT/'experiments/perception_efficiency48/fixture_benchmark.json')
        if not benchmark['all_accepted_candidates_identical']:raise RuntimeError('Pruner changed original accepted candidates')
        # The admissible prefilter remains uninstalled; this physical pair
        # changes only the separately tested measured-start registration path.
        unchanged();target=DEV/'aic_model/aic_model/insertion/contact_registration.py'
        if target.read_text()!= (ROOT/'aic_model/aic_model/insertion/contact_registration.py').read_text():raise RuntimeError('Unexpected registration parent')
        shutil.copy2(target,OUT/'baseline_contact_registration.py')
        source=ROOT/'experiments/candidates/registration47/contact_registration.py';shutil.copy2(source,target)
        state.update(status='physical_measured_start_registration',candidate_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     candidate_installed_only_in_development=True);save()
        call([sys.executable,'experiments/development_slot47.py','--revision','3','--only','sfp'],'physical_registration',3300)
        candidate_state=read(ROOT/'experiments/runtime_qualification47_v3/progress.json')
        candidate=candidate_state['results'][0]
        if not candidate.get('verified_success'):raise RuntimeError('Measured-start insertion did not verify')
        baseline=next(r for r in status['results'] if r['case']['connector']=='sfp');comparison=[]
        for name,row in [('baseline',baseline),('measured_start',candidate)]:
            folder=DEV/row['evidence_dir'].replace('\\','/')/row['run'];policy=read(folder/'policy.json')
            log=(DEV/'.artifacts/container'/(row['run']+'_policy.log')).read_text(errors='replace');phases={}
            for match in re.finditer(r'DATUM_PHASE (\S+) sim_s=([\d.]+)',log):phases[match[1]]=float(match[2])
            stage=phases['full_depth_insertion']-phases['opposed_side_wall_registration']
            comparison.append(dict(condition=name,run=row['run'],verified_success=True,
                registration_sim_s=stage,action_sim_s=policy['sim_s'],record_wall_s=policy['wall_s'],
                raw_peak_N=row['high_rate_wrist_audit']['peak_tared_force_N'],
                force_coverage=row['high_rate_wrist_audit']['complete_action_coverage']))
            images=list(folder.glob('target_*.jpg'))
            if images:
                frame=max(images,key=lambda p:float(p.stem.split('_')[-1]));shutil.copy2(frame,OUT/(name+'_target_final.jpg'))
        unchanged();state.update(status='completed_development_pair_requires_review',comparison=comparison,
            final_sources_unchanged=True,not_a_randomized_success_rate=True);save()
    except Exception as exc:
        state.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
