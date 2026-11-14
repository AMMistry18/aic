"""Qualify immutable-CAD/image reuse physically after the SFP motion pair."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];DEV=ROOT/'.artifacts/dev46_workspace';OUT=ROOT/'experiments/rim_followup49'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'progress.json').exists():raise FileExistsError('Preserve first candidate qualification')
    state=dict(status='waiting_for_SFP_motion_pair',final_policy_modified=False)
    def save():
        state['updated_unix']=time.time();temp=OUT/'progress.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(OUT/'progress.json')
    save()
    try:
        deadline=time.monotonic()+9000
        while True:
            status=read(ROOT/'experiments/efficiency_followup48/progress.json')
            if status['status']=='stopped_requires_inspection':raise RuntimeError('Earlier qualification requires diagnosis; no SC change applied')
            if status['status']=='completed_development_pair_requires_review':
                lease=read(ROOT/'experiments/runtime_qualification47_v3/slot_lease.json')
                if lease['state']=='released':break
            if time.monotonic()>deadline:raise TimeoutError('Earlier physical pair incomplete')
            time.sleep(10)
        benchmark=read(ROOT/'experiments/perception_efficiency49/benchmark.json')
        if len(benchmark['rows'])!=12 or not benchmark['all_gate_decisions_match'] or any(not r['candidate_vs_original']['bit_identical'] for r in benchmark['rows']):
            raise RuntimeError('Require all 12 full numerical outputs identical before installation')
        freeze=read(ROOT/'experiments/full_evaluation44/source_freeze.json')['source_sha256']
        def unchanged():
            if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in freeze.items()):raise RuntimeError('Frozen final source changed')
        unchanged();target=DEV/'aic_model/aic_model/insertion/sc_rim_refinement.py'
        if target.read_text()!= (ROOT/'aic_model/aic_model/insertion/sc_rim_refinement.py').read_text():raise RuntimeError('Unexpected SC rim parent')
        shutil.copy2(target,OUT/'baseline_sc_rim_refinement.py')
        source=ROOT/'experiments/candidates/efficiency49/sc_rim_refinement.py';shutil.copy2(source,target)
        state.update(status='physical_SC_reuse_qualification',candidate_sha256=hashlib.sha256(source.read_bytes()).hexdigest());save()
        with (OUT/'physical.log').open('wb') as log:
            subprocess.run([sys.executable,'experiments/development_slot47.py','--revision','4','--only','sc'],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3300)
        candidate=read(ROOT/'experiments/runtime_qualification47_v4/progress.json')['results'][0]
        if not candidate.get('verified_success'):raise RuntimeError('Cached SC insertion did not independently verify')
        baseline=next(r for r in read(ROOT/'experiments/runtime_qualification47_v2/progress.json')['results'] if r['case']['connector']=='sc')
        comparison=[]
        for name,row in [('baseline',baseline),('cached',candidate)]:
            folder=DEV/row['evidence_dir'].replace('\\','/')/row['run'];policy=read(folder/'policy.json')
            log=(DEV/'.artifacts/container'/(row['run']+'_policy.log')).read_text(errors='replace');phases={}
            for m in re.finditer(r'SC_TIMING phase=(\S+).*total_wall_s=([\d.]+) sim_elapsed_s=([\d.]+)',log):phases[m[1]]=(float(m[2]),float(m[3]))
            comparison.append(dict(condition=name,run=row['run'],verified_success=True,record_wall_s=policy['wall_s'],action_sim_s=policy['sim_s'],
                port_measurement_wall_s=phases['port_perceived'][0]-phases['port_perception_start'][0],
                raw_peak_N=row['high_rate_wrist_audit']['peak_tared_force_N'],force_coverage=row['high_rate_wrist_audit']['complete_action_coverage']))
        unchanged();state.update(status='completed_development_pair_requires_review',comparison=comparison,
            final_sources_unchanged=True,not_a_randomized_success_rate=True);save()
    except Exception as exc:
        state.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
