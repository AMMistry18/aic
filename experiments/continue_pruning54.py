"""Qualify exact association pruning on dev_45005, never on final cases."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / '.artifacts/dev46_workspace'
OUT = ROOT / 'experiments/pruning_followup54'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'progress.json').exists():
        raise FileExistsError('Preserve existing attempt')
    state = dict(status='preparing', final_cases_used_for_tuning=0,
                 comparison='runtime47v5 vs v6; exact NIC association pruning only',
                 prospective_case='dev_45005', thresholds_changed=False)
    def save():
        state['updated_unix'] = time.time()
        p = OUT / 'progress.tmp'
        p.write_text(json.dumps(state, indent=2)); p.replace(OUT / 'progress.json')
    freeze = read(ROOT / 'experiments/full_evaluation44/source_freeze.json')['source_sha256']
    def unchanged():
        if any(sha(ROOT / p) != h for p, h in freeze.items()):
            raise RuntimeError('Frozen final sources changed')
    save()
    try:
        unchanged()
        bound = ROOT / 'experiments/candidates/efficiency48/epipolar_bound.py'
        evidence = read(ROOT / 'experiments/perception_efficiency48/fixture_benchmark.json')
        if not evidence['all_accepted_candidates_identical'] or sha(bound) != evidence['source_sha256']:
            raise RuntimeError('Exact development replay/source proof missing')
        target = DEV / 'aic_model/aic_model/insertion/sfp_fixture_refinement.py'
        original = target.read_text(encoding='utf-8-sig')
        if original != (ROOT / 'aic_model/aic_model/insertion/sfp_fixture_refinement.py').read_text(encoding='utf-8-sig'):
            raise RuntimeError('Unexpected development fixture implementation')
        anchor = '                fit=fit_fixture(picks)'
        if original.count(anchor) != 1:
            raise RuntimeError('Ambiguous insertion point')
        candidate = original.replace(anchor,
            '                from .epipolar_bound import impossible_under_original_gate\n'
            '                if impossible_under_original_gate(picks):continue\n' + anchor)
        shutil.copy2(target, OUT / 'baseline_sfp_fixture_refinement.py')
        (OUT / 'candidate_sfp_fixture_refinement.py').write_text(candidate, encoding='utf-8')
        shutil.copy2(bound, target.with_name('epipolar_bound.py'))
        target.write_text(candidate, encoding='utf-8')
        state.update(status='waiting_for_serialized_physical_test',
                     candidate_sha256=sha(target), bound_sha256=sha(bound))
        save()
        with (OUT / 'physical.log').open('wb') as log:
            subprocess.run([sys.executable, 'experiments/development_slot47.py',
                '--revision', '6', '--only', 'sfp'], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, timeout=3300, check=True)
        result = read(ROOT / 'experiments/runtime_qualification47_v6/progress.json')['results'][0]
        unchanged()
        state.update(status='completed_requires_review', result=result,
                     final_sources_unchanged=True)
        save()
    except Exception as exc:
        state.update(status='stopped_requires_inspection', error=f'{type(exc).__name__}: {exc}')
        save()
        raise

if __name__ == '__main__':
    main()
