"""Replay the remaining historical development cases without policy tuning."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_seeded_evaluation import SOURCES

OUT = ROOT/'experiments/recovery_20261112'
SC = 'original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff'
SFP = 'original_perception_sfp_contact_datum_handoff'
PLAN = [('sc',43001),('sfp',45009),('sc',43003),('sfp',45002),
        ('sc',43004),('sfp',45001),('sc',42001),('sfp',45008),
        ('sc',42004),('sfp',45007),('sc',42003),('sfp',45004),
        ('sc',43000),('sfp',45006),('sc',42002)]


def completed_case_ids(results):
    # Case identity includes grasp selection; simulator seeds can be shared.
    return {r['run'].split('_recovery')[0] for r in results if r.get('status')=='completed'}


def main():
    folder = OUT/'historical_batch_replay'
    resume='--resume' in sys.argv
    folder.mkdir(exist_ok=resume)
    sources = set(SOURCES)
    sources.update(str(p.relative_to(ROOT)).replace('\\','/')
                   for p in (OUT/'checkpoints').rglob('best.pt'))
    sources.update(['experiments/configs/sc_finish41_qualification.json',
                    'experiments/configs/sfp_contact34_development.json',
                    'experiments/recovery_20261112/revalidated_historical_contact_proof.json'])
    if resume:
        state=json.loads((folder/'progress.json').read_text())
        if state['status']!='stopped':raise RuntimeError('Resume requires stopped ledger')
        (folder/('progress_before_resume_'+str(int(time.time()))+'.json')).write_text(json.dumps(state,indent=2))
        hashes=state['source_sha256']
        state['resume_count']=state.get('resume_count',0)+1
        state.pop('error',None)
        label='recovery20261112resume'+str(state['resume_count'])
    else:
        hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)}
        state = dict(status='starting', purpose='historical_development_case_replay_not_held_out',
                     source_sha256=hashes, planned_remaining=PLAN, results=[])
        for kind in ('sc','sfp'):
            state['results'] += json.loads((OUT/f'{kind}_verification_v1_batch.json').read_text(encoding='utf-8-sig'))['results']
        label='recovery20261112v1'
    def save():
        state['updated_unix'] = time.time()
        (folder/'progress.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        for kind,seed in PLAN:
            if f'dev_{seed}' in completed_case_ids(state['results']):
                continue
            for path,digest in hashes.items():
                if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Frozen source changed: '+path)
            case=f'dev_{seed}'; method=SC if kind=='sc' else SFP
            state.update(status='running',current_case=case);save()
            args=[sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest',
                  'experiments/configs/'+('sc_finish41_qualification.json' if kind=='sc' else 'sfp_contact34_development.json'),
                  '--case-ids',case,'--methods',method,'--label',label,
                  '--physics-backend','source_float64_damped','--physics-step','0.0005',
                  '--wrist-rate','0','--solver-iterations','200','--force-ceiling-N','20',
                  '--positive-control-proof','experiments/recovery_20261112/revalidated_historical_contact_proof.json']
            weights='experiments/recovery_20261112/checkpoints/'
            if kind=='sc':
                args += ['--grasp-profile','challenge_sc','--cable-initialization','sc_straight_outboard',
                         '--sc-mouth-weights',weights+'sc_mouth_recovery20261112/best.pt']
            else:
                args += ['--sfp-motion-profile','fast38','--nic-weights',weights+'nic_recovery20261112/best.pt',
                         '--sfp-plug-weights',weights+'plug_recovery20261112/best.pt']
            tag=case+'_'+label if resume else case
            with (folder/(tag+'.log')).open('wb') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2000)
            batch=json.loads((ROOT/'.artifacts/seeded_evaluation_progress.json').read_text())
            result=batch['results'][0]
            if len(batch['results'])!=1 or result['run']!=case+'_'+label+'_'+method:
                raise RuntimeError('Unexpected result identity')
            (folder/(tag+'_batch.json')).write_text(json.dumps(batch,indent=2))
            state['results'].append(result);save()
            if result['status']!='completed':
                raise RuntimeError('Infrastructure failure preserved; inspect before resuming')
            with (folder/(tag+'_export.log')).open('wb') as log:
                subprocess.run([sys.executable,'experiments/export_original_repair_evidence.py',
                    '--runs',result['run'],'--out',str(folder/'evidence'/tag)],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        state.update(status='complete_requires_analysis',current_case=None);save()
    except Exception as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}');save()
        raise


if __name__=='__main__':
    main()
