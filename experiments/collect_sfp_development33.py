"""Scene-separated SFP training data; never reads final insertion manifests."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/sfp_visual33/data'
CONTAINER='research-aic-eval'
SOURCES=['experiments/collect_sfp_development33.py','experiments/render_perception_training_case.py',
         'experiments/render_perception_training_case.sh','experiments/lastinch_scene.py',
         'experiments/lastinch_geometry.py','experiments/challenge_grasp.py','experiments/cable_initialization.py',
         'aic_model/aic_model/insertion/sfp_plug_pose_geometry.py',
         'aic_model/aic_model/insertion/sc_plug_pose_geometry.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest=OUT/'manifest.json'
    if not manifest.exists():
        cases=[]
        for split,count,first in [('train',20,81000),('val',6,91000)]:
            for i in range(count):
                seed=first+i;rng=np.random.default_rng(seed)
                cases.append(dict(case_id=f'{split}_{seed}',split=split,seed=seed,target_index=i%5,sfp_port=(i//5)%2,
                                  gap_mm=float(rng.uniform(7,24)),lateral_mm=rng.uniform(-1.5,1.5,2).tolist(),
                                  angular_deg=rng.uniform(-2,2,3).tolist(),grasp_jitter_seed=seed))
        manifest.write_text(json.dumps(dict(purpose='training_and_development_only_sfp',split_unit='whole_scene',
                     cases=cases,source_sha256={s:sha(ROOT/s) for s in SOURCES}),indent=2)+'\n',encoding='utf-8')
    frozen=json.loads(manifest.read_text());results=[];started=time.monotonic()
    subprocess.run(['docker','start',CONTAINER],check=True,stdout=subprocess.DEVNULL)
    for case in frozen['cases']:
        if any(sha(ROOT/s)!=h for s,h in frozen['source_sha256'].items()):raise RuntimeError('Frozen data source changed')
        scene=OUT/'cases'/case['case_id'];record=scene/'training_capture.json'
        remote='/research/'+str(scene.relative_to(ROOT)).replace('\\','/')
        if not record.exists():
            if scene.exists():raise RuntimeError(f'Inspect preserved unfinished scene {scene}')
            with (OUT/(case['case_id']+'.log')).open('wb') as log:
                subprocess.run(['docker','exec','-w','/research',CONTAINER,'python3','-m','experiments.lastinch_scene',
                    '--out',remote,'--connector','sfp','--target-index',str(case['target_index']),'--sfp-port',str(case['sfp_port']),
                    '--seed',str(case['seed']),'--gap-mm',str(case['gap_mm']),'--lateral-mm',*map(str,case['lateral_mm']),
                    '--angular-deg',*map(str,case['angular_deg']),'--native-dynamic','--grasp-profile','legacy_sfp',
                    '--grasp-jitter-seed',str(case['grasp_jitter_seed'])],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=100)
                subprocess.run(['docker','exec',CONTAINER,'bash','/research/experiments/render_perception_training_case.sh',
                    '--scene',remote,'--dataset','/research/.artifacts/sfp_visual33/data/dataset','--split',case['split'],
                    '--label-connector','sfp'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=150)
        capture=json.loads(record.read_text())
        if len(capture['cameras'])!=3 or min(c['pixel_std'] for c in capture['cameras'].values())<8:raise RuntimeError('Camera integrity')
        results.append(dict(case_id=case['case_id'],split=case['split'],capture_wall_s=capture['wall_s']))
        progress=dict(status='collecting',completed=len(results),planned=len(frozen['cases']),results=results,wall_s=time.monotonic()-started)
        (OUT/'progress.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
        print(case['case_id'],len(results),'/',len(frozen['cases']),flush=True)
    for kind in ['sfp_nic','sfp_plug']:
        dataset=OUT/'dataset'/kind
        (dataset/'dataset.yaml').write_text(f'path: /research/.artifacts/sfp_visual33/data/dataset/{kind}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [{kind}]\nkpt_shape: [8, 3]\n',encoding='utf-8')
    (OUT/'dataset_SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in (OUT/'dataset').rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf-8')
    progress['status']='collection_complete';(OUT/'progress.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
