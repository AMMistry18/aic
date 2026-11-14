"""Collect frozen, scene-separated development data with actual headless RGB.

No final evaluation cases are accessed. Static training scenes are not physical
insertion trials. Fails on incomplete camera integrity rather than inventing
labels for an unverified rendering.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/perception_data29/qualified'
CONTAINER='research-aic-eval'
SOURCES=['experiments/render_perception_training_case.py','experiments/render_perception_training_case.sh',
         'experiments/lastinch_scene.py','experiments/lastinch_geometry.py','experiments/challenge_grasp.py',
         'experiments/cable_initialization.py','experiments/collect_perception_development.py',
         'aic_model/aic_model/insertion/sc_plug_pose_geometry.py','aic_model/aic_model/insertion/sfp_plug_pose_geometry.py',
         'aic_example_policies/aic_example_policies/ros/sc_mouth_pose_geometry.py']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'manifest.json'
    if not manifest.exists():
        cases=[]
        for split,count,first in [('train',20,80010),('val',6,90010)]:
            for index in range(count):
                seed=first+index;rng=np.random.default_rng(seed)
                cases.append({'case_id':f'{split}_{seed}','split':split,'seed':seed,'target_index':index%5,
                    'gap_mm':float(rng.uniform(7,24)),'lateral_mm':rng.uniform(-1.5,1.5,2).tolist(),
                    'angular_deg':rng.uniform(-3,3,3).tolist()})
        frozen={'purpose':'training/development only; final 100000 series untouched','cases':cases,
            'split_unit':'complete independent randomized scene and grasp',
            'source_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES}}
        manifest.write_text(json.dumps(frozen,indent=2)+'\n')
    frozen=json.loads(manifest.read_text());results=[];started=time.monotonic()
    progress={'status':'collecting','planned_scenes':len(frozen['cases']),'completed_scenes':0,'results':results}
    subprocess.run(['docker','start',CONTAINER],check=True,stdout=subprocess.DEVNULL)
    for case in frozen['cases']:
        for source,digest in frozen['source_sha256'].items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Frozen collection source changed: {source}')
        name=case['case_id'];scene=OUT/'cases'/name;record=scene/'training_capture.json'
        if not record.exists():
            if scene.exists():raise RuntimeError(f'Unfinished case preserved; inspect {scene} before resuming')
            log=OUT/f'{name}.log'
            remote=f'/research/.artifacts/perception_data29/qualified/cases/{name}'
            with log.open('wb') as stream:
                subprocess.run(['docker','exec','-w','/research',CONTAINER,'python3','-m','experiments.lastinch_scene',
                    '--out',remote,'--connector','sc','--target-index',str(case['target_index']),'--seed',str(case['seed']),
                    '--gap-mm',str(case['gap_mm']),'--lateral-mm',*map(str,case['lateral_mm']),'--angular-deg',*map(str,case['angular_deg']),
                    '--native-dynamic','--cable-initialization','sc_straight_outboard','--grasp-profile','challenge_sc',
                    '--grasp-jitter-seed',str(case['seed'])],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=90)
                subprocess.run(['docker','exec',CONTAINER,'bash','/research/experiments/render_perception_training_case.sh',
                    '--scene',remote,'--dataset','/research/.artifacts/perception_data29/qualified/dataset','--split',case['split']],
                    check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        capture=json.loads(record.read_text())
        if sum(v>=300 for v in capture['camera_integrity_blue_counts'].values())<2:raise RuntimeError('Camera integrity failed')
        results.append({'case_id':name,'split':case['split'],'camera_count':len(capture['cameras']),'capture_wall_s':capture['wall_s']})
        progress.update(completed_scenes=len(results),wall_s=time.monotonic()-started,current_case=name)
        (OUT/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps({k:progress[k] for k in ['completed_scenes','planned_scenes','current_case','wall_s']}),flush=True)
    progress['status']='collection_complete'
    (OUT/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    for kind,points in [('sc_plug',8),('sc_mouth',5)]:
        root=OUT/'dataset'/kind
        (root/'dataset.yaml').write_text(f'path: /research/.artifacts/perception_data29/qualified/dataset/{kind}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [{kind}]\nkpt_shape: [{points}, 3]\n')
    print('Collection complete; neural accuracy and insertion robustness remain unevaluated.',flush=True)


if __name__=='__main__':main()
