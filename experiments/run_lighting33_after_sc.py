"""Serial static lighting diagnosis after the owned SC regression batch exits."""
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.artifacts/sfp_visual33/lighting'


def command(args,log,timeout=180):
    with (OUT/(log+'.log')).open('wb') as f:subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    while True:
        phase=json.loads((ROOT/'.artifacts/repair33/progress.json').read_text())['status']
        if phase=='development_complete_needs_inspection':break
        if phase=='queue_failed':raise RuntimeError('Inspect SC queue failure before using its container')
        time.sleep(15)
    command(['docker','start','research-aic-eval'],'start')
    cases=[]
    for name,gi in [('source_direct33v1',False),('source_gi128_33v1',True)]:
        args=['docker','exec','-w','/research','research-aic-eval','python3','experiments/prepare_sfp_lighting33.py','--name',name]
        if gi:args.append('--gi')
        command(args,name+'_prepare')
        command(['docker','exec','-w','/research','research-aic-eval','bash','experiments/render_perception_training_case.sh',
                 '--scene','/research/.artifacts/sfp_visual33/lighting/cases/'+name,
                 '--dataset','/research/.artifacts/sfp_visual33/lighting/dataset','--split','val','--label-connector','sfp'],name+'_render',240)
        cases.append(dict(case_id=name,split='val',target_index=0,sfp_port=0))
        (OUT/'progress.json').write_text(json.dumps(dict(status='rendering',completed=len(cases),planned=2),indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(purpose='lighting diagnostic; paired same training geometry, not independent trials',cases=cases),indent=2)+'\n')
    command(['docker','exec','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-w','/research','research-aic-eval','bash',
             'experiments/evaluate_sfp_visual33.sh','--data-root','/research/.artifacts/sfp_visual33/lighting',
             '--nic-weights','/research/aic_example_policies/aic_example_policies/ros/weights/best.pt',
             '--plug-weights','/research/aic_example_policies/aic_example_policies/ros/weights/best_sfp_plug_pose.pt',
             '--name','lighting33_original'], 'original_evaluation',600)
    (OUT/'progress.json').write_text(json.dumps(dict(status='complete',completed=2,planned=2),indent=2)+'\n')


if __name__=='__main__':main()
