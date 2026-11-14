"""Paired static source-lighting comparison with the enclosure frame corrected."""
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.artifacts/sfp_visual33/lighting_aligned'


def main():
    OUT.mkdir(parents=True,exist_ok=True);cases=[]
    for name,gi in [('source_direct33v2',False),('source_gi128_33v2',True)]:
        with (OUT/(name+'.log')).open('wb') as log:
            args=['docker','exec','-w','/research','research-aic-eval','python3','experiments/prepare_sfp_lighting33.py','--name',name]
            if gi:args.append('--gi')
            subprocess.run(args,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90)
            subprocess.run(['docker','exec','-w','/research','research-aic-eval','bash','experiments/render_perception_training_case.sh',
                '--scene','/research/.artifacts/sfp_visual33/lighting_aligned/cases/'+name,
                '--dataset','/research/.artifacts/sfp_visual33/lighting_aligned/dataset','--split','val','--label-connector','sfp'],
                cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        cases.append(dict(case_id=name,split='val',target_index=0,sfp_port=0))
        (OUT/'progress.json').write_text(json.dumps(dict(completed=len(cases),planned=2),indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(purpose='same-geometry static illumination diagnostic, not independent trials',cases=cases),indent=2)+'\n')
    with (OUT/'original_evaluation.log').open('wb') as log:
        subprocess.run(['docker','exec','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-w','/research','research-aic-eval','bash',
            'experiments/evaluate_sfp_visual33.sh','--data-root','/research/.artifacts/sfp_visual33/lighting_aligned',
            '--nic-weights','/research/aic_example_policies/aic_example_policies/ros/weights/best.pt',
            '--plug-weights','/research/aic_example_policies/aic_example_policies/ros/weights/best_sfp_plug_pose.pt',
            '--name','lighting33_aligned_original'],cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)


if __name__=='__main__':main()
