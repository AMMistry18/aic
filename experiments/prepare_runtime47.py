"""Restore full-scene development paths and enable complete runtime qualification."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'.artifacts/dev46_workspace'
OUT=ROOT/'experiments/runtime_qualification47'
REMOTE='/research/.artifacts/dev46_workspace'


def main():
    OUT.mkdir(exist_ok=True);record=OUT/'preparation.json'
    if record.exists():raise FileExistsError('Runtime preparation is versioned; inspect before rerun')
    changes=[];backup=OUT/'development_sources_before'
    def replace(path,text):
        old=path.read_bytes();new=text.replace('\r\n','\n').encode('utf-8')
        if old==new:return
        relative=path.relative_to(DEV);saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True)
        saved.write_bytes(old);path.write_bytes(new)
        changes.append(dict(path=str(relative),before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(new).hexdigest()))
    for folder in ['experiments','aic_model','aic_example_policies','flowstate']:
        for path in (DEV/folder).rglob('*'):
            if path.suffix not in ('.py','.sh') or not path.is_file():continue
            text=path.read_text(encoding='utf-8-sig')
            text=text.replace('/opt/research/.artifacts/dev46_workspace-','/opt/research-')
            text=text.replace('/sensor/research/.artifacts/dev46_workspace_','/sensor/research_')
            text=text.replace('/research/.artifacts/dev46_workspace/contacts/','/research/contacts/')
            replace(path,text)
    # Production preparation retains every mesh and collision at all five
    # positions; ROOT is derived from this copied file's location.
    replace(DEV/'experiments/prepare_kilted_handoff.py',(ROOT/'experiments/prepare_kilted_handoff.py').read_text())
    text=(ROOT/'experiments/start_kilted_handoff.sh').read_text()
    text=text.replace('/research/',REMOTE+'/')
    text=text.replace('ROS_DOMAIN_ID=47','ROS_DOMAIN_ID=49').replace('research_lastinch_kilted','research_lastinch_dev46')
    text=text.replace('export LIBGL_ALWAYS_SOFTWARE=1','export LIBGL_ALWAYS_SOFTWARE=1\nexport LP_NUM_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1')
    replace(DEV/'experiments/start_kilted_handoff.sh',text)
    path=DEV/'experiments/run_seeded_evaluation.py'
    replace(path,path.read_text().replace("'AIC_DEV_SHORT_TEST=1'","'AIC_DEV_SHORT_TEST=0'"))
    path=DEV/'experiments/run_original_insertion.py'
    replace(path,path.read_text().replace("'short_probe_no_insertion_claim'","'complete_runtime_qualification_no_angular_recovery_claim'"))
    freeze=json.loads((ROOT/'experiments/full_evaluation44/source_freeze.json').read_text())['source_sha256']
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in freeze.items()):raise RuntimeError('Final source changed')
    record.write_text(json.dumps(dict(scope='Copied development files only; full scene and full original insertion with runtime callback',
        changes=changes,weights_changed=False,final_sources_unchanged=True,all_five_positions_rendered=True,
        ROS_DOMAIN_ID=49,remote_workspace=REMOTE,physics_and_acceptance_criteria_unchanged=True),indent=2))
    print(json.dumps(dict(changed_development_files=len(changes),final_sources_unchanged=True)))


if __name__=='__main__':main()
