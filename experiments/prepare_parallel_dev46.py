"""Create a small independent GPU development workspace; never edit final44.

Source files are separate copies. Immutable meshes and weights are hard-linked
to avoid duplicating storage. Only the idle Intrinsic perception container is
used; ROS domain, Gazebo partition, artifacts and native plugin build differ.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'.artifacts/dev46_workspace'
REMOTE='/research/.artifacts/dev46_workspace'


def main():
    if DEV.exists():raise FileExistsError('Preserve existing development workspace')
    DEV.mkdir(parents=True)
    freeze=json.loads((ROOT/'experiments/full_evaluation44/source_freeze.json').read_text())
    files=set(freeze['source_sha256'])
    files.add('experiments/gazebo/CMakeLists.txt')
    for folder in ['aic_model/aic_model','aic_example_policies/aic_example_policies','flowstate/aic_perception/aic_perception']:
        files.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py','.npz','.yaml','.json'))
    files.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'experiments').glob('*.py'))
    files.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'experiments').glob('*.sh'))
    files.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'experiments/configs').glob('*.json'))
    original_hashes={}
    for name in sorted(files):
        source=ROOT/name;target=DEV/name;target.parent.mkdir(parents=True,exist_ok=True)
        original_hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        if source.suffix=='.pt':target.hardlink_to(source)
        elif source.suffix in ('.py','.sh'):
            text=source.read_text(encoding='utf-8-sig')
            text=text.replace('/research/',REMOTE+'/').replace("'/research'",repr(REMOTE)).replace('ROS_DOMAIN_ID=47','ROS_DOMAIN_ID=49')
            text=text.replace('research_lastinch_kilted','research_lastinch_dev46')
            text=text.replace('/opt/research-grasp','/opt/research-dev46-grasp')
            if name=='experiments/run_seeded_evaluation.py':
                text=text.replace("CONTAINER='research-aic-eval'","CONTAINER='research-aic-perception30'")
            target.write_text(text,encoding='utf-8',newline='\n')
        else:shutil.copy2(source,target)
    # Immutable asset files only, never shared .git metadata or source code.
    asset_files=0
    for folder in ['.artifacts/upstream-phase1','.artifacts/ur_description','.artifacts/lastinch_resources']:
        for source in (ROOT/folder).rglob('*'):
            if not source.is_file() or '.git' in source.parts:continue
            target=DEV/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            target.hardlink_to(source);asset_files+=1
    (DEV/'.artifacts/container').mkdir(parents=True,exist_ok=True)
    (DEV/'workspace_provenance.json').write_text(json.dumps(dict(
        purpose='Independent DEVELOPMENT only; never run final seeds here',
        container='research-aic-perception30',ros_domain_id=49,gz_partition='research_lastinch_dev46',
        source_parent_sha256=original_hashes,immutable_asset_hardlinks=asset_files,
        shared_assets_and_checkpoints_must_never_be_edited=True,
        final_policy_files_changed=False,remote_workspace=REMOTE),indent=2))
    print(json.dumps(dict(workspace=str(DEV),copied_sources=len(files),asset_hardlinks=asset_files)))


if __name__=='__main__':main()
