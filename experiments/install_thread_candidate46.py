"""Install the verified runtime guard only into the copied development workspace."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'.artifacts/dev46_workspace'
OUT=ROOT/'experiments/candidates/efficiency46'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    proof=json.loads((ROOT/'experiments/parallel_dev46/thread_callback_verification.json').read_text())
    if not proof['passed']:raise RuntimeError('Real inference callback check required')
    if (OUT/'installation.json').exists():raise FileExistsError('Preserve candidate version; do not reinstall')
    freeze=json.loads((ROOT/'experiments/full_evaluation44/source_freeze.json').read_text())['source_sha256']
    if any(sha(ROOT/p)!=h for p,h in freeze.items()):raise RuntimeError('Frozen final source changed')
    paths=['aic_model/aic_model/insertion/sfp_plug_pose.py',
           'aic_example_policies/aic_example_policies/ros/perception_core.py']
    changes=[];patch=[]
    for name in paths:
        path=(DEV/name).resolve()
        if DEV.resolve() not in path.parents:raise RuntimeError('Outside development workspace')
        before=path.read_text();after=before
        if name.endswith('sfp_plug_pose.py'):
            old='            self._model = YOLO(str(self._weights_path))'
            new='            from .perception_runtime_candidate import bind_inference_threads\n            self._model = bind_inference_threads(YOLO(str(self._weights_path)), cpu_threads=1)'
            if after.count(old)!=1:raise RuntimeError('Unknown plug estimator version')
            after=after.replace(old,new)
        else:
            for target,argument in [('_yolo','_nic_weights'),('_sc_yolo','_sc_weights')]:
                old=f'            self.{target} = YOLO(self.{argument})'
                new=f'            from aic_model.insertion.perception_runtime_candidate import bind_inference_threads\n            self.{target} = bind_inference_threads(YOLO(self.{argument}), cpu_threads=1)'
                if after.count(old)!=1:raise RuntimeError('Unknown port detector version')
                after=after.replace(old,new)
        previous=sha(path);path.write_text(after,encoding='utf-8',newline='\n')
        changes.append(dict(path=name,before_sha256=previous,after_sha256=sha(path)))
        patch+=list(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=name,tofile=name))
    helper=DEV/'aic_model/aic_model/insertion/perception_runtime_candidate.py'
    if helper.exists():raise FileExistsError('Helper already exists')
    shutil.copy2(OUT/'perception_runtime.py',helper)
    (OUT/'thread_guard.patch').write_text(''.join(patch))
    (OUT/'installation.json').write_text(json.dumps(dict(scope='Copied development workspace only; frozen final policy untouched',
        installed_workspace=str(DEV),changes=changes,helper_sha256=sha(helper),
        inference_proof='experiments/parallel_dev46/thread_callback_verification.json',
        weights_changed=False,acceptance_thresholds_changed=False,physical_insertion_qualified=False),indent=2))
    print(json.dumps(dict(installed_in_development_copy=True,final_sources_unchanged=True)))


if __name__=='__main__':main()
