"""One-pass path repair in the copied scene startup; preserve original attempt."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DEV=ROOT/'.artifacts/dev46_workspace'
REMOTE='/research/.artifacts/dev46_workspace'


def main():
    out=ROOT/'experiments/runtime_qualification47_v2';out.mkdir(exist_ok=True);changes=[]
    if (out/'path_repair.json').exists():raise FileExistsError('Preserve this repair version')
    for name in ['start_kilted_handoff.sh','prepare_kilted_handoff.py']:
        path=DEV/'experiments'/name;before=path.read_bytes();(out/(name+'.before')).write_bytes(before)
        text=(ROOT/'experiments'/name).read_text()
        if name.endswith('.sh'):
            text=text.replace('/research/',REMOTE+'/')
            text=text.replace('ROS_DOMAIN_ID=47','ROS_DOMAIN_ID=49').replace('research_lastinch_kilted','research_lastinch_dev46')
            text=text.replace('export LIBGL_ALWAYS_SOFTWARE=1','export LIBGL_ALWAYS_SOFTWARE=1\nexport LP_NUM_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1')
        text=text.replace('/opt/research-grasp','/opt/research-dev47-grasp')
        path.write_text(text,encoding='utf-8',newline='\n')
        if REMOTE+'/.artifacts/dev46_workspace' in text:raise RuntimeError('Repeated prefix')
        changes.append(dict(path=str(path.relative_to(DEV)),before_sha256=hashlib.sha256(before).hexdigest(),
                            after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out/'path_repair.json').write_text(json.dumps(dict(changes=changes,final_sources_changed=False),indent=2))
    flight=out/'preflight.sh'
    flight.write_text('''#!/usr/bin/env bash
set -eo pipefail
source /ws_aic/install/setup.bash
root=/research/.artifacts/dev46_workspace
test -f "$root/experiments/gazebo/CMakeLists.txt"
test -x /opt/research-perception/bin/python
bash -n "$root/experiments/start_kilted_handoff.sh"
cmake -S "$root/experiments/gazebo" -B /opt/research-dev47-grasp
cmake --build /opt/research-dev47-grasp -j 1
''',newline='\n')
    print(json.dumps(dict(repaired_paths=True,final_sources_changed=False)))


if __name__=='__main__':main()
