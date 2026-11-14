"""Evaluator-only archive adapter for offline original-network diagnostics.

Camera pixels/exposure TF are independent of truth. Nearest-stamp truth is
attached here only for scoring after replay; this file is never a live input.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--capture',default='perception_before')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.run_dir/'truth_trace.json.gz','rt') as stream:truth=json.load(stream)
    camera=json.loads((a.run_dir/a.capture/'camera_geometry.json').read_text())
    rows=[]
    for view in camera:
        row=min(truth,key=lambda r:abs(r['sim_time_s']-view['stamp_s']))
        skew=abs(row['sim_time_s']-view['stamp_s'])
        if skew>.015:raise ValueError(f'Truth diagnostic skew {skew}s is too large')
        name=view['camera']
        shutil.copy2(a.run_dir/a.capture/f'{name}_raw.png',a.out/f'{name}_raw.png')
        rows.append({'camera':name,'stamp_s':view['stamp_s'],'K':view['K'],
                     'T_camera_from_base_exact':np.linalg.inv(view['T_base_from_camera']).tolist(),
                     'tip_world':row['tip_world'],'diagnostic_truth_skew_s':skew})
    (a.out/'audit.json').write_text(json.dumps({'purpose':'evaluator-only post-inference diagnostic truth',
        'source_run':str(a.run_dir),'capture':a.capture,'views':rows},indent=2))
    print(json.dumps({'cameras':len(rows),'max_truth_skew_s':max(r['diagnostic_truth_skew_s'] for r in rows)}))


if __name__=='__main__':main()
