"""Replay a static CAD-rim tracker on frozen live RGB/exposure-time geometry."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_rim_refinement import refine_with_agreement


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[];prior=None
    for folder in sorted(a.inputs.iterdir()):
        if not (folder/'inputs.json').is_file():continue
        data=json.loads((folder/'inputs.json').read_text())
        views=[{'image':cv2.imread(str(folder/f'{v["camera"]}.png')),'P':np.array(v['P'])} for v in data['views']]
        xyz,R=(np.array(data['initial_position']),np.array(data['initial_rotation'])) if prior is None else prior
        accepted,result=refine_with_agreement(views,xyz,R,raster_offset_px=-.5)
        result['capture']=folder.name;result['uses_cached_pose']=prior is not None
        rows.append(result)
        if accepted is not None:prior=accepted
        print(folder.name,result['reason'],result.get('disagreements_m'),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
