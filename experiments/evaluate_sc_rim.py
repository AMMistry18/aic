"""Scene-separated static development replay of public-rim refinement."""
import json
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.diagnose_sc_rim import fit_rim


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raster-offset-px',type=float,default=0.)
    parser.add_argument('--label',default='development');parser.add_argument('--full-rotation',action='store_true');parser.add_argument('--agreement',action='store_true');args=parser.parse_args()
    data=ROOT/'.artifacts/perception_data29/qualified'
    predictions=json.loads((ROOT/'.artifacts/perception_data29/evaluation/mouth_transfer_original_plug_multiview_val_without_truth.json').read_text())
    out=ROOT/'.artifacts/reference30'/args.label;out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for row in predictions:
        if 'port_position' not in row: continue
        case=row['case_id'];capture=json.loads((data/'cases'/case/'training_capture.json').read_text())
        views=[{'camera':name,'image':cv2.imread(str(data/'dataset/sc_plug/images/val'/f'{case}_{name}.png')),
                'P':np.array(v['K'])@np.linalg.inv(v['T_world_from_camera'])[:3]} for name,v in capture['cameras'].items()]
        q=row['port_quat_wxyz'];R=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
        if args.agreement:
            sys.path.insert(0,str(ROOT/'aic_model'))
            from aic_model.insertion.sc_rim_refinement import refine_with_agreement
            accepted,fit=refine_with_agreement(views,np.array(row['port_position']),R,raster_offset_px=args.raster_offset_px)
            fit['accepted']=accepted is not None
        else:
            fit=fit_rim(views,np.array(row['port_position']),R,raster_offset_px=args.raster_offset_px,full_rotation=args.full_rotation)
        fit.update(case_id=case,initial_position=row['port_position'])
        rows.append(fit);print(case,fit['mean_edge_distance_px'],fit.get('reason'),fit.get('disagreements_m'),flush=True)
    (out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        truth=np.array(json.loads((data/'cases'/row['case_id']/'scene.json').read_text())['reset']['mouth_world'])
        error=truth[:3,:3].T@(np.array(row['position'])-truth[:3,3])
        row.update(diagnostic_error_mm=float(np.linalg.norm(error)*1000),diagnostic_lateral_mm=float(np.linalg.norm(error[:2])*1000),
                   diagnostic_axial_mm=float(error[2]*1000),diagnostic_angle_deg=float(Rotation.from_matrix(truth[:3,:3].T@row['rotation']).magnitude()*180/np.pi))
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['case_id','mean_edge_distance_px','diagnostic_error_mm','diagnostic_lateral_mm','diagnostic_axial_mm','diagnostic_angle_deg']} for r in rows],indent=2))


if __name__=='__main__':main()
