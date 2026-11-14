"""Training-only keypoint bias/precision calibration, with frozen dev scoring.

Original checkpoint/native crop are preserved. Calibration is fitted only on
the pre-existing 20 training scenes; six static development scenes and optional
new recorded images are scored afterwards. No live policy is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import PlugPoseView,triangulate_dlt,fit_rigid_transform
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as POINTS


def estimate(views,calibration,mode):
    xyz=[];local=[];weights=[]
    for k in range(8):
        pixels=[];Ps=[];ws=[]
        for v in views:
            c=v['confidence'][k]
            if c<.15:continue
            uv=np.array(v['uv'][k]);K=np.array(v['K']);P=np.array(v['P']);name=v['camera']
            if 'bias' in mode:uv-=np.array(calibration[name]['bias_ray'][k])*[K[0,0],K[1,1]]
            if 'precision' in mode:c*=calibration[name]['relative_precision'][k]
            pixels.append(uv);Ps.append(P);ws.append(c)
        if len(pixels)<2:continue
        xyz.append(triangulate_dlt(pixels,Ps,ws));local.append(POINTS[k]);weights.append(np.mean(ws))
    if len(xyz)<6:return {'reason':'insufficient_keypoints'}
    R,pos,shape=fit_rigid_transform(np.array(local),np.array(xyz),np.array(weights))
    return {'position':pos.tolist(),'rotation':R.tolist(),'shape_mm':float(shape*1000)}


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'.artifacts/perception_data29/qualified';manifest=json.loads((data/'manifest.json').read_text());cache=a.out/'predictions_without_truth.json'
    torch.set_num_threads(2);cv2.setNumThreads(1)
    weight=ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt'
    if cache.exists():rows=json.loads(cache.read_text())
    else:
        estimator=ScPlugPoseEstimator(weight,device='cpu',crop_refine=True,crop_pad_scale=6,refinement_strategy='fixed');rows=[]
        for case in manifest['cases']:
            name=case['case_id'];split=case['split'];capture=json.loads((data/'cases'/name/'training_capture.json').read_text());views=[]
            for camera,v in capture['cameras'].items():
                path=data/'dataset/sc_plug/images'/split/(name+'_'+camera+'.png');image=cv2.imread(str(path))
                views.append(PlugPoseView(camera,image,np.array(v['K']),np.array(v['T_world_from_camera']),v['image_stamp_s'],camera))
            ds=estimator.detect_views(views);by={v.camera_name:v for v in views};row={'case_id':name,'split':split,'views':[]}
            for d in ds:
                v=by[d.camera_name];P=v.K@np.linalg.inv(v.T_world_from_camera)[:3]
                row['views'].append(dict(camera=d.camera_name,uv=d.keypoints_px.tolist(),confidence=d.keypoint_confidences.tolist(),K=v.K.tolist(),P=P.tolist(),image_shape=list(v.image_bgr.shape)))
            rows.append(row);print(json.dumps({'inferred':name,'views':len(ds)}),flush=True)
        cache.write_text(json.dumps(rows,indent=2)+'\n')
    # Only training labels are read to fit the adapter.
    errors={}
    for row in rows:
        if row['split']!='train':continue
        for v in row['views']:
            n=v['camera'];label=np.loadtxt(data/'dataset/sc_plug/labels/train'/(row['case_id']+'_'+n+'.txt'),ndmin=2)[0]
            gt=label[5:].reshape(8,3)[:,:2]*[v['image_shape'][1],v['image_shape'][0]]
            K=np.array(v['K']);err=(np.array(v['uv'])-gt)/[K[0,0],K[1,1]]
            errors.setdefault(n,[]).append(err)
    calibration={}
    variances={n:np.var(e,axis=0,ddof=1).sum(1) for n,e in errors.items()};reference=float(np.median(np.concatenate(list(variances.values()))))
    for n,e in errors.items():
        calibration[n]={'bias_ray':np.mean(e,axis=0).tolist(),'relative_precision':(reference/np.maximum(variances[n],np.finfo(float).eps)).tolist(),'training_scenes':len(e)}
    contract={'purpose':__doc__,'fit_split':'train_only_80010_to_80029','calibration':calibration,'original_weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'production_enabled':False}
    (a.out/'calibration_train_only.json').write_text(json.dumps(contract,indent=2)+'\n')
    results=[]
    for row in rows:
        if row['split']!='val':continue
        for mode in ['original','precision','bias','bias_precision']:
            results.append(dict(case_id=row['case_id'],mode=mode,**estimate(row['views'],calibration,mode)))
    (a.out/'dev_fits_without_truth.json').write_text(json.dumps(results,indent=2)+'\n')
    for row in results:
        if 'position' not in row:continue
        truth=np.array(json.loads((data/'cases'/row['case_id']/'scene.json').read_text())['reset']['tip_world'])
        row.update(diagnostic_position_mm=float(np.linalg.norm(np.array(row['position'])-truth[:3,3])*1000),diagnostic_angle_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@np.array(row['rotation'])).magnitude())))
    (a.out/'dev_diagnostic_scores.json').write_text(json.dumps(results,indent=2)+'\n')
    summary={m:{k:float(np.median([r[k] for r in results if r['mode']==m and k in r])) for k in ['diagnostic_position_mm','diagnostic_angle_deg']} for m in ['original','precision','bias','bias_precision']}
    print(json.dumps(summary,indent=2));(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
