"""Replay original learned checkpoints on archived, exposure-calibrated images.

No ROS node or simulator is started. Ground truth is read only after inference
for diagnostic scoring, never for candidate selection or estimator fitting.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'aic_model'))
sys.path.insert(0, str(ROOT/'aic_example_policies'))
from aic_model.insertion.sc_plug_pose import ScPlugPoseEstimator
from aic_model.insertion.sfp_plug_pose import PlugPoseView, PlugKeypointDetection, fuse_multiview_keypoints, SfpPlugPoseEstimator
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
from aic_example_policies.ros.perception_core import PerceptionCore
from aic_model.insertion.perception_quality import select_geometric_refinement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--connector',choices=['sc','sfp'],default='sc')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    archive = json.loads(args.audit.read_text())
    views = [PlugPoseView(r['camera'], cv2.imread(str(args.audit.parent/f'{r["camera"]}_raw.png')),
                         np.array(r['K']), np.linalg.inv(r['T_camera_from_base_exact']), r['stamp_s'], r['camera'])
             for r in archive['views']]
    weights = ROOT/'aic_example_policies/aic_example_policies/ros/weights'
    model_path = weights/f'best_{args.connector}_plug_pose.pt'
    points=SC_PLUG_LOCAL_KEYPOINTS_M if args.connector=='sc' else SFP_PLUG_LOCAL_KEYPOINTS_M
    estimator_type=ScPlugPoseEstimator if args.connector=='sc' else SfpPlugPoseEstimator
    est = estimator_type(model_path, device='cpu', crop_refine=False, refinement_strategy='fixed')
    result = {'purpose': 'original_checkpoint_replay_development_only', 'robot_commands_sent': 0,
              'GT_used_for_inference_or_selection': False, 'connector':args.connector,'audit_sha256': sha(args.audit),
              'source_images': {v.camera_name: sha(args.audit.parent/f'{v.camera_name}_raw.png') for v in views},
              'weights': {name: sha(weights/name) for name in ['best.pt','best_sc_mouth_pose.pt','best_sc_plug_pose.pt','best_sfp_plug_pose.pt']},
              'conditions': [], 'all_full_frame_candidates': {}}
    # Keep all detections: independent per-camera argmax is the original path.
    start = time.monotonic()
    raw = est._load_model()([v.image_bgr for v in views], imgsz=est.imgsz, conf=est.conf_threshold, verbose=False, device='cpu')
    all_dets = {}
    for view, prediction in zip(views, raw):
        candidates = []
        if prediction.keypoints is not None and prediction.boxes is not None:
            for i in range(len(prediction.boxes)):
                candidates.append({'keypoints_px': prediction.keypoints.xy[i].cpu().numpy().tolist(),
                                   'keypoint_confidences': prediction.keypoints.conf[i].cpu().numpy().tolist(),
                                   'box_confidence': float(prediction.boxes.conf[i]),
                                   'bbox_xyxy': prediction.boxes.xyxy[i].cpu().numpy().tolist()})
        result['all_full_frame_candidates'][view.camera_name] = candidates
        all_dets[view.camera_name] = [PlugKeypointDetection(view.camera_name, np.array(x['keypoints_px']),
                                   np.array(x['keypoint_confidences']), x['box_confidence']) for x in candidates]
    result['cold_full_frame_wall_s'] = time.monotonic()-start
    fits = []
    for names in itertools.combinations([v.camera_name for v in views], 2):
        for indices in itertools.product(*(range(len(all_dets[n])) for n in names)):
            ds = [all_dets[n][i] for n,i in zip(names,indices)]
            try:
                p,R,shape,reprojection,count,confidence = fuse_multiview_keypoints(views,ds,local_keypoints_m=points)
                fits.append({'cameras':names,'candidate_indices':indices,'position':p.tolist(),'rotation':R.tolist(),
                             'shape_rmse_mm':shape*1000,'reprojection_px':reprojection,
                             'original_geometry_gates_passed':shape<=.0035 and reprojection<=6})
            except ValueError:
                pass
    result['all_pair_fits_without_GT_selection'] = sorted(fits,key=lambda x:x['reprojection_px'])
    alternatives=[]
    for crop,pad in [(False,6),(True,3),(True,6)]:
        est.crop_refine=crop;est.crop_pad_scale=pad
        start=time.monotonic();ds=est.detect_views(views)
        row={'crop_refine':crop,'pad':pad,'wall_s':time.monotonic()-start,
             'detections':[{'camera':d.camera_name,'keypoints_px':d.keypoints_px.tolist(),
                            'keypoint_confidences':d.keypoint_confidences.tolist(),'box_confidence':d.box_confidence} for d in ds]}
        try:
            p,R,shape,reprojection,count,confidence=fuse_multiview_keypoints(views,ds,local_keypoints_m=points)
            row.update(position=p.tolist(),rotation=R.tolist(),shape_rmse_mm=shape*1000,reprojection_px=reprojection,
                       original_geometry_gates_passed=shape<=.0035 and reprojection<=6)
        except ValueError as exc:
            row['failure']=str(exc)
        result['conditions'].append(row)
        if not crop or pad==6:alternatives.append(('native_crop' if crop else 'full_frame',ds))
        for v in views:
            img=v.image_bgr.copy()
            for det in ds:
                if det.camera_name==v.camera_name:
                    for k,point in enumerate(det.keypoints_px):
                        uv=tuple(np.rint(point).astype(int));cv2.circle(img,uv,4,(0,0,255),-1)
                        cv2.putText(img,str(k),uv,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.imwrite(str(args.out/f'{v.camera_name}_crop{int(crop)}_pad{pad}.jpg'),img)
    pc=PerceptionCore(nic_weights=str(weights/'best.pt'),sc_weights=str(weights/'best_sc_mouth_pose.pt'))
    result['port_detector_methods']=[n for n in dir(pc) if n.startswith('detect_')]
    selected,result['selective_refinement']=select_geometric_refinement(
        views,alternatives,points,max_reprojection_error_px=6.,
        max_keypoint_rmse_m=.0035,max_pose_disagreement_m=.00025 if args.connector=='sc' else .0001)
    result['port_detections']={}
    for view in views:
        start=time.monotonic()
        sc=pc.detect_sc_pose(view.image_bgr)
        nic=pc.detect_nic(view.image_bgr)
        result['port_detections'][view.camera_name]={'sc':sc,'nic':nic,'wall_s':time.monotonic()-start}
    # Only score ground truth AFTER all candidates and inference are complete.
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    truth=np.linalg.inv(base)@np.array(archive['views'][0]['tip_world'])
    for fit in result['conditions']+result['all_pair_fits_without_GT_selection']:
        if 'position' in fit:
            fit['diagnostic_GT_tip_error_mm']=float(np.linalg.norm(np.array(fit['position'])-truth[:3,3])*1000)
            fit['diagnostic_GT_rotation_error_deg']=float(Rotation.from_matrix(truth[:3,:3].T@np.array(fit['rotation'])).magnitude()*180/np.pi)
    (args.out/'replay.json').write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
    print(json.dumps({'conditions':[{k:v for k,v in c.items() if k!='detections'} for c in result['conditions']],
                      'candidates_per_view':{n:len(v) for n,v in all_dets.items()},
                      'pair_fit_count':len(fits),'passing_pair_fits':sum(x['original_geometry_gates_passed'] for x in fits),
                      'selective_refinement':result['selective_refinement'],
                      'port_detection_counts':{n:{k:len(v[k]) for k in ['sc','nic']} for n,v in result['port_detections'].items()}},indent=2))


if __name__=='__main__':
    main()
