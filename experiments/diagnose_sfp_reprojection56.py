"""Separate DLT/Kabsch fitting error from learned-image inconsistency on DEV RGB.

No truth is read. No rejected pose is sent to a controller. This is a diagnostic
of the original 6 px gate, not an authorization to relax it or ignore views.
"""
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from experiments.angular_pose_development46 import fit_joint
from aic_model.insertion.sfp_plug_pose import PlugPoseView,PlugKeypointDetection,fuse_multiview_keypoints
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M

def main():
    out=ROOT/'experiments/reprojection_development56';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Preserve existing diagnosis')
    benchmark=json.loads((ROOT/'experiments/parallel_dev46/cropped_device_benchmark.json').read_text())
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    runs={r['run']:r for r in ledger['results']};rows=[]
    cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        for row in benchmark['rows']:
            if row['connector']!='sfp':continue
            run=runs[row['run']]
            folder=ROOT/run['evidence_dir'].replace('\\','/')/run['run']/'perception_before'
            spec=json.loads((folder/'camera_geometry.json').read_text())
            inputs=[dict(camera=v['camera'],K=v['K'],T_world_from_camera=v['T_base_from_camera']) for v in spec]
            views=[PlugPoseView(v['camera'],np.zeros((1,1,3),np.uint8),np.array(v['K']),
                np.array(v['T_base_from_camera']),v['stamp_s'],v['frame_id']) for v in spec]
            detection_rows=row['measurements']['0']['outputs'][0]['detections']
            ds=[PlugKeypointDetection(d['camera_name'],np.array(d['keypoints_px']),
                np.array(d['keypoint_confidences']),d['box_confidence']) for d in detection_rows]
            p,R,rmse,reprojection,count,conf=fuse_multiview_keypoints(views,ds,local_keypoints_m=SFP_PLUG_LOCAL_KEYPOINTS_M)
            joint=fit_joint(inputs,detection_rows,SFP_PLUG_LOCAL_KEYPOINTS_M,p,R)
            result=dict(run=row['run'],original_failure=row['measurements']['0']['outputs'][0]['reason'],
                original_reprojection_px=reprojection,original_rmse_m=rmse,joint_fit=joint,
                input_sync_spread_s=max(v.stamp_s for v in views)-min(v.stamp_s for v in views),
                improvement_is_not_pose_accuracy_or_insertion_success=True)
            rows.append(result)
            print(json.dumps(dict(run=row['run'],original_px=reprojection,joint_px=joint['mean_reprojection_error_px'],
                leave_one_view_out=joint['leave_one_view_out'])),flush=True)
    (out/'results.json').write_text(json.dumps(dict(scope='Cached DEV detections only',
        ground_truth_read=False,final_cases_used=0,policy_changed=False,rows=rows),indent=2))

if __name__=='__main__':main()
