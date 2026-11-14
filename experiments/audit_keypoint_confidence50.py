"""Diagnostic score-versus-pixel-error audit on old static DEVELOPMENT data.

Truth is used exclusively for post-inference error analysis, never inference,
weight selection or runtime policy inputs. Scores are not assumed to be
calibrated coordinate probabilities; this audit does not change any weights.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M


def main():
    cache=json.loads((ROOT/'experiments/parallel_dev46/angular_inputs.json').read_text());rows=[]
    for case in cache['rows']:
        local=SC_PLUG_LOCAL_KEYPOINTS_M if case['connector']=='sc' else SFP_PLUG_LOCAL_KEYPOINTS_M
        truth=np.array(case['scoring_only']['tip_world']);world=(truth[:3,:3]@local.T).T+truth[:3,3]
        views={v['camera']:v for v in case['inputs']}
        for detection in case['detections']:
            v=views[detection['camera_name']];P=np.array(v['K'])@np.linalg.inv(v['T_world_from_camera'])[:3]
            q=(P@np.c_[world,np.ones(len(world))].T).T;expected=q[:,:2]/q[:,2:]
            error=np.linalg.norm(np.array(detection['keypoints_px'])-expected,axis=1)
            for index,(score,value) in enumerate(zip(detection['keypoint_confidences'],error)):
                rows.append(dict(connector=case['connector'],case=case['case_id'],camera=detection['camera_name'],
                    keypoint=index,score=float(score),scoring_only_error_px=float(value)))
    summary={}
    for kind in ['sc','sfp']:
        selected=[r for r in rows if r['connector']==kind];high=[r for r in selected if r['score']>=.99]
        summary[kind]=dict(points=len(selected),score_ge_099=len(high),
            mean_score=float(np.mean([r['score'] for r in selected])),
            high_score_median_error_px=float(np.median([r['scoring_only_error_px'] for r in high])) if high else None,
            high_score_p95_error_px=float(np.percentile([r['scoring_only_error_px'] for r in high],95)) if high else None,
            high_score_over_2px=sum(r['scoring_only_error_px']>2 for r in high))
    result=dict(scope='Post-inference projected-label diagnostic on eight reused static TRAINING/development scenes, not held-out accuracy',
        final_cases_used=0,policy_weights_changed=False,truth_used_for_runtime_or_fitting=False,
        caveat='Network keypoint score is not a calibrated probability of subpixel accuracy; high-score errors here do not establish population miscalibration.',
        summary=summary,rows=rows)
    out=ROOT/'experiments/angular_development50';out.mkdir(exist_ok=True)
    (out/'keypoint_score_errors.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
