"""Post-hoc tracking/grasp audit; none of these truth values reach the policy."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--verification-dir',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();policy=json.loads((a.run_dir/'policy.json').read_text())
    truth=json.load(gzip.open((a.verification_dir or a.run_dir)/'verification_trace.json.gz','rt'))
    rows=policy['trace'];end=rows[-1]['sim_s'];terminal=[r for r in rows if r['sim_s']>=end-1]
    frame=np.array(policy['fixture_orientation_prior'])
    tracking=np.array([frame.T@(np.array(r['desired_tip'])-r['actual_tip']) for r in terminal])
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    recorded={round(r['sim_time_s'],6):r for r in truth};position=[];angles=[]
    for row in rows:
        actual=recorded.get(round(row['sim_s'],6))
        if actual is None:continue
        gt=np.linalg.inv(base)@np.array(actual['tip_world'])
        position.append(np.array(row['actual_tip'])-gt[:3,3])
        estimated_R=np.array(row['actual_tip_rotation']) if 'actual_tip_rotation' in row else frame@Rotation.from_euler('z',row['yaw_relative_rad']).as_matrix()
        # yaw-only reconstruction is not a full orientation observation.
        angles.append(float(np.linalg.norm(Rotation.from_matrix(estimated_R.T@gt[:3,:3]).as_rotvec())*180/np.pi))
    position=np.array(position)
    result={'purpose':'evaluator_only_tracking_diagnostic','run':policy['run'],
        'terminal_own_tracking_error_tip_axes_mean_mm':(tracking.mean(axis=0)*1000).tolist(),
        'terminal_own_tracking_error_tip_axes_max_abs_mm':(abs(tracking).max(axis=0)*1000).tolist(),
        'exact_stamp_position_matches':len(position),
        'fixed_grasp_tip_position_residual_median_mm':(np.median(position,axis=0)*1000).tolist() if len(position) else None,
        'fixed_grasp_tip_position_residual_max_norm_mm':float(np.linalg.norm(position,axis=1).max()*1000) if len(position) else None,
        'tip_orientation_residual_median_deg':float(np.median(angles)) if angles else None,
        'orientation_caveat':'Full measured tip rotation is used when present; older yaw-only traces omit roll/pitch and cannot certify grasp rotation.',
        'full_orientation_recorded':all('actual_tip_rotation' in row for row in rows)}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
