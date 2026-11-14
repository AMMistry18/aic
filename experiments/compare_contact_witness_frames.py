"""Diagnostic cross-check of ECM world poses against separate PosePublisher TF.

Reports nearest stream timestamp offsets explicitly; does not change any
seating, penetration or synchronization gate. Supports a flushed live prefix.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np
from scipy.spatial.transform import Rotation


def prefix(path):
    raw=path.read_bytes();data=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(raw).decode()
    lines=data.splitlines()
    if not data.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines],hashlib.sha256(raw).hexdigest()


def transform(value):
    p=value['translation'];q=value['rotation'];T=np.eye(4)
    T[:3,3]=[p[k] for k in ('x','y','z')]
    T[:3,:3]=Rotation.from_quat([q[k] for k in ('x','y','z','w')]).as_matrix();return T


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    poses,pose_hash=prefix(a.capture/'poses.jsonl.gz');health,health_hash=prefix(a.capture/'audit.jsonl.gz')
    frames=[];times=[]
    for row in poses:
        stamp=row['transforms'][0]['header']['stamp'];times.append(stamp['sec']+stamp['nanosec']/1e9)
        local={t['child_frame_id']:transform(t['transform']) for t in row['transforms']}
        root=next(t['child_frame_id'] for t in row['transforms'] if '/' not in t['child_frame_id'])
        frames.append({name:local[root]@T for name,T in local.items() if name.startswith(root+'/')})
    offsets=[];residuals=[];angles=[];unmatched=0;worst=None
    for row in health:
        t=row['sim_time_s'];i=bisect.bisect_left(times,t)
        choices=[k for k in (i-1,i) if 0<=k<len(times)]
        if not choices:unmatched+=1;continue
        k=min(choices,key=lambda j:abs(times[j]-t));delta=times[k]-t
        if abs(delta)>.00101:unmatched+=1;continue
        offsets.append(delta)
        for name,value in row['cable_link_world_poses'].items():
            T=frames[k][name];position=float(np.linalg.norm(T[:3,3]-value[:3])*1000)
            angle=float(np.linalg.norm(Rotation.from_matrix(T[:3,:3].T@Rotation.from_quat(value[3:]).as_matrix()).as_rotvec())*180/np.pi)
            residuals.append(position);angles.append(angle)
            if worst is None or position>worst['position_difference_mm']:
                worst={'observer_sim_s':t,'pose_sim_s':times[k],'link':name,'position_difference_mm':position,'angle_difference_deg':angle}
    if not residuals:raise ValueError('No nearby pose-stream observations')
    metadata=json.loads((a.capture/'capture.json').read_text())
    result={'purpose':'diagnostic_frame_crosscheck_not_exact_stamp_geometry_validation','run':metadata['run'],
        'coverage':'complete capture' if metadata.get('stop_reason') else 'flushed live prefix',
        'nearby_pose_packets':len(offsets),'unmatched_audit_packets':unmatched,
        'stream_stamp_offset_s_min_max':[min(offsets),max(offsets)],
        'link_position_difference_mm_median_p95_max':np.percentile(residuals,[50,95,100]).tolist(),
        'link_angle_difference_deg_median_p95_max':np.percentile(angles,[50,95,100]).tolist(),'worst':worst,
        'caveat':'Nearby stamps differ by up to one physics step; residual includes real motion. This is a coordinate-frame diagnostic, not a passed contact or physical-fidelity test.',
        'input_compressed_prefix_sha256':{'poses':pose_hash,'audit':health_hash},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
