"""Post-hoc tracking audit; evaluator truth is never passed to a policy.

Reconstruct the robot TCP from the rigid reset attachment and recorded actual
tip pose, then compare with issued TCP position commands at matching times.
The source robot base placement is declared in prepare_kilted_handoff.py.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result={'purpose':'post_hoc_diagnostic_only','production_policy_inputs':False,'runs':[]}
    base_rotation=Rotation.from_euler('z',-np.pi).as_matrix();base_translation=np.array([-.2,.2,0.])
    for run in a.runs:
        archive=ROOT/'.artifacts/lastinch_runs'/run;path=archive/run
        scene=json.loads((archive/'scene.json').read_text());policy=json.loads((path/'policy.json').read_text())
        trace=json.load(gzip.open(path/'verification_trace.json.gz','rt'))
        stamps=np.array([r['sim_time_s'] for r in trace]);grasp=np.asarray(scene['reset']['tip_in_tcp_reset'])
        port_axes=base_rotation.T@np.asarray(scene['reset']['mouth_world'])[:3,:3]
        contact_start=(policy.get('contact_grasp_registration') or {}).get('start_sim_s',float('inf'))
        rows=[]
        for command in policy['trace']:
            t=command['sim_s']
            if command['target_tcp'] is None or t>=contact_start:continue
            index=np.searchsorted(stamps,t)
            index=min([max(0,index-1),min(len(stamps)-1,index)],key=lambda k:abs(stamps[k]-t))
            observed=trace[index]
            if abs(stamps[index]-t)>.015 or observed['depth_m']<=0:continue
            tip=np.asarray(observed['tip_world']);world_tcp_R=tip[:3,:3]@grasp[:3,:3].T
            tcp_base=base_rotation.T@(tip[:3,3]-world_tcp_R@grasp[:3,3]-base_translation)
            error=port_axes.T@(np.asarray(command['target_tcp'])-tcp_base)
            rows.append({'command_sim_s':t,'truth_sim_s':float(stamps[index]),
                         'tcp_command_error_port_m':error.tolist(),
                         'actual_depth_m':observed['depth_m'],'actual_lateral_error_m':observed['lateral_error_m']})
        lateral=np.array([np.linalg.norm(r['tcp_command_error_port_m'][:2]) for r in rows])
        inputs=[archive/'scene.json',path/'policy.json',path/'verification_trace.json.gz']
        result['runs'].append({'run':run,'window':'actual depth >0; before first contact registration',
            'samples':len(rows),'median_lateral_tcp_tracking_error_m':float(np.median(lateral)),
            'max_lateral_tcp_tracking_error_m':float(lateral.max()),'trace':rows,
            'input_sha256':{str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict((k,v) for k,v in r.items() if k not in ['trace','input_sha256']) for r in result['runs']]))


if __name__=='__main__':main()
