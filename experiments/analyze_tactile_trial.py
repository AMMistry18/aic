"""Evaluator-only phase/contact audit; truth is never fed to the controller."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from contact_depth import resolve_contract,penetration_m,touching

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-dir',type=Path,required=True);p.add_argument('--run',required=True)
    p.add_argument('--runtime-audit',type=Path);p.add_argument('--verification-dir',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    folder=a.scene_dir/a.run;spec=json.loads((a.scene_dir/'scene.json').read_text());policy=json.loads((folder/'policy.json').read_text())
    runtime=resolve_contract(spec,json.loads(a.runtime_audit.read_text()) if a.runtime_audit else None)
    convention=runtime['depth_convention']
    truth=json.load(gzip.open((a.verification_dir or folder)/'verification_trace.json.gz','rt'));contact=json.load(gzip.open(folder/'contacts.json.gz','rt'))
    trace=policy['trace'];segments=[]
    for row in trace:
        if not segments or row['phase']!=segments[-1]['phase']:segments.append({'phase':row['phase'],'start_s':row['sim_s']})
    mouth=np.array(spec['reset']['mouth_world']);body=f"cable_{spec['reset']['target_index']}::";target=spec['reset']['target'].split('/')[0]+'::'
    held=('sc_plug_link','sc_tip_link') if spec['connector']=='sc' else ('sfp_module_link','sfp_tip_link')
    for i,segment in enumerate(segments):
        start=segment['start_s'];end=segments[i+1]['start_s'] if i+1<len(segments) else trace[-1]['sim_s']
        selected=[r for r in truth if max(start,end-.2)<=r['sim_time_s']<=end]
        pairs=[]
        for message in contact:
            stamp=message['header']['stamp'];t=stamp['sec']+stamp['nanosec']/1e9
            if start<=t<end:pairs.extend(message['contacts'])
        physical=[c for c in pairs if touching(c,convention)]
        requested=[c for c in physical if any(target in c[k]['name'] for k in ('collision1','collision2')) and
            any(any(body+link+'::' in c[k]['name'] for link in held) for k in ('collision1','collision2'))]
        if selected:
            offsets=np.array([(np.linalg.inv(mouth)@np.array(r['tip_world']))[:3,3] for r in selected])
            segment.update(last_0p2s_tip_offset_mean_mm=(offsets.mean(axis=0)*1000).tolist(),
                last_0p2s_lateral_max_mm=max(r['lateral_error_m'] for r in selected)*1000,
                last_0p2s_angle_max_deg=max(r['angular_error_deg'] for r in selected))
        worst=max(physical,key=lambda c:max(penetration_m(d,convention) for d in c['depths']),default=None)
        segment.update(end_s=end,requested_contact_pairs=len(requested),physical_contact_pairs=len(physical),raw_cached_contact_pairs=len(pairs),
            requested_max_penetration_mm=max([0.]+[penetration_m(d,convention)*1000 for c in requested for d in c['depths']]),
            global_max_penetration_mm=max([0.]+[penetration_m(d,convention)*1000 for c in physical for d in c['depths']]),
            worst_pair_links=[worst[k]['name'].rsplit('::',1)[0] for k in ('collision1','collision2')] if worst else None)
    output={'purpose':'posthoc_evaluator_only','contact_contract':runtime,'run':a.run,'policy_reason':policy['reason'],'phases':segments}
    (a.out or folder/'tactile_phase_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
