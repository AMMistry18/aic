"""Evaluator-only link-level contact and independent FT peak attribution.

A concurrent contact is a diagnostic association, not a causal attribution of
the wrist impulse. Raw plugin contact-wrench fields are deliberately unused.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from contact_depth import resolve_contract,penetration_m,touching

def stamp(m):return m['header']['stamp']['sec']+m['header']['stamp']['nanosec']/1e9

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene-dir',type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--runtime-audit',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();folder=a.scene_dir/a.run;spec=json.loads((a.scene_dir/'scene.json').read_text())
    runtime=resolve_contract(spec,json.loads(a.runtime_audit.read_text()) if a.runtime_audit else None)
    convention=runtime['depth_convention'];wrench=json.load(gzip.open(folder/'wrench.json.gz','rt'))
    analysis=json.loads((folder/'wrench_analysis.json').read_text());policy=json.loads((folder/'policy.json').read_text())
    first=policy['trace'][0]['sim_s'] if policy.get('trace') else stamp(wrench[0])
    action=[m for m in wrench if stamp(m)>=first]
    baseline=np.array(analysis['baseline_N'])
    peak=max(action,key=lambda m:np.linalg.norm(np.array([m['wrench']['force'][k] for k in ('x','y','z')])-baseline))
    peak_stamp=stamp(peak);groups=defaultdict(lambda:{'physical_pairs':0,'max_penetration_mm':0.,'worst_sim_s':None})
    simultaneous=[]
    for m in json.load(gzip.open(folder/'contacts.json.gz','rt')):
        t=stamp(m)
        for c in m['contacts']:
            if not touching(c,convention):continue
            pair=tuple(sorted(c[k]['name'].rsplit('::',1)[0] for k in ('collision1','collision2')))
            pen=max(penetration_m(d,convention) for d in c['depths'])*1000
            g=groups[pair];g['physical_pairs']+=1
            if pen>g['max_penetration_mm']:g.update(max_penetration_mm=pen,worst_sim_s=t)
            if abs(t-peak_stamp)<=.02:simultaneous.append({'sim_s':t,'links':pair,'penetration_mm':pen})
    truth=json.load(gzip.open(folder/'truth_trace.json.gz','rt'))
    near=min(truth,key=lambda r:abs(r['sim_time_s']-peak_stamp)) if truth else None
    result={'purpose':'posthoc_evaluator_only','run':a.run,'contact_contract':runtime,
        'peak_independent_tared_force_N':analysis['peak_independent_tared_force_N'],'peak_force_sim_s':peak_stamp,
        'nearest_truth_at_peak':{k:near[k] for k in ('sim_time_s','depth_m','lateral_error_m','angular_error_deg')} if near else None,
        'link_contacts_by_penetration':[{'links':pair,**g} for pair,g in sorted(groups.items(),key=lambda item:-item[1]['max_penetration_mm'])],
        'physical_contacts_within_20ms_of_force_peak':simultaneous,
        'caveat':'Concurrent contact does not prove which body caused the wrist impulse; shape-level names and plugin wrench values are not used.'}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('physical_contacts_within_20ms_of_force_peak','contact_contract')},indent=2))

if __name__=='__main__':main()
