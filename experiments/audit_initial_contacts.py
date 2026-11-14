"""Post-hoc pre-command overlap screen, retaining the original trial verdict.

This short recorded window cannot replace the new full setup hold gate. An
observed gross overlap invalidates a policy comparison, not the raw execution.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from contact_depth import resolve_contract,penetration_m
ROOT=Path(__file__).resolve().parents[1]

def stamp(m):return m['header']['stamp']['sec']+m['header']['stamp']['nanosec']/1e9

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--runtime-audit',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.index.read_text());audit=json.loads(a.runtime_audit.read_text()) if a.runtime_audit else None
    data['initial_contact_screen']={'source_index':str(a.index),'source_sha256':hashlib.sha256(a.index.read_bytes()).hexdigest(),
        'window_before_first_command_s':.1,'maximum_penetration_m':.0001,
        'caveat':'Post-hoc screen of recorded pre-command samples; passing does not certify a full 0.5 s setup hold.'}
    for r in data['results']:
        if r['status']!='completed':continue
        scene=ROOT/r['evidence_dir'];folder=scene/r['run'];policy=json.loads((folder/'policy.json').read_text())
        if not policy.get('trace'):continue
        t=policy['trace'][0]['sim_s'];spec=json.loads((scene/'scene.json').read_text())
        convention=resolve_contract(spec,audit)['depth_convention']
        messages=[m for m in json.load(gzip.open(folder/'contacts.json.gz','rt')) if t-.1<=stamp(m)<=t]
        maximum=0.;worst=None;bad=0
        for m in messages:
            values=[(penetration_m(d,convention),c) for c in m['contacts'] for d in c['depths']]
            value,c=max(values,key=lambda item:item[0]) if values else (0.,None)
            if value>.0001:bad+=1
            if value>maximum:
                maximum=value;worst=[c[k]['name'].rsplit('::',1)[0] for k in ('collision1','collision2')]
        result={'window_s':.1,'packet_count':len(messages),'packets_above_limit':bad,
            'maximum_penetration_m':maximum,'worst_links':worst,'complete_setup_hold_certified':False}
        r['initial_contact_screen']=result
        # Positive evidence of invalid setup is sufficient to flag; missing
        # packets remain explicitly unknown rather than silently passing.
        r['eligible_for_policy_comparison']=False if bad else None if not messages else True
        print(json.dumps({'run':r['run'],'eligible':r['eligible_for_policy_comparison'],**result}))
    a.out.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
