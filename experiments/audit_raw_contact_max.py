"""Cross-check exported contact maxima against every raw signed-distance packet."""
import argparse
import gzip
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    exported=json.loads((a.evidence/'results.json').read_text());rows=[]
    for row in exported['trials']:
        if row['status']!='completed':continue
        info=json.loads((a.evidence/row['run']/'trial_index.json').read_text())['trial']
        if info['physics_contact_contract']['depth_convention']!='signed_distance_negative_penetration':
            raise ValueError('This independent audit is specific to the audited Bullet signed-distance binary')
        raw=ROOT/info['evidence_dir']/row['run']/'contacts.json.gz'
        contacts=json.load(gzip.open(raw,'rt'))
        actual=max([0.]+[-float(d) for m in contacts for c in m['contacts'] for d in c['depths']])
        stored=info['max_global_penetration_m']
        rows.append({'run':row['run'],'all_raw_packets_max_penetration_m':actual,'timestamp_joined_max_m':stored,
            'raw_minus_joined_m':actual-stored,'recorded_whole_trial_0p1mm_pass':actual<=.0001})
    output={'purpose':'independent_packet_scope_crosscheck_not_engine_physical_validation','runs':rows,
        'larger_raw_maximum_runs':sum(r['raw_minus_joined_m']>1e-9 for r in rows)}
    a.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
