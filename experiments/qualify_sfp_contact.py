"""Qualify an executed intentional lateral miss as an active SFP contact control."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.contact_contract import contract
from experiments.contact_depth import resolve_contract,touching,physical_contact

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-dir',type=Path,required=True);p.add_argument('--run',required=True)
    p.add_argument('--runtime-audit',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    spec,active,target,digest=contract(a.scene_dir);folder=a.scene_dir/a.run
    runtime=resolve_contract(spec,json.loads(a.runtime_audit.read_text()) if a.runtime_audit else None);convention=runtime['depth_convention']
    contacts=json.load(gzip.open(folder/'contacts.json.gz','rt'));audit=json.load(gzip.open(folder/'contact_audit_status.json.gz','rt'))
    policy=json.loads((folder/'policy.json').read_text());prefix=f"cable_{spec['reset']['target_index']}::sfp_"
    target_prefix=spec['reset']['target'].split('/')[0]+'::'
    pairs=[c for m in contacts for c in m['contacts'] if any(prefix in c[k]['name'] for k in ('collision1','collision2'))
        and any(target_prefix in c[k]['name'] for k in ('collision1','collision2'))]
    physical=[c for c in pairs if physical_contact(c,runtime)]
    ok=bool(physical and audit and min(x['observed_collision_components'] for x in audit)>=spec['minimum_expected_contact_components']
        and policy['reason']=='force_abort' and (spec.get('solver_iterations_requested') is None
            or all(s.get('solver_iterations_configured')==spec['solver_iterations_requested'] for s in audit)))
    proof={'qualified':ok,'purpose':'intentional_2mm_lateral_miss_not_insertion_success','source_run':a.run,
        'collision_contract_sha256':digest,'timestep_s':spec['timestep_s'],'requested_pair_count':len(pairs),
        'raw_cached_manifold_pair_count':len(pairs),'physical_pair_count':len(physical),'depth_convention':convention,
        'physics_contact_contract':runtime,
        'solver_iterations_requested':spec.get('solver_iterations_requested'),
        'pairs':sorted({(c['collision1']['name'],c['collision2']['name']) for c in pairs}),
        'peak_tared_force_N':policy['peak_tared_force_N'],'policy_stop':policy['reason'],
        'world_sha256':spec['world_sha256']}
    (a.out or ROOT/'.artifacts/sfp_contact_positive_control.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
    if not ok:raise RuntimeError('Contact positive control did not qualify')

if __name__=='__main__':main()
