"""Post-hoc timestamp join; transport arrival order is not simulation time.

Contact consistency requires live physics observation, bounded penetration and
no wrong target contact. SC requires terminal requested-target contact, proved
by signed geometric touch or an audited nonzero solver normal impulse. Cached
positive-gap manifolds without an applied impulse are insufficient. An SFP
clearance seat additionally needs a qualified collision positive control with
the identical collision contract and timestep, plus a source-geometry check.
All final pose samples are retained, including the post-command tail.
"""
import argparse
import bisect
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.verification import SeatingProfile,verify_seating
from experiments.contact_depth import penetration_m,touching,physical_contact,active_constraint_impulses,resolve_contract,same_engine_contract

def stamp(m):return m['header']['stamp']['sec']+m['header']['stamp']['nanosec']/1e9

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene-dir',type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--runtime-audit',type=Path)
    p.add_argument('--positive-control-proof',type=Path);p.add_argument('--output-dir',type=Path)
    a=p.parse_args();folder=a.scene_dir/a.run;out=a.output_dir or folder;out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((a.scene_dir/'scene.json').read_text());connector=spec['connector']
    runtime=resolve_contract(spec,json.loads(a.runtime_audit.read_text()) if a.runtime_audit else None)
    convention=runtime['depth_convention']
    profile_path=ROOT/'experiments/configs'/f'{connector}_seating_profile.json'
    profile=SeatingProfile(**json.loads(profile_path.read_text()))
    rows=json.load(gzip.open(folder/'truth_trace.json.gz','rt'))
    messages=sorted(json.load(gzip.open(folder/'contacts.json.gz','rt')),key=stamp)
    times=[stamp(m) for m in messages]
    body=f"cable_{spec['reset']['target_index']}::{'sc_plug_link' if connector=='sc' else 'sfp_module_link'}::"
    # Both SFP cages share one NIC collision link. Slot identity is established
    # by the independent tip pose in the requested slot frame, not its name.
    target=spec['reset']['target'].split('/')[0]+'::'
    def requested(c):return any(body in c[k]['name'] for k in ('collision1','collision2')) and any(target in c[k]['name'] for k in ('collision1','collision2'))
    if connector=='sfp':
        prefixes=[body,body.replace('sfp_module_link','sfp_tip_link')]
        def requested(c):return any(any(prefix in c[k]['name'] for prefix in prefixes) for k in ('collision1','collision2')) and any(target in c[k]['name'] for k in ('collision1','collision2'))
    terminal_start=rows[-1]['sim_time_s']-profile.dwell_s if rows else 0
    terminal_pairs=[c for m in messages if stamp(m)>=terminal_start for c in m['contacts'] if requested(c)]
    terminal_contact=any(physical_contact(c,runtime) for c in terminal_pairs)
    impulse_points=[point for c in terminal_pairs for point in active_constraint_impulses(c,runtime)]
    audit_path=folder/'contact_audit_status.json.gz'
    audit=sorted(json.load(gzip.open(audit_path,'rt')),key=lambda x:x['sim_time_s']) if audit_path.exists() else []
    audit_times=[x['sim_time_s'] for x in audit]
    clearance_basis=None;contact_adequate=terminal_contact
    if connector=='sfp' and not terminal_contact:
        from experiments.contact_contract import terminal_box_clearance
        final=[r for r in rows if r['sim_time_s']>=terminal_start]
        selected=[final[i] for i in sorted({0,len(final)//2,len(final)-1})] if final else []
        overlap,digest=terminal_box_clearance(a.scene_dir,selected)
        proof_path=a.positive_control_proof or a.scene_dir/'sfp_contact_positive_control.json'
        if not proof_path.exists():proof_path=ROOT/'.artifacts/sfp_contact_positive_control.json'
        proof=json.loads(proof_path.read_text()) if proof_path.exists() else {}
        contact_adequate=bool(selected and overlap<1e-7 and proof.get('qualified') and
            proof.get('depth_convention')==convention and proof.get('physical_pair_count',0)>0 and
            proof.get('collision_contract_sha256')==digest and proof.get('timestep_s')==spec['timestep_s']
            and proof.get('solver_iterations_requested')==spec.get('solver_iterations_requested')
            and same_engine_contract(proof.get('physics_contact_contract'),runtime))
        clearance_basis={'maximum_source_box_overlap_m':overlap,'collision_contract_sha256':digest,
            'positive_control_qualified':contact_adequate,'basis':'clearance at nominal seat; contact-enabled lateral miss independently qualified'}
    maximum_target=0.;maximum_all=0.;output=[]
    for row in rows:
        t=row['sim_time_s'];lo=bisect.bisect_left(times,t-.025);hi=bisect.bisect_right(times,t+.025)
        near=[c for m in messages[lo:hi] for c in m['contacts']]
        target_pen=max([0.]+[penetration_m(d,convention) for c in near if requested(c) for d in c['depths']])
        all_pen=max([0.]+[penetration_m(d,convention) for c in near for d in c['depths']])
        maximum_target=max(maximum_target,target_pen);maximum_all=max(maximum_all,all_pen)
        wrong=any(physical_contact(c,runtime) and any(body in c[k]['name'] for k in ('collision1','collision2')) and
            any(('sc_port_' in c[k]['name'] or 'nic_card_mount_' in c[k]['name']) and target not in c[k]['name'] for k in ('collision1','collision2')) for c in near)
        health=True;nearest=None
        if 'contact_audit_source' in spec:
            j=bisect.bisect_left(audit_times,t)
            nearest=min(audit[max(0,j-1):j+1],key=lambda x:abs(x['sim_time_s']-t),default=None)
            health=bool(nearest and abs(nearest['sim_time_s']-t)<=.025 and
                nearest['observed_collision_components']>=spec['minimum_expected_contact_components'] and
                nearest['detachable_joint_components']==(2 if spec['remote_endpoint_support'].startswith('prescribed') else 1))
        if spec.get('solver_iterations_requested') is not None:
            health=health and nearest is not None and nearest.get('solver_iterations_configured')==spec['solver_iterations_requested']
        output.append({**row,'max_penetration_m':all_pen,'contact_consistent':contact_adequate and hi>lo and not wrong and health,
            'physics_contact_heartbeat_valid':health,
            'requested_contact_timestamp_joined':any(requested(c) and physical_contact(c,runtime) for c in near),'contact_stream_covered':hi>lo})
    result=verify_seating(output,profile)
    policy=json.loads((folder/'policy.json').read_text())
    if policy['reason']!='command_and_dwell_completed':result={'verified_success':False,'reason':policy['reason'],'geometry_verifier':result}
    if spec.get('high_rate_wrist_audit_required'):
        force_audit=json.loads((folder/'wrench_analysis.json').read_text()).get('high_rate_wrist_audit',{})
        if force_audit.get('force_ceiling_N',8.)!=spec.get('evaluation_force_ceiling_N',8.):
            force_audit={**force_audit,'qualified':False,'reason':'force_protocol_mismatch'}
        if not force_audit.get('qualified'):
            result={'verified_success':False,'reason':'high_rate_force_audit:'+force_audit.get('reason','missing'),
                'geometry_and_policy_verifier':result}
        result['high_rate_wrist_audit']=force_audit
    terminal=[r for r in output if r['sim_time_s']>=terminal_start]
    metrics={key:{'min':min(r[key] for r in terminal),'max':max(r[key] for r in terminal)} for key in
        ('depth_m','lateral_error_m','angular_error_deg','max_penetration_m')} if terminal else {}
    result.update(run=a.run,connector=connector,terminal_metrics=metrics,requested_contact_during_terminal_dwell=terminal_contact,
        max_requested_target_penetration_m=maximum_target,max_global_penetration_m=maximum_all,
        wall_s=policy['wall_s'],sim_s=policy['sim_s'],peak_policy_tared_force_N=policy['peak_tared_force_N'],
        profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        contact_consistency_definition=__doc__,world_sha256=spec['world_sha256'])
    result['clearance_seat_contact_basis']=clearance_basis
    result['physics_contact_contract']=runtime
    result['verification_semantics_revision']=3
    result['terminal_requested_contact_evidence']={
        'pair_count':len(terminal_pairs),'signed_touch_pair_count':sum(touching(c,convention) for c in terminal_pairs),
        'active_impulse_point_count':len(impulse_points),
        'active_impulse_distance_range_m':[min(p['raw_signed_distance_m'] for p in impulse_points),max(p['raw_signed_distance_m'] for p in impulse_points)] if impulse_points else None,
        'maximum_normal_impulse_Ns':max((p['normal_impulse_Ns'] for p in impulse_points),default=None)}
    result['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if result.get('verified_success') and clearance_basis and not terminal_contact:
        result['reason']='stable_geometric_clearance_seat_collision_control_validated'
    result['terminal_violation_counts']={
        'depth':sum(abs(r['depth_m']-profile.seated_depth_m)>profile.depth_tolerance_m for r in terminal),
        'lateral':sum(r['lateral_error_m']>profile.lateral_tolerance_m for r in terminal),
        'angular':sum(r['angular_error_deg']>profile.angular_tolerance_deg for r in terminal),
        'penetration':sum(r['max_penetration_m']>profile.max_penetration_m for r in terminal),
        'contact_observation':sum(not r['contact_consistent'] for r in terminal)}
    (out/'verified.json').write_text(json.dumps(result,indent=2)+'\n')
    with gzip.open(out/'verification_trace.json.gz','wt') as f:json.dump(output,f)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
