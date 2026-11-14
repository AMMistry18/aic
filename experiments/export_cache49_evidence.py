"""Preserve compact actual runtime qualification evidence outside .artifacts."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1];DEV=ROOT/'.artifacts/dev46_workspace'
OUT=ROOT/'experiments/evidence/cache49'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    status=read(ROOT/'experiments/runtime_qualification47_v4/progress.json')
    if status['status']!='complete_requires_review':raise RuntimeError('Qualification is incomplete')
    freeze=read(ROOT/'experiments/full_evaluation44/source_freeze.json')['source_sha256']
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in freeze.items()):raise RuntimeError('Final source changed')
    OUT.mkdir(parents=True,exist_ok=True);rows=[];hashes={}
    def preserve(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('Existing evidence differs')
        else:shutil.copy2(source,target)
        hashes[str(target.relative_to(OUT)).replace('\\','/')]=digest
    for row in status['results']:
        if not row.get('verified_success'):raise RuntimeError('Keep failure separately, do not export it as qualification')
        scene=DEV/row['evidence_dir'].replace('\\','/');folder=scene/row['run'];kind=row['case']['connector'];target=OUT/kind
        for name in ['verified.json','policy.json','evaluation.json','wrench_analysis.json','wrench_raw.json.gz',
                     'wrench.json.gz','truth_trace.json.gz','verification_trace.json.gz','contacts.json.gz','contact_audit_status.json.gz']:
            preserve(folder/name,target/name)
        for name in ['scene.json','controller_bootstrap.json','controller_hold_calibrated.json','static_load_calibration.json','world.sdf','robot.urdf']:
            preserve(scene/name,target/name)
        preserve(DEV/'.artifacts/container'/(row['run']+'_policy.log'),target/'policy.log')
        for label in ['perception_before','perception_after']:
            for path in (folder/label).glob('*'):
                if path.is_file():preserve(path,target/label/path.name)
        frames=list(folder.glob('target_*.jpg'))
        if frames:
            first=min(frames,key=lambda p:float(p.stem.split('_')[-1]));last=max(frames,key=lambda p:float(p.stem.split('_')[-1]))
            preserve(first,target/'target_initial.jpg');preserve(last,target/'target_final.jpg')
        p=read(folder/'policy.json');v=read(folder/'verified.json');geometry=read(scene/'scene.json')
        rows.append(dict(connector=kind,run=row['run'],verified_success=True,reason=v['reason'],
            terminal_metrics=v['terminal_metrics'],raw_peak_N=v['high_rate_wrist_audit']['peak_tared_force_N'],
            complete_2000Hz_coverage=v['high_rate_wrist_audit']['complete_action_coverage'],
            action_sim_s=p['sim_s'],record_wall_s=p['wall_s'],configuration_wall_s=p.get('configuration_wall_s'),
            actual_scene_counts=geometry['counts'],physics_contract=geometry['physics_contact_contract']))
    preserve(ROOT/'experiments/candidates/efficiency49/sc_rim_refinement.py',OUT/'candidate_sc_rim_refinement.py')
    preserve(ROOT/'experiments/rim_followup49_v2/progress.json',OUT/'paired_comparison.json')
    (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    result=dict(scope='One reused SC image/CAD-cache development case; not randomized reliability or angular recovery',
        final_sources_unchanged=True,final_cases_used_for_tuning=0,rows=rows,
        SC_image_CAD_cache_included=True,axial_unload_proposal_not_installed=True,
        wall_comparison_note='Shared-laptop / runtime configuration differences preclude attributing all historical timing differences to GPU alone')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(exported_files=len(hashes),rows=rows)))


if __name__=='__main__':main()
