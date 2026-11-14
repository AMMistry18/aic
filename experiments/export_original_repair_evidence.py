"""Export explicit completed original-policy development runs and checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def copy(source,destination):
    if source.is_file():
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',type=Path,default=ROOT/'experiments/evidence/original_repair')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    ledger=[]
    for name in a.runs:
        archive=ROOT/'.artifacts/lastinch_runs'/name;run=archive/name
        policy=json.loads((run/'policy.json').read_text());verified=json.loads((run/'verified.json').read_text())
        if json.loads((run/'status.json').read_text())['active']:raise RuntimeError('Only completed runs can be exported')
        destination=out/'runs'/name
        for file in ['policy.json','verified.json','evaluation.json','wrench_analysis.json','truth_trace.json.gz',
                     'verification_trace.json.gz','wrench.json.gz','wrench_raw.json.gz','contacts.json.gz','contact_audit_status.json.gz','status.json']:
            copy(run/file,destination/file)
        for file in ['scene.json','controller_bootstrap.json','controller_hold_calibrated.json','static_load_calibration.json','physics_process_audit.json']:
            copy(archive/file,destination/'environment'/file)
        (destination/'environment'/'README.md').write_text(
            'The scene.json backend flags were authored before runtime controller validation. '
            'They are preserved historical setup metadata, not the final controller verdict. '
            'Use controller_bootstrap.json (active native AIC controller/broadcasters), '
            'controller_hold_calibrated.json and physics_process_audit.json for executed '
            'backend/hold evidence. Simulator object poses in these setup/evaluation files '
            'are not production-policy observations.\n',encoding='utf-8')
        for capture in ['perception_before','perception_after','rim_inputs','body_inputs']:
            if (run/capture).exists():shutil.copytree(run/capture,destination/capture,dirs_exist_ok=True)
        for camera in ['target','board','overview']:
            images=sorted(run.glob(f'{camera}_*.jpg'),key=lambda path:float(path.stem.split('_')[-1]))
            if images:
                for image in {images[0],images[len(images)//2],images[-1]}:copy(image,destination/'sampled_cameras'/image.name)
        for kind in ['policy','server','hold','recorder','verify','wrench']:
            copy(ROOT/f'.artifacts/container/{name}_{kind}.log',destination/'logs'/f'{kind}.log')
        sources={str(f.relative_to(archive/'source_snapshot')).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in (archive/'source_snapshot').rglob('*') if f.is_file()}
        (destination/'source_sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
        # Retain executable version-specific code without duplicating large
        # model binaries. Every checkpoint's exact hash remains in the ledger.
        with zipfile.ZipFile(destination/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for file in (archive/'source_snapshot').rglob('*'):
                if file.is_file() and file.suffix!='.pt':
                    bundle.write(file,str(file.relative_to(archive/'source_snapshot')).replace('\\','/'))
        ledger.append({k:policy.get(k) for k in ['run','connector','method','harness_revision','reason','command_count',
            'visual_grasp_available','wall_s','sim_s','controller_tare','visual_reacquisition_events',
            'rim_refinement_enabled','contact_grasp_registration','contact_entry_search',
            'orientation_condition','native_wrench_feedback_gain','body_translation_refinement_enabled',
            'sc_body_diagnostic','lateral_tracking_compensation_enabled','peak_tared_force_N']}|
            {'verified_success':verified['verified_success'],'max_global_penetration_m':verified['max_global_penetration_m'],
             'tracking_peak_compensation_N':(policy.get('lateral_tracking_compensation') or {}).get('peak_compensation_N'),
             'angular_screen_diagnostic_ablation':policy.get('angular_screen_diagnostic_ablation',False),
             'counts_as_guarded_production_policy':policy.get('counts_as_guarded_production_policy',True),
             'terminal_metrics':verified['terminal_metrics']})
    (out/'executed_trials.json').write_text(json.dumps(ledger,indent=2)+'\n')
    mutable={'persistent_screen32v1.json','persistent_screen32v1.csv'}
    checksums={str(f.relative_to(out)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
               for f in out.rglob('*') if f.is_file() and f.name!='SHA256SUMS.json'
               and str(f.relative_to(out)).replace('\\','/') not in mutable}
    (out/'SHA256SUMS.json').write_text(json.dumps(checksums,indent=2)+'\n')
    print(json.dumps({'runs':len(ledger),'checksummed_files':len(checksums)}))


if __name__=='__main__':main()
