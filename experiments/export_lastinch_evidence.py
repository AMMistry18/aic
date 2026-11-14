"""Copy selected, unmodified measured artifacts into the reviewable evidence set."""
import hashlib
import json
from pathlib import Path
import shutil
import gzip

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'experiments/evidence/lastinch'
    out.mkdir(parents=True,exist_ok=True)
    manifest={}
    for label,run in (('sc','lastinch_sc_final'),('sfp','lastinch_sfp_v1')):
        source=ROOT/'.artifacts'/run
        dest=out/label;dest.mkdir(exist_ok=True)
        for name in ('scene.json','validation.json','capture.json'):
            shutil.copy2(source/name,dest/name)
        for camera in ('overview','target','center_camera'):
            shutil.copy2(source/'cameras'/camera/'002.png',dest/f'{camera}.png')
    shutil.copy2(ROOT/'.artifacts/lastinch_matrix/geometry_preflight.json',out/'geometry_preflight.json')
    container=out/'container';container.mkdir(exist_ok=True)
    for name in ('model_probe.log','eval_probe.log','source_tests.log'):
        shutil.copy2(ROOT/'.artifacts/container'/name,container/name)
    runtime=ROOT/'.artifacts/kilted_handoff'
    if (runtime/'controller_bootstrap.json').exists():
        dest=out/'kilted';dest.mkdir(exist_ok=True)
        for name in ('scene.json','controller_bootstrap.json','observed_models.json',
                     'observed_reset_audit.json','sc_policy_pilot_final.json'):
            if (runtime/name).exists():shutil.copy2(runtime/name,dest/name)
        for name in ('controller_hold_calibrated.json','port_contact_probe.json',
                     'observed_scene.pbtxt','world.sdf','cable_transforms.json'):
            if (runtime/name).exists():
                with gzip.open(dest/(name+'.gz'),'wb') as stream:stream.write((runtime/name).read_bytes())
        if (runtime/'controller_hold_calibrated.json').exists():
            hold=json.loads((runtime/'controller_hold_calibrated.json').read_text())
            (dest/'hold_summary.json').write_text(json.dumps({k:v for k,v in hold.items() if k not in ('samples','forces')},indent=2)+'\n')
        if (runtime/'port_contact_probe.json').exists():
            probe=json.loads((runtime/'port_contact_probe.json').read_text())
            pairs=[c for m in probe['contacts'] for c in m.get('contacts',[])]
            summary={k:v for k,v in probe.items() if k not in ('contacts','trace')}
            summary['contact_message_count']=len(probe['contacts'])
            summary['gripper_contact_pairs']=sum(any('gripper/' in c.get(k,{}).get('name','') for k in ('collision1','collision2')) for c in pairs)
            summary['port_contact_pairs']=sum(any('sc_port_2' in c.get(k,{}).get('name','') for k in ('collision1','collision2')) for c in pairs)
            (dest/'contact_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        for camera in ('overview','target','center_camera'):
            if (runtime/'cameras'/f'{camera}.png').exists():shutil.copy2(runtime/'cameras'/f'{camera}.png',dest/f'{camera}.png')
        shutil.copy2(ROOT/'.artifacts/container/checkpoints.json',container/'checkpoints.json')
    for path in out.rglob('*'):
        if path.is_file() and path.name!='manifest.json':
            manifest[str(path.relative_to(out))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)


if __name__=='__main__': main()
