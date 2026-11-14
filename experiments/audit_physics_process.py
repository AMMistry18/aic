"""Read-only /proc proof that the actual simulator loaded the selected backend."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path('/research/.artifacts/kilted_handoff')
    spec=json.loads((root/'scene.json').read_text());contract=spec['physics_contact_contract']
    expected=Path(contract['plugin_path']).resolve();candidates=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:lines=(process/'maps').read_text().splitlines()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        paths={Path(line.split(None,5)[5]).resolve() for line in lines if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
        if expected in paths:candidates.append((process,paths))
    if len(candidates)!=1:raise RuntimeError(f'Expected one simulator mapping selected plugin, found {len(candidates)}')
    process,paths=candidates[0];selected={p for p in paths if 'Bullet' in p.name or 'LinearMath' in p.name or p==expected}
    is_double=contract['scalar_precision_bits']==64
    bullet={p for p in selected if 'Bullet' in p.name or 'LinearMath' in p.name}
    dependencies={p for p in bullet if p!=expected}
    if len(dependencies)<3 or any(('float64' in p.name)!=is_double for p in dependencies):
        raise RuntimeError('Missing or mixed-precision Bullet dependencies')
    digests={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)}
    if digests[str(expected)]!=contract['plugin_sha256']:raise RuntimeError('Mapped plugin hash differs')
    audit=contract.get('binary_audit')
    if audit and any(digests.get(name)!=sha for name,sha in audit['binary_sha256'].items()):
        raise RuntimeError('Mapped dependency differs from audited binary')
    report={'passed':True,'purpose':'actual_process_mapping_not_configuration_only','simulator_pid':int(process.name),
        'backend':contract['backend'],'scalar_precision_bits':contract['scalar_precision_bits'],'mapped_binary_sha256':digests}
    (root/'physics_process_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
