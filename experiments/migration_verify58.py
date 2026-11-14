"""Read every archived file and authenticate it before local cleanup."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
from restore_migration58 import Joined
ROOT=Path(__file__).resolve().parents[1]
EXPORT=ROOT.parent/'migration-export-20261114'
def main():
    report=dict(status='waiting_for_archives',verified=[])
    def save():
        p=EXPORT/'archive-verification.tmp';p.write_text(json.dumps(report,indent=2));p.replace(EXPORT/'archive-verification.json')
    save()
    try:
        for name in ['workspace-index.json','legacy-index.json']:
            while not (EXPORT/name).exists():time.sleep(5)
            index=json.loads((EXPORT/name).read_text());expected={r['path']:r for r in index['files']};seen={}
            report.update(status='verifying',current=name);save()
            with io.BufferedReader(Joined([EXPORT/r['name'] for r in index['parts']])) as joined:
                with tarfile.open(fileobj=joined,mode='r|gz') as archive:
                    for member in archive:
                        row=expected.get(member.name)
                        if row is None:raise RuntimeError('Unindexed member: '+member.name)
                        if member.islnk():
                            digest,size=seen[member.linkname]
                        elif member.isfile():
                            with archive.extractfile(member) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                            size=member.size
                        else:raise RuntimeError('Unexpected member type')
                        if digest!=row['sha256'] or size!=row['size']:raise RuntimeError('Archive file hash mismatch: '+member.name)
                        seen[member.name]=(digest,size)
            if set(seen)!=set(expected):raise RuntimeError('Archive is missing indexed files')
            report['verified'].append(dict(index=name,files=len(seen),completed_unix=time.time()));save()
        report.update(status='complete_verified',current=None);save()
    except Exception as exc:
        report.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise
if __name__=='__main__':main()
