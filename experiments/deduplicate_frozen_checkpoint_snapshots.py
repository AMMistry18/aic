"""Preserve completed trial checkpoint bytes using local NTFS hard links.

Only source_snapshot/*.pt under completed archives are eligible. Live policy
weights, raw measurements and active simulation files are never modified.
Every replacement has a verified independent content-addressed copy and an
atomic rollback backup. Hard links retain all original paths and SHA256 bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
ROOT=Path(__file__).resolve().parents[1]
ARCHIVES=(ROOT/'.artifacts/lastinch_runs').resolve()
CACHE=(ROOT/'.artifacts/frozen_checkpoint_objects').resolve()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def checked(p,root):
    resolved=p.resolve()
    if root not in resolved.parents:raise ValueError(f'Path outside intended directory: {resolved}')
    return resolved


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');a=p.parse_args()
    entries=[];sizes={};counts={};free_before=shutil.disk_usage(ROOT).free
    for archive in ARCHIVES.iterdir():
        if not archive.is_dir() or not (archive/archive.name/'verified.json').is_file():continue
        for file in (archive/'source_snapshot').rglob('*.pt'):
            file=checked(file,ARCHIVES);digest=sha(file);size=file.stat().st_size
            entries.append((file,digest));sizes[digest]=size;counts[digest]=counts.get(digest,0)+1
    plan=dict(checkpoint_paths=len(entries),distinct_binaries=len(counts),estimated_reclaim_GiB=sum((counts[h]-1)*sizes[h] for h in counts)/2**30,apply=a.apply)
    print(json.dumps(plan),flush=True)
    if not a.apply:return
    CACHE.mkdir(exist_ok=True);verified_objects=set();replaced=0
    for source,digest in entries:
        target=checked(CACHE/(digest+'.pt'),CACHE);backup=checked(source.with_suffix('.pt.dedupbackup'),ARCHIVES)
        if backup.exists():raise FileExistsError('Preserve existing rollback backup')
        if digest not in verified_objects:
            if not target.exists():shutil.copy2(source,target)
            if sha(target)!=digest:raise ValueError('Checkpoint object integrity failure')
            verified_objects.add(digest)
        if os.path.samefile(source,target):continue
        # Only a byte-verified archival duplicate is replaced. If linking or
        # verification fails, restore that original file before propagating.
        source.replace(backup)
        try:
            os.link(target,source)
            if sha(source)!=digest:raise ValueError('Replacement changed checkpoint bytes')
        except BaseException:
            if source.exists():source.unlink()
            backup.replace(source)
            raise
        backup.unlink();replaced+=1
    # Freeze cached objects and their archive links; original live weights are
    # independent files and retain their existing permissions/content.
    for digest in verified_objects:(CACHE/(digest+'.pt')).chmod(stat.S_IREAD)
    report=plan|dict(replaced_paths=replaced,free_before_bytes=free_before,free_after_bytes=shutil.disk_usage(ROOT).free,
        original_bytes_preserved=True,live_policy_files_modified=False,
        objects={h:dict(size_bytes=sizes[h],archive_paths=[str(f.relative_to(ROOT)) for f,d in entries if d==h]) for h in counts})
    (CACHE/'deduplication_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='objects'}),flush=True)


if __name__=='__main__':main()
