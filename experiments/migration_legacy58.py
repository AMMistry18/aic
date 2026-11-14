"""Preserve older Flowstate sources/models, omitting obsolete image bundles."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from migration_pack58 import Parts,EXPORT,ROOT,EXCLUDED_DIRS,EXCLUDED_NAMES

def main():
    source=ROOT.parent/'.flowstate-build'
    if (EXPORT/'legacy-index.json').exists():raise FileExistsError('Preserve first export')
    parts=Parts('legacy-flowstate');records=[];omitted=[];known={}
    pattern=re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    with gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=parts,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for base,dirs,names in os.walk(source):
                dirs[:]=sorted(d for d in dirs if d not in EXCLUDED_DIRS)
                for name in sorted(names):
                    p=Path(base)/name
                    if name in EXCLUDED_NAMES or p.suffix in ('.tar','.pyc','.wandb') or p.is_symlink():
                        omitted.append(p.relative_to(source).as_posix());continue
                    if p.stat().st_size<2*1024**2 and p.suffix in ('.py','.sh','.json','.yaml','.txt','.log','.md',''):
                        if pattern.search(p.read_bytes()):
                            omitted.append(p.relative_to(source).as_posix());continue
                    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                    member='.migration_legacy58/flowstate-build/'+p.relative_to(source).as_posix()
                    info=archive.gettarinfo(str(p),arcname=member);key=(info.size,digest)
                    records.append(dict(path=member,size=p.stat().st_size,sha256=digest,original_relative=p.relative_to(source).as_posix()))
                    if key in known:
                        info.type=tarfile.LNKTYPE;info.linkname=known[key];info.size=0;archive.addfile(info)
                    else:
                        known[key]=member
                        with p.open('rb') as f:archive.addfile(info,f)
    parts.finish()
    (EXPORT/'legacy-index.json').write_text(json.dumps(dict(kind='legacy-flowstate',parts=parts.rows,files=records,
        excluded=omitted,excluded_directories=sorted(EXCLUDED_DIRS),scope='Older Flowstate sources/weights; obsolete 31 image tar bundles omitted'),indent=2))
if __name__=='__main__':main()
