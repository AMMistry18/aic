"""Content-deduplicated, split migration archive with a per-file SHA256 index."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
EXPORT=ROOT.parent/'migration-export-20261114'
EXCLUDED_DIRS={'.git','__pycache__','.pytest_cache','.pixi','.tacc','.blender-toolkit','node_modules'}
EXCLUDED_NAMES={'credentials.txt','.env','info.txt','.DS_Store'}

class Parts:
    def __init__(self,name):
        self.name=name;self.index=0;self.file=None;self.size=0;self.digest=None;self.rows=[]
    def writable(self):return True
    def flush(self):
        if self.file:self.file.flush()
    def finish(self):
        if self.file:
            self.file.close();self.rows.append(dict(name=self.path.name,size=self.size,sha256=self.digest.hexdigest()))
            print(json.dumps(dict(part=self.path.name,bytes=self.size,sha256=self.digest.hexdigest())),flush=True)
            self.file=None
    def write(self,data):
        view=memoryview(data);total=len(view)
        while view:
            if self.file is None:
                self.path=EXPORT/f'{self.name}.tar.gz.part{self.index:03d}'
                self.file=self.path.open('xb');self.index+=1;self.size=0;self.digest=hashlib.sha256()
            n=min(len(view),1024**3-self.size)
            self.file.write(view[:n]);self.digest.update(view[:n]);self.size+=n;view=view[n:]
            if self.size==1024**3:self.finish()
        return total

def pack_workspace():
    if json.loads((ROOT/'experiments/migration58/stop_progress.json').read_text())['status']!='stopped_at_completed_trial_boundary':
        raise RuntimeError('Stop research writers before archiving')
    EXPORT.mkdir(exist_ok=True)
    if (EXPORT/'workspace-index.json').exists():raise FileExistsError('Preserve previous archive')
    files=[];excluded=[]
    for base,dirs,names in os.walk(ROOT):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(names):
            p=Path(base)/name
            if name in EXCLUDED_NAMES or name.endswith(('.pyc','.tmp')):
                excluded.append(str(p.relative_to(ROOT)));continue
            if p.is_symlink():
                excluded.append(str(p.relative_to(ROOT)));continue
            files.append(p)
    parts=Parts('research-workspace');records=[];known={};began=time.time()
    with gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=parts,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for i,p in enumerate(files):
                digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
                name=p.relative_to(ROOT).as_posix();info=archive.gettarinfo(str(p),arcname=name)
                key=(info.size,digest)
                records.append(dict(path=name,size=info.size,sha256=digest))
                if key in known:
                    info.type=tarfile.LNKTYPE;info.linkname=known[key];info.size=0;archive.addfile(info)
                else:
                    known[key]=name
                    with p.open('rb') as stream:archive.addfile(info,stream)
                if i%1000==0:print(json.dumps(dict(files=i,total=len(files),elapsed_s=time.time()-began)),flush=True)
    parts.finish()
    manifest=dict(kind='research-workspace',parts=parts.rows,files=records,excluded=excluded,
                  archive_layout='repository root; concatenate numbered parts, gzip decompress, tar extract',
                  excluded_directories=sorted(EXCLUDED_DIRS),created_unix=time.time())
    (EXPORT/'workspace-index.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(completed=True,files=len(files),compressed_bytes=sum(p['size'] for p in parts.rows))),flush=True)

def pack_docker():
    EXPORT.mkdir(exist_ok=True)
    if (EXPORT/'docker-index.json').exists():raise FileExistsError('Preserve image archive')
    metadata=[];tags=[]
    for name,tag in [('research-aic-eval','aic-research-migration:baseline-20261114'),('research-aic-perception30','aic-research-migration:development-20261114')]:
        info=json.loads(subprocess.run(['docker','inspect',name],capture_output=True,text=True,check=True).stdout)[0]
        if info['State']['Running']:raise RuntimeError('Container still running')
        subprocess.run(['docker','commit',name,tag],check=True,capture_output=True,timeout=180)
        # Deliberately save runtime config without host-specific bind paths.
        metadata.append(dict(container=name,tag=tag,config=info['Config'],host_config=info['HostConfig']))
        tags.append(tag)
    (EXPORT/'docker-runtime.json').write_text(json.dumps(metadata,indent=2))
    parts=Parts('research-docker')
    process=subprocess.Popen(['docker','save',*tags],stdout=subprocess.PIPE,stderr=(EXPORT/'docker-save.log').open('wb'))
    with gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=parts,mtime=0) as compressed:
        while data:=process.stdout.read(4*1024**2):compressed.write(data)
    if process.wait()!=0:raise RuntimeError('Docker save failed')
    parts.finish()
    (EXPORT/'docker-index.json').write_text(json.dumps(dict(kind='research-docker',parts=parts.rows,tags=tags),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['workspace','docker']);args=p.parse_args()
    pack_workspace() if args.kind=='workspace' else pack_docker()
