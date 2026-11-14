"""Restore verified migration release assets into a fresh repository checkout."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
class Joined(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:return 0
            count=self.current.readinto(buffer)
            if count:return count
            self.current.close();self.current=None

def verified_parts(folder,index):
    manifest=json.loads((folder/index).read_text())
    paths=[]
    for row in manifest['parts']:
        path=folder/row['name']
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if path.stat().st_size!=row['size'] or digest!=row['sha256']:raise RuntimeError('Corrupt/missing asset: '+path.name)
        paths.append(path)
    return manifest,paths

def verify_workspace(index):
    failures=[]
    for row in index['files']:
        path=ROOT/row['path']
        if not path.is_file():failures.append(row['path']);continue
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if path.stat().st_size!=row['size'] or digest!=row['sha256']:failures.append(row['path'])
    if failures:raise RuntimeError(f'Workspace verification failed: {failures[:20]}')
    print(f"Verified {len(index['files'])} restored files",flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--load-docker',action='store_true');parser.add_argument('--create-containers',action='store_true')
    args=parser.parse_args()
    index,paths=verified_parts(args.assets,'workspace-index.json')
    with io.BufferedReader(Joined(paths)) as joined:
        with tarfile.open(fileobj=joined,mode='r|gz') as archive:
            archive.extractall(ROOT,filter='data')
    verify_workspace(index)
    if args.load_docker:
        _,paths=verified_parts(args.assets,'docker-index.json')
        process=subprocess.Popen(['docker','load'],stdin=subprocess.PIPE)
        with io.BufferedReader(Joined(paths)) as joined:
            with gzip.GzipFile(fileobj=joined,mode='rb') as raw:
                while block:=raw.read(4*1024**2):process.stdin.write(block)
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Docker load failed')
    if args.create_containers:
        for name,tag in [('research-aic-eval','baseline'),('research-aic-perception30','development')]:
            found=subprocess.run(['docker','container','inspect',name],capture_output=True).returncode==0
            if found:raise RuntimeError('Container already exists; inspect manually: '+name)
            subprocess.run(['docker','create','--name',name,'--gpus','all','--cpus','8',
                '--memory','12g','--memory-swap','13g','--shm-size','2g',
                '--mount',f'type=bind,source={ROOT},target=/research',
                '--entrypoint','sleep',f'aic-research-migration:{tag}-20261114','infinity'],check=True)
    print('Restored. Read MIGRATION_HANDOFF.md before continuing trials.')

if __name__=='__main__':main()
