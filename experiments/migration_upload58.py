"""Upload and independently verify each release asset using GitHub SHA256."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
EXPORT=ROOT.parent/'migration-export-20261114'
REPO='AMMistry18/aic';TAG='research-migration-20261114'

def run(args,**kwargs):return subprocess.run(args,capture_output=True,text=True,check=True,**kwargs)
def api():
    releases=json.loads(run(['gh','api',f'repos/{REPO}/releases?per_page=100'],timeout=60).stdout)
    return next(r for r in releases if r['tag_name']==TAG)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    state=dict(status='waiting_for_branch_and_archives',verified_assets=[],started_unix=time.time())
    def save():
        state['updated_unix']=time.time();p=EXPORT/'upload-progress.tmp'
        p.write_text(json.dumps(state,indent=2));p.replace(EXPORT/'upload-progress.json')
    save()
    try:
        while True:
            probe=subprocess.run(['git','ls-remote','--heads','origin','research/gazebo-rap-evaluation'],capture_output=True,text=True,timeout=60)
            if probe.returncode==0 and probe.stdout.strip():break
            time.sleep(10)
        state['remote_branch']=probe.stdout.strip();save()
        existing=subprocess.run(['gh','release','view',TAG,'--repo',REPO],capture_output=True,text=True,timeout=60)
        if existing.returncode:
            run(['gh','release','create',TAG,'--repo',REPO,'--target','research/gazebo-rap-evaluation',
                '--title','Intrinsic insertion research — complete machine migration (2026-11-14)',
                '--notes-file',str(EXPORT/'release-notes.md'),'--draft'],timeout=60)
        def upload(path):
            digest=sha(path);size=path.stat().st_size
            assets={a['name']:a for a in api()['assets']}
            asset=assets.get(path.name)
            if asset is None:
                state.update(status='uploading',current_asset=path.name,current_bytes=size);save()
                for attempt in range(3):
                    try:
                        run(['gh','release','upload',TAG,str(path),'--repo',REPO],timeout=7200);break
                    except Exception:
                        asset=next((a for a in api()['assets'] if a['name']==path.name),None)
                        if asset and asset.get('digest')=='sha256:'+digest:break
                        if asset:run(['gh','api','-X','DELETE',f"repos/{REPO}/releases/assets/{asset['id']}"],timeout=60)
                        if attempt==2:raise
                        time.sleep(5)
            asset=next(a for a in api()['assets'] if a['name']==path.name)
            if asset['size']!=size:raise RuntimeError('Remote size mismatch: '+path.name)
            if asset.get('digest'):
                if asset['digest']!='sha256:'+digest:raise RuntimeError('Remote digest mismatch: '+path.name)
                basis='GitHub server-computed SHA256'
            else:
                verify=EXPORT/'remote-verify';verify.mkdir(exist_ok=True)
                run(['gh','release','download',TAG,'--repo',REPO,'--pattern',path.name,'--dir',str(verify)],timeout=7200)
                copy=verify/path.name
                if sha(copy)!=digest:raise RuntimeError('Downloaded asset digest mismatch')
                copy.unlink();basis='Downloaded asset SHA256'
            row=dict(name=path.name,size=size,sha256=digest,basis=basis,url=asset['browser_download_url'])
            state['verified_assets'].append(row);save();print(json.dumps(row),flush=True)
        for index_name in ['workspace-index.json','legacy-index.json','docker-index.json']:
            state.update(status='waiting_for_archive',current_asset=index_name);save()
            while not (EXPORT/index_name).exists():time.sleep(5)
            index=json.loads((EXPORT/index_name).read_text())
            for part in index['parts']:upload(EXPORT/part['name'])
            upload(EXPORT/index_name)
        upload(EXPORT/'docker-runtime.json')
        # Publish only after every advertised archive and index is verified.
        run(['gh','release','edit',TAG,'--repo',REPO,'--draft=false'],timeout=60)
        state.update(status='complete_remote_verified',current_asset=None,release_url=api()['html_url'],ended_unix=time.time());save()
    except Exception as exc:
        state.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise
if __name__=='__main__':main()
