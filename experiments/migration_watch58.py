"""Keep this laptop awake during transfer; preserve a compact status receipt."""
import ctypes
import json
from pathlib import Path
import platform
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
EXPORT=ROOT.parent/'migration-export-20261114'
def read(path):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def main():
    power=ctypes.windll.kernel32 if platform.system()=='Windows' else None
    if power:power.SetThreadExecutionState(0x80000001)
    try:
        while True:
            upload=read(EXPORT/'upload-progress.json');cleanup=read(EXPORT/'cleanup-progress.json')
            state=dict(upload=upload.get('status'),current_asset=upload.get('current_asset'),
                remotely_verified_assets=len(upload.get('verified_assets',[])),
                cleanup=cleanup.get('status'),updated_unix=time.time(),
                release='https://github.com/AMMistry18/aic/releases/tag/research-migration-20261114')
            path=ROOT/'experiments/migration58/live-status.json'
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(path)
            if upload.get('status')=='stopped_requires_inspection' or cleanup.get('status')=='stopped_requires_inspection':return
            finished=(cleanup.get('status')=='completed_verified_scoped_cleanup' or
                      upload.get('status')=='complete_remote_verified' and cleanup.get('status')=='blocked_by_automatic_review')
            if finished:
                subprocess.run(['gh','release','edit','research-migration-20261114',
                    '--repo','AMMistry18/aic','--title','Intrinsic insertion research — complete verified machine migration (2026-11-14)',
                    '--notes-file',str(EXPORT/'release-notes.md')],cwd=ROOT,check=True,timeout=120)
                # Preserve the remote verification and cleanup receipts in Git.
                for name in ['upload-progress.json','cleanup-progress.json','archive-verification.json']:
                    (ROOT/'experiments/migration58'/name).write_bytes((EXPORT/name).read_bytes())
                subprocess.run(['git','add','--','experiments/migration58'],cwd=ROOT,check=True)
                subprocess.run(['git','commit','-qm','Record verified remote migration and local cleanup status'],cwd=ROOT,check=True)
                subprocess.run(['git','push','origin','research/gazebo-rap-evaluation'],cwd=ROOT,check=True,timeout=300)
                return
            time.sleep(20)
    finally:
        if power:power.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
