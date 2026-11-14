"""Install reviewed free-waypoint candidate only after the frozen queue ends."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
state=subprocess.check_output(['docker','inspect','research-aic-eval','--format','{{.State.Status}}'],text=True).strip()
assert state=='exited',state
OUT=ROOT/'.artifacts/repair37/waypoint_candidate'
meta=json.loads((OUT/'candidate.json').read_text())
target=ROOT/'aic_model/aic_model/insertion'
assert hashlib.sha256((target/'sfp_contact_datum.py').read_bytes()).hexdigest()==meta['base_sha256']
assert hashlib.sha256((target/'tracking_compensation.py').read_bytes()).hexdigest()==meta['tracking_base_sha256']
backup=ROOT/'.artifacts/repair37/source_v33';backup.mkdir(exist_ok=False)
for name in ['sfp_contact_datum.py','tracking_compensation.py']:shutil.copy2(target/name,backup/name)
for name in ['run_original_insertion.py','run_seeded_evaluation.py']:shutil.copy2(ROOT/'experiments'/name,backup/name)
for name in ['sfp_contact_datum.py','tracking_compensation.py','waypoint_motion.py']:shutil.copy2(OUT/name,target/name)
runner=ROOT/'experiments/run_seeded_evaluation.py';text=runner.read_text()
assert "choices=['legacy36','fast36']" in text
text=text.replace("choices=['legacy36','fast36']","choices=['legacy36','fast36','fast37']")
text=text.replace("a.sfp_motion_profile=='fast36'","a.sfp_motion_profile in ('fast36','fast37')")
old="if __name__=='__main__':main()"
assert old in text
text=text.replace(old,"""if __name__=='__main__':
    # Per-worker idle-sleep protection; no display or global power changes.
    power=None
    if sys.platform=='win32':
        import ctypes
        power=ctypes.windll.kernel32
        if not power.SetThreadExecutionState(0x80000001):
            raise RuntimeError('Could not request active evaluation idle-sleep protection')
    try:main()
    finally:
        if power is not None:power.SetThreadExecutionState(0x80000000)
""")
runner.write_text(text)
harness=ROOT/'experiments/run_original_insertion.py';text=harness.read_text()
assert "result['harness_revision']=33" in text
harness.write_text(text.replace("result['harness_revision']=33","result['harness_revision']=34"))
meta.update(installed=True,harness_revision=34)
(OUT/'installed.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Installed fast37 free-waypoint candidate; legacy36 and fast36 paths retained')
