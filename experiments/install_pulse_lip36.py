"""Install reviewed candidate after the previous owned simulator has stopped."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
state=subprocess.check_output(['docker','inspect','research-aic-eval','--format','{{.State.Status}}'],text=True).strip()
assert state=='exited',state
out=ROOT/'.artifacts/repair36/source_v32';out.mkdir(exist_ok=False)
for name in ['aic_model/aic_model/insertion/sfp_contact_datum.py','aic_model/aic_model/insertion/tracking_compensation.py',
             'experiments/run_original_insertion.py','experiments/run_seeded_evaluation.py',
             'experiments/analyze_recorded_wrench.py','experiments/verify_recorded_trial.py']:
    shutil.copy2(ROOT/name,out/Path(name).name)
folder=ROOT/'.artifacts/repair36/pulse_lip_candidate'
spec=json.loads((folder/'candidate.json').read_text())
target=ROOT/'aic_model/aic_model/insertion/sfp_contact_datum.py'
assert hashlib.sha256(target.read_bytes()).hexdigest()==spec['based_on_sha256']
shutil.copy2(folder/'sfp_contact_datum.py',target)
shutil.copy2(folder/'qualified_lip_contact.py',target.parent/'qualified_lip_contact.py')
h=ROOT/'experiments/run_original_insertion.py';text=h.read_text()
assert "result['harness_revision']=32" in text
h.write_text(text.replace("result['harness_revision']=32","result['harness_revision']=33"))
spec.update(installed=True,installed_harness_revision=33)
(folder/'installed.json').write_text(json.dumps(spec,indent=2)+'\n')
