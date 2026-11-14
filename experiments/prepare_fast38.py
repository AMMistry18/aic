"""Prepare a travel-only speed comparison outside the active frozen sources."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/speed38/candidate'
OUT.mkdir(parents=True,exist_ok=True)
changes={
 'aic_model/aic_model/insertion/sfp_contact_datum.py':[
  ("if profile in ('fast36','fast37'):return .005,.001,.5,True", "if profile in ('fast36','fast37'):return .005,.001,.5,True\n    if profile=='fast38':return .010,.002,.5,True"),
  ("getattr(self,'_motion_profile','legacy36')=='fast37'", "getattr(self,'_motion_profile','legacy36') in ('fast37','fast38')"),
  ("        if profile in ('fast36','fast37'):", "        if profile in ('fast36','fast37','fast38'):"),
  ("self._axial_tracking_bound_N=1.;self._combined_tracking_bound_N=1.2", "self._axial_tracking_bound_N=1.;self._combined_tracking_bound_N=1.2\n        if profile=='fast38':\n            self._axial_tracking_bound_N=1.5;self._combined_tracking_bound_N=1.75"),
 ],
 'aic_model/aic_model/insertion/tracking_compensation.py':[
  ("((.25,.5),(1.,1.2))", "((.25,.5),(1.,1.2),(1.5,1.75))"),
 ],
 'experiments/run_seeded_evaluation.py':[
  ("choices=['legacy36','fast36','fast37']", "choices=['legacy36','fast36','fast37','fast38']"),
  ("a.sfp_motion_profile in ('fast36','fast37')", "a.sfp_motion_profile in ('fast36','fast37','fast38')"),
 ],
 'experiments/run_original_insertion.py':[("result['harness_revision']=34", "result['harness_revision']=35")],
}
metadata={}
for name,replacements in changes.items():
 source=(ROOT/name).read_bytes();text=source.decode()
 for old,new in replacements:
  assert old in text,(name,old)
  text=text.replace(old,new)
 target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(text.replace('\r\n','\n').encode())
 metadata[name]={'before_sha256':hashlib.sha256(source).hexdigest(),
                 'candidate_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(OUT/'changes.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'prepared':len(metadata),'installed':False,'profile':'fast38',
 'free_mm_s':10,'terminal_mm_s':2,'force_pause_N':3,'abort_N':6,
 'axial_compensation_bound_N':1.5,'combined_compensation_bound_N':1.75,
 'measurement_and_seating_criteria':'unchanged'},indent=2))
