"""Replay immutable policy observations; no simulator truth enters the detector."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from aic_model.insertion.lip_search import stationary_search_candidate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair39';OUT.mkdir(exist_ok=True)
source=OUT/'source_v36/aic_model/aic_model/insertion/sfp_contact_datum.py'
text=source.read_text();function=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='kinematic_lip_candidate')
namespace={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
old=namespace['kinematic_lip_candidate']
run='dev_45001_causal38v1_original_perception_sfp_contact_datum_handoff'
policy_path=ROOT/'.artifacts/lastinch_runs'/run/run/'policy.json';policy=json.loads(policy_path.read_text())
raw=[r for r in policy['sfp_contact_datum']['axial_probe_samples'] if r['offset_m']==-.002]
samples=[[r['sim_s'],r['measured_depth_m'],r['applied_command_m'],-r['force_port_N'][2]] for r in raw]
events={};first_stationary=None
for i,row in enumerate(samples):
 window=np.array([r for r in samples[:i+1] if r[0]>=row[0]-.83])
 if len(window)>=20 and window[-1,0]-window[0,0]>=.77:
  travel=float(np.ptp(window[:,1]));lead=float(np.median(window[-8:,2]-window[-8:,1]))
  if first_stationary is None and travel<=.000025 and lead>=.00015:
   first_stationary=dict(sim_s=row[0],maximum_gap_s=float(np.diff(window[:,0]).max()),
      current_window_command_advance_m=float(window[-1,2]-window[0,2]),observed_travel_m=travel)
 for name,fn in [('original',old),('repaired',stationary_search_candidate)]:
  if name in events:continue
  result=fn(samples[:i+1])
  if result is not None:events[name]=dict(sim_s=row[0],since_search_start_s=row[0]-samples[0][0],evidence=result)
report=dict(run=run,policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
 old_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input='recorded policy tip/command/time/wrist observations only',
 old_readiness=events.get('original'),repaired_readiness=events.get('repaired'),first_stationary_window=first_stationary,
 original_search_duration_s=samples[-1][0]-samples[0][0],physical_repaired_outcome='pending',
 interpretation='Readiness for one mandatory fixed reversible force test, not contact or seating acceptance')
(OUT/'lip_search_replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
