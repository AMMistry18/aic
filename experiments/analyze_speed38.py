"""Audit completed physical runs and distinguish force pauses from speed limits."""
import json
import gzip
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/speed38';OUT.mkdir(exist_ok=True)
METHOD='original_perception_sfp_contact_datum_handoff'
runs=[f'dev_{seed}_{label}_{METHOD}' for label,seeds in [
 ('waypoint37v1',[45006,45005]),('qualification37v1',[45000,45001,45002]),
 ('speed38v1',[45000,45001]),('causal38v1',[45000,45001])] for seed in seeds]
rows=[]
for run in runs:
 folder=ROOT/'.artifacts/lastinch_runs'/run/run
 if not (folder/'policy.json').exists():continue
 policy=json.loads((folder/'policy.json').read_text())
 datum=policy.get('sfp_contact_datum',{});samples=datum.get('insertion_samples',[])
 result_path=folder/'verification.json'
 # The top-level verifier result is also preserved in the immutable batch snapshot.
 candidates=[ROOT/'.artifacts/repair37/waypoint37v1/progress.json',
  ROOT/'.artifacts/repair37/qualification37v1/progress.json',ROOT/'.artifacts/seeded_evaluation_progress.json',
  OUT/'speed38v1_progress.json',OUT/'causal38v1_progress.json']
 result=None
 for path in candidates:
  if path.exists():
   for item in json.loads(path.read_text()).get('results',[]):
    if item['run']==run:result=item
 if result is None:continue  # Never report unfinished/unchecked attempts as verified.
 force=np.linalg.norm([s['force_port_N'] for s in samples],axis=1) if samples else np.array([])
 times=np.array([s['sim_s'] for s in samples]);depth=np.array([s['estimated_depth_m'] for s in samples])
 phases=[]
 log=ROOT/'.artifacts/container'/(run+'_policy.log')
 for match in re.finditer(r'DATUM_PHASE (\S+) sim_s=([\d.]+)',log.read_text(errors='replace')):
  phases.append({'phase':match[1],'sim_s':float(match[2])})
 row=dict(run=run,verified_success=result['verified_success'],reason=result['reason'],
  motion_profile=datum['motion_profile'],commanded_travel_mm_s=datum['travel_speed_m_s']*1000,
  commanded_terminal_mm_s=datum['terminal_speed_m_s']*1000,
  travel_sim_s=float(times[-1]-times[0]) if samples else None,travel_control_samples=len(samples),
  travel_force_pause_samples=int(np.sum(force>=3)),
  travel_control_observed_peak_N=float(force.max()) if samples else None,
  complete_wrist_audit=result['high_rate_wrist_audit'],terminal_metrics=result['terminal_metrics'],
  policy_recorded_wall_s=policy['wall_s'],configuration_wall_s=policy['configuration_wall_s'],
  action_sim_s=policy['sim_s'],phases=phases,
  physical_scope='stable clearance seat while held; upstream aligned rigid grasp')
 truth_path=folder/'truth_trace.json.gz'
 if truth_path.exists() and samples:
  truth=json.load(gzip.open(truth_path,'rt'))
  interior=[p for p in truth if times[0]<=p['sim_time_s']<=times[-1] and .005<=p['depth_m']<=.040]
  if len(interior)>2:
   t=np.array([p['sim_time_s'] for p in interior]);z=np.array([p['depth_m'] for p in interior])
   dt=np.diff(t);vel=np.diff(z)[dt>0]/dt[dt>0]*1000
   row['independent_interior_velocity_mm_s']={
    'median':float(np.median(vel)),'p25':float(np.percentile(vel,25)),
    'p75':float(np.percentile(vel,75)),'intervals':len(vel),
    'definition':'independent consecutive truth-depth differences during travel, 5-40 mm depth; offline diagnostic only'}
 rows.append(row)
comparisons=[]
for label,seed in [(label,seed) for label in ['speed38v1','causal38v1'] for seed in [45000,45001]]:
 pair={row['motion_profile']:row for row in rows if row['run'].startswith(f'dev_{seed}_') and
       ('qualification37v1' in row['run'] or label in row['run'])}
 if len(pair)!=2:continue
 old,new=pair['fast37'],pair['fast38']
 if old['travel_sim_s'] is None or new['travel_sim_s'] is None:
  comparisons.append(dict(candidate_label=label,seed=seed,both_verified=old['verified_success'] and new['verified_success'],
   travel_comparison_available=False,reason='At least one attempt failed before travel; retain it without assigning a speed gain'))
  continue
 comparisons.append(dict(candidate_label=label,seed=seed,both_verified=old['verified_success'] and new['verified_success'],
  travel_reduction_percent=100*(1-new['travel_sim_s']/old['travel_sim_s']),
  action_sim_reduction_percent=100*(1-new['action_sim_s']/old['action_sim_s']),
  recorded_wall_reduction_percent=100*(1-new['policy_recorded_wall_s']/old['policy_recorded_wall_s']),
  interpretation='same development seed; separate RGB captures and host scheduling; not held-out population estimate'))
report=dict(rows=rows,comparisons=comparisons,held_out_final_trials_run=0,paper_ready=False,
 force_pause_interpretation='A low peak is not a commanded-force cap; count actual pause conditions during travel',
 timing_caveat='Recorded process wall excludes setup; V34 45006 postprocessing crossed host sleep')
(OUT/'speed_attribution.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'completed_analyzed':len(rows),'travel_force_pauses':sum(r['travel_force_pause_samples'] for r in rows),
 'comparisons':comparisons},indent=2))
