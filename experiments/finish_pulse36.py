"""Archive and describe only the actual completed V33 development outcome."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
run='dev_45002_pulse36v1_original_perception_sfp_contact_datum_handoff'
batch=json.loads((ROOT/'.artifacts/seeded_evaluation_progress.json').read_text())
assert len(batch['results'])==1 and batch['results'][0]['run']==run
result=batch['results'][0]
assert result['status']=='completed'
folder=ROOT/'.artifacts/lastinch_runs'/run/run
policy=json.loads((folder/'policy.json').read_text())
d=policy['sfp_contact_datum'];samples=d.get('insertion_samples',[])
excitation=[dict(offset_m=r['offset_m'],accepted=r['proof']['accepted'],
    rise_N=r['proof'].get('contrast_N'),fall_N=r['proof'].get('force_fall_N'),
    rise_margin_3sem_N=r['proof'].get('margin_3sem_N'),fall_margin_3sem_N=r['proof'].get('force_fall_margin_3sem_N'),
    stationary=r['own_tip_stationary_across_excitation']) for r in d.get('qualified_lip_excitations',[])]
summary=dict(run=run,verified_success=result['verified_success'],reason=result['reason'],policy_reason=d['reason'],
             terminal_metrics=result['terminal_metrics'],force_audit=result['high_rate_wrist_audit'],
             wall_s=policy['wall_s'],configuration_wall_s=policy['configuration_wall_s'],sim_s=policy['sim_s'],
             insertion_segment_sim_s=samples[-1]['sim_s']-samples[0]['sim_s'] if samples else None,
             qualified_excitations=excitation,held_out_final_trials_run=0,
             scope='one selected development case; no robustness claim')
(ROOT/'.artifacts/repair36/pulse36v1_completed.json').write_text(json.dumps(batch,indent=2)+'\n')
(ROOT/'.artifacts/repair36/pulse36v1_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
note=f"""Latest completed V33 `pulse36v1` outcome: **{int(result['verified_success'])}/1**
verified SFP insertion on selected development case `dev_45002`.
Policy reason: `{d['reason']}`; independent reason: `{result['reason']}`.
Complete raw-force audit: {result['high_rate_wrist_audit']['qualified']};
2000 Hz peak {result['high_rate_wrist_audit']['peak_tared_force_N']:.6f} N under
the prospectively labelled 20 N ceiling. Wall {policy['wall_s']:.6f} s includes
configuration {policy['configuration_wall_s']:.6f} s; action sim {policy['sim_s']:.6f} s.
Recorded insertion segment: {summary['insertion_segment_sim_s']} sim s.
Actual lip-excitation measurements, independent terminal ranges and all audit
details: `.artifacts/repair36/pulse36v1_summary.json`; frozen full batch:
`.artifacts/repair36/pulse36v1_completed.json`. Do not pool different revisions.
Prior running statuses below are historical. Final cases remain unopened and
combined SC/SFP robustness is not yet qualified.

"""
for name in ['EXPERIMENT_PLAN.md','EXPERIMENTS.md','PAPER_RESULTS.md','RESEARCH_PROGRESS.md']:
    path=ROOT/name;first,rest=path.read_text(encoding='utf-8').split('\n',1)
    path.write_text(first+'\n\n'+note+rest.lstrip('\n'),encoding='utf-8')
path=ROOT/'.artifacts/active_original_perception29.json';active=json.loads(path.read_text())
active.update(status='pulse36v1_completed_development_case',live_insertion_world=False,current_run=None,
              last_run=run,last_result=summary,held_out_final_trials_run=0,paper_ready=False)
path.write_text(json.dumps(active,indent=2)+'\n')
print(json.dumps(summary,indent=2))
