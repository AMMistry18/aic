"""Read-only original/checkpoint lineage and archived runtime identity audit."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='aic_example_policies/aic_example_policies/ros/weights/'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

definitions=[('SC plug','best_sc_plug_pose.pt',None,None),
    ('SC mouth','best_sc_mouth_pose.pt','.artifacts/perception_data30/training/sc_mouth_standard30_v1','AIC_SC_POSE_WEIGHTS'),
    ('SFP plug','best_sfp_plug_pose.pt','.artifacts/sfp_visual33/training/plug33_v1','AIC_SFP_PLUG_POSE_WEIGHTS'),
    ('SFP NIC','best.pt','.artifacts/sfp_visual33/training/nic33_v1','AIC_NIC_WEIGHTS')]
rows=[]
remote=json.loads((ROOT/'.artifacts/repair36/article/remote_checkpoint_metadata.json').read_text())
remote_by_name={r['name']:r for r in remote['checkpoints']}
for name,original,training,variable in definitions:
    path=ROOT/BASE/original;digest=sha(path)
    recorded=subprocess.check_output(['git','cat-file','blob','HEAD:'+BASE+original],cwd=ROOT)
    original_preserved=hashlib.sha256(recorded).hexdigest()==digest
    assert original_preserved,name
    blob=hashlib.sha1(b'blob '+str(len(recorded)).encode()+b'\0'+recorded).hexdigest()
    assert blob==remote_by_name[original]['git_blob_sha']
    row=dict(component=name,original_path=BASE+original,original_sha256=digest,original_matches_tracked_HEAD=True,
             original_matches_article_linked_repository=True,remote_git_blob_sha=blob)
    if training:
        record=json.loads((ROOT/training/'training_result.json').read_text())
        adapted=ROOT/training/'run/weights/best.pt'
        assert record['original_weight_sha256']==digest
        saved=record['checkpoints']['best.pt'] if 'checkpoints' in record else record['recovery']['best.pt_sha256']
        assert sha(adapted)==saved
        row.update(training_record=training+'/training_result.json',active_path=adapted.relative_to(ROOT).as_posix(),
                   active_sha256=sha(adapted),runtime_override=variable,adapted_from_original=True,
                   final_cases_used=record.get('held_out_final_cases_used',0))
        if 'final_100000_series_used' in record:assert record['final_100000_series_used'] is False
        else:assert record['held_out_final_cases_used']==0
    else:row.update(active_path=BASE+original,active_sha256=digest,adapted_from_original=False)
    rows.append(row)
runs={
    'sc':'dev_42000_regression35v1_original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff',
    'sfp':'dev_45003_pulse36regv1_original_perception_sfp_contact_datum_handoff'}
runtime=[]
for connector,run in runs.items():
    p=ROOT/'.artifacts/lastinch_runs'/run/run/'policy.json';policy=json.loads(p.read_text())
    for row in rows:
        if not row['component'].lower().startswith(connector):continue
        if row.get('runtime_override'):
            measured=policy['explicit_runtime_weight_overrides'][row['runtime_override']]['sha256']
        else:measured=policy['weight_sha256'][Path(row['original_path']).name]
        assert measured==row['active_sha256'],(run,row['component'])
        runtime.append(dict(run=run,component=row['component'],loaded_sha256=measured,matches_expected=True))
out=dict(article_url='https://github.com/SatyaA27/intrinsic-ai-for-industry-challenge-team-tar-2-sol/blob/main/index.md',
    article_git_blob_sha='6e0483ce24854aaa2aea6b28962ee0270ca31360',
    article_does_not_pin_binary_checkpoint_hashes=True,
    article_linked_repository=remote['repository'],article_linked_repository_commit=remote['commit'],
    scope='Original files match the article-linked repository at the pinned commit; adapted weights match training lineage and runtime hashes',
    original_pipeline_retained=['confidence-weighted multiview plug estimation','SC native crop refinement',
        'per-grasp RGB-to-TCP transform','geometric/force insertion, no RL checkpoint'],
    current_scope_changes=['last-inch aligned rigid grasp instead of full survey/pick/motion sequence',
        'adapted mouth/NIC/SFP weights','SFP CAD fixture refinement and connector-specific contact datums',
        'bounded own-TCP tracking and qualified reversible lip excitation'],
    checkpoints=rows,archived_runtime_checks=runtime)
(ROOT/'.artifacts/repair36/checkpoint_lineage.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
