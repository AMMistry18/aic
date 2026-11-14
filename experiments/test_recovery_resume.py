"""Shared simulator seeds must not hide distinct grasp cases when resuming."""
import json
from pathlib import Path
from experiments.verify_recovery_batch_20261112 import completed_case_ids


def test_real_manifest_shared_seed_does_not_mark_second_grasp_case_complete():
    root=Path(__file__).resolve().parents[1]
    cases={c['case_id']:c for c in json.loads((root/'experiments/configs/sc_finish41_qualification.json').read_text())['cases']}
    assert cases['dev_43001']['seed']==cases['dev_42001']['seed']
    assert cases['dev_43001']['grasp_jitter_seed']!=cases['dev_42001']['grasp_jitter_seed']
    rows=[dict(run='dev_43001_recovery20261112v1_method',status='completed',case=cases['dev_43001'])]
    assert 'dev_43001' in completed_case_ids(rows)
    assert 'dev_42001' not in completed_case_ids(rows)


def test_failed_setup_retry_remains_unfinished_and_does_not_remove_failure():
    failed=dict(run='dev_45007_recovery20261112v1_method',status='environment_or_execution_failure')
    rows=[failed]
    assert 'dev_45007' not in completed_case_ids(rows)
    rows.append(dict(run='dev_45007_recovery20261112resume2_method',status='completed'))
    assert completed_case_ids(rows)=={'dev_45007'}
    assert len(rows)==2 and rows[0] is failed
