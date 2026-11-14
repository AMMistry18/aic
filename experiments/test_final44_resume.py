"""A failed attempt stays in the prefix; case identity cannot be substituted."""
import copy
import pytest
from experiments.resume_full_evaluation44 import validate_prefix,run_name


def row(case,status='completed'):
    chosen=copy.deepcopy(case);chosen['case_id']+='_final44v1_full'
    return dict(run=run_name(case,'full'),base_case_id=case['case_id'],ablation='full',
                case=chosen,final=True,status=status)


def test_failed_setup_is_consumed_and_cannot_be_replaced_by_later_success():
    cases=[dict(case_id=f'fake_{i}',connector='sc',seed=58000+i) for i in range(3)]
    jobs=[(c,'full') for c in cases]
    rows=[row(cases[0]),row(cases[1],'environment_or_execution_failure')]
    validate_prefix(rows,jobs)
    assert jobs[len(rows)][0]==cases[2]
    with pytest.raises(ValueError):validate_prefix([rows[0],row(cases[2])],jobs)


def test_same_identity_cannot_hide_changed_grasp_or_duplicate_execution():
    case=dict(case_id='fake_3',connector='sfp',seed=58003,grasp_jitter_seed=58004)
    jobs=[(case,'full'),(dict(case,case_id='fake_4'),'full')]
    changed=row(case);changed['case']['grasp_jitter_seed']=58005
    with pytest.raises(ValueError):validate_prefix([changed],jobs)
    with pytest.raises(ValueError):validate_prefix([row(case),row(case)],jobs)
