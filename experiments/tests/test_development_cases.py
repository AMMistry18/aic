import json
from pathlib import Path
import pytest
from experiments.run_seeded_evaluation import validate_cases

def test_all_five_targets_and_both_sfp_slots_are_legal():
    cases=json.loads((Path(__file__).resolve().parents[1]/'configs/development_cases.json').read_text())['cases']
    validate_cases(cases)
    for kind in ('sc','sfp'):
        assert {c['target_index'] for c in cases if c['connector']==kind}==set(range(5))
    assert {c['sfp_port'] for c in cases if c['connector']=='sfp'}=={0,1}

def test_sc_cannot_borrow_second_sfp_slot():
    with pytest.raises(ValueError,match='Invalid sc slot'):
        validate_cases([{'case_id':'dev_bad','connector':'sc','target_index':1,'sfp_port':1}])
