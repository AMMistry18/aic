import numpy as np
from aic_model.insertion.sc_body_refinement import partial_translation_contract


def test_partial_contract_separates_lateral_budget_from_contact_capture():
    vectors=[[.00018,0,.0002],[0,.0001,-.0001],[0,0,.0003]]
    result=partial_translation_contract(vectors,np.eye(3),usable_fit=True)
    assert result['lateral_candidate_qualified']
    assert result['requires_axial_contact_measurement']
    assert not result['full_visual_pose_certified']


def test_partial_contract_cannot_hide_excess_lateral_or_unbounded_axial_error():
    for v in [[[.000251,0,0]]*3,[[0,0,.000501]]*3]:
        assert not partial_translation_contract(v,np.eye(3),usable_fit=True)['lateral_candidate_qualified']
    assert not partial_translation_contract(np.zeros((3,3)),np.eye(3),usable_fit=False)['lateral_candidate_qualified']
