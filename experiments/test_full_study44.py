from collections import Counter
from experiments.full_study44 import connector_cases,paired_exact_mcnemar,summarize


def test_balanced_all_targets_and_slots_without_opening_final_seeds():
    # Dedicated development seeds verify the generator without opening final cases.
    sc=connector_cases('sc',56000);sfp=connector_cases('sfp',57000)
    assert Counter(c['target_index'] for c in sc)=={i:10 for i in range(5)}
    assert Counter((c['target_index'],c['sfp_port']) for c in sfp)=={(i,j):5 for i in range(5) for j in range(2)}
    assert sum(c['grasp_jitter_seed'] is not None for c in sc)==25
    assert sc==connector_cases('sc',56000)
    assert all(abs(v)<=.03 for c in sfp for v in c['angular_deg'])


def test_paired_analysis_retains_failed_and_successful_discordances():
    rows=[]
    for i in range(6):
        rows.extend([dict(base_case_id=str(i),ablation='full',verified_success=True),
                     dict(base_case_id=str(i),ablation='no_axial_correction',verified_success=False)])
    result=paired_exact_mcnemar(rows,'no_axial_correction')
    assert result['pairs']==6 and result['exact_two_sided_p']==.03125
