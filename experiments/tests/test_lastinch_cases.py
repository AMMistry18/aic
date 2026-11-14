from experiments.lastinch_cases import cases, TARGETS


def test_pilot_covers_every_target_at_both_handoff_distances():
    rows=cases('pilot')
    for connector,index,port in TARGETS:
        assert {r['gap_mm'] for r in rows if (r['connector'],r['target_index'],r['sfp_port'])==(connector,index,port)}=={5,15}


def test_final_counts_are_balanced_and_split_seeds_never_overlap():
    groups={s:cases(s) for s in ('pilot','final','robustness')}
    seeds=[r['seed'] for rows in groups.values() for r in rows]
    assert len(seeds)==len(set(seeds))
    assert sum(r['connector']=='sc' for r in groups['final'])==100
    assert sum(r['connector']=='sfp' for r in groups['final'])==100
    for connector,i,p in TARGETS:
        expected=20 if connector=='sc' else 10
        assert sum((r['connector'],r['target_index'],r['sfp_port'])==(connector,i,p) for r in groups['final'])==expected


def test_conditions_share_case_seed_and_keep_force_protection():
    for row in cases('pilot')+cases('final'):
        assert len(set(row['method_order']))==3
        assert 0<row['force_abort_n']<=18
        assert row['execution_status']=='not_run'
