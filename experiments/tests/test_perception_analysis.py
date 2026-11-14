import pytest
from experiments.analyze_perception import identity, paired_cluster_bootstrap


def test_metadata_keeps_snapshots_in_same_trial():
    assert identity("/old/location/trial_0010_v000_s01.json")[1] == "trial_0010"
    assert identity("trial_0010_v001_s02.json")[1] == "trial_0010"


def test_bootstrap_rejects_unmatched_groups():
    with pytest.raises(ValueError, match="matched"):
        paired_cluster_bootstrap({"one": {}}, {"two": {}}, 100)
