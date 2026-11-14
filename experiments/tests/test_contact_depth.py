import pytest
from experiments.contact_depth import penetration_m,touching,resolve_contract,SIGNED_DISTANCE,PENETRATION
from experiments.contact_depth import same_engine_contract

def test_bullet_8p3_cached_positive_gap_cannot_become_penetration_or_touch():
    assert penetration_m(.0007048,SIGNED_DISTANCE)==0
    assert not touching({'depths':[.0007048]},SIGNED_DISTANCE)
    assert penetration_m(-.00102,SIGNED_DISTANCE)==.00102
    assert touching({'depths':[-.00102]},SIGNED_DISTANCE)

def test_native_positive_penetration_engine_is_not_sign_flipped():
    assert penetration_m(.001,PENETRATION)==.001
    with pytest.raises(ValueError):penetration_m(.1,'unknown')
    with pytest.raises(ValueError):penetration_m(float('nan'),SIGNED_DISTANCE)
    with pytest.raises(ValueError):resolve_contract({})


def test_collision_control_cannot_cross_precision_or_unidentified_binary():
    reference={'plugin_sha256':'float32_sha','gz_physics_version':'8.3.0','depth_convention':SIGNED_DISTANCE}
    assert same_engine_contract(reference,dict(reference))
    assert not same_engine_contract(reference,{**reference,'plugin_sha256':'float64_sha'})
    assert not same_engine_contract({},reference)
