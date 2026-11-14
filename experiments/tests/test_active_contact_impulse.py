import copy
from experiments.contact_depth import SIGNED_DISTANCE,physical_contact,active_constraint_impulses

RUNTIME={'gz_physics_version':'8.3.0','plugin_sha256':'audited_binary',
         'depth_convention':SIGNED_DISTANCE,
         'contact_wrench_semantics':'not a force vector: 8.3.0 reports scalar impulse in xyz; use independent wrist FT'}


def packet(impulse):
    return {'depths':[1e-11],'wrenches':[{
        'body_1_wrench':{'force':dict.fromkeys(('x','y','z'),impulse)},
        'body_2_wrench':{'force':dict.fromkeys(('x','y','z'),-impulse)}}]}


def test_active_solver_contact_differs_from_empty_cached_manifold():
    assert not physical_contact(packet(0),RUNTIME)
    assert physical_contact(packet(5.8e-6),RUNTIME)
    assert active_constraint_impulses(packet(5.8e-6),RUNTIME)[0]['normal_impulse_Ns']==5.8e-6


def test_unrecognized_encoding_or_binary_never_supplies_impulse_evidence():
    for change in ({'gz_physics_version':'8.4.0'},{'plugin_sha256':None},{'contact_wrench_semantics':'ordinary force vector'}):
        assert not physical_contact(packet(5.8e-6),{**RUNTIME,**change})
    malformed=packet(5.8e-6)
    malformed['wrenches'][0]['body_2_wrench']['force']['z']=0
    assert not physical_contact(malformed,RUNTIME)
    assert not physical_contact(packet(float('nan')),RUNTIME)
    assert not physical_contact({'depths':[1e-11],'wrenches':[]},RUNTIME)
