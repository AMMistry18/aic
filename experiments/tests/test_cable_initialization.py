import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from experiments.cable_initialization import orient_sc_free_chain_outboard,straighten_sc_free_chain
from experiments.lastinch_geometry import sdf_frames,parse_pose,transform

@pytest.mark.parametrize('horizontal',[False,True,'straight'])
def test_source_geometry_and_all_joint_anchor_positions_survive_initialization(horizontal):
    source=Path(__file__).resolve().parents[2]/'.artifacts/lastinch_resources/cable.sdf'
    if not source.exists():pytest.skip('Pinned expanded source cable must be prepared')
    model=ET.parse(source).getroot().find('model');before=copy.deepcopy(model);old=sdf_frames(before)
    # An arbitrary world orientation ensures this is not a model-axis trick.
    world=transform([.1,-.2,.08,.2,-.1,.7])
    if horizontal=='straight':straighten_sc_free_chain(model,world)
    else:orient_sc_free_chain_outboard(model,world,horizontal=horizontal)
    new=sdf_frames(model)
    for name in ('cable_connection_1','sc_plug_link','sc_tip_link'):
        np.testing.assert_allclose(new[name],old[name],atol=1e-12)
    for original,changed in zip(before.findall('link'),model.findall('link')):
        for tag in ('visual','collision','inertial'):
            assert [ET.tostring(e) for e in original.findall(tag)]==[ET.tostring(e) for e in changed.findall(tag)]
    assert [ET.tostring(j) for j in before.findall('joint')]==[ET.tostring(j) for j in model.findall('joint')]
    for joint in before.findall('joint'):
        parent=joint.findtext('parent');child=joint.findtext('child');child_anchor=parse_pose(joint)
        parent_anchor=np.linalg.inv(old[parent])@old[child]@child_anchor
        a=new[parent]@parent_anchor;b=new[child]@child_anchor
        np.testing.assert_allclose(a[:3,3],b[:3,3],atol=1e-11,err_msg=joint.get('name'))
        if joint.get('type')=='fixed':np.testing.assert_allclose(a,b,atol=1e-11)
