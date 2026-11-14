"""Check reset frame conventions against a synthetic engine world placement."""
import xml.etree.ElementTree as ET
import numpy as np
from experiments import challenge_grasp as cg


def test_world_offset_is_not_rotated_as_tcp_local(monkeypatch, tmp_path):
    models=tmp_path/'models'
    content={
        'sfp_sc_cable_reversed': '<sdf><model name="c"><experimental:params/><include><uri>model://SC Plug</uri><pose relative_to="cable_connection_0">-.052 0 0 0 0 3.14159</pose></include></model></sdf>',
        'cable_base_c_rotated_reversed': '<sdf><model name="b"><link name="cable_connection_0"><pose>.01 .02 .03 0 0 0</pose></link></model></sdf>',
        'SC Plug': '<sdf><model name="p"><link name="sc_tip_link"><pose>.01165 0 0 -1.570796 0 -1.570796</pose></link></model></sdf>'}
    # Include a version attribute, as the actual SDFormat root has one.
    for name,text in content.items():
        path=models/name/'model.sdf';path.parent.mkdir(parents=True)
        path.write_text(text.replace('<sdf>','<sdf version="1.9">'))
    monkeypatch.setattr(cg,'MODELS',models)
    monkeypatch.setattr(cg,'source_model',lambda name:ET.parse(models/name/'model.sdf').getroot().find('model'))
    tcp=cg.transform([.4,-.2,.5,.2,-.3,1.2])
    monkeypatch.setattr(cg,'robot_fk',lambda robot,q:{'gripper/tcp':tcp})
    grasp,provenance=cg.sc_challenge_grasp(None)
    world_cable=cg.transform([0,.015385,.04045,.4432,-.4838,1.3303])
    world_cable[:3,3]+=tcp[:3,3]
    cable_tip=cg.transform([.01,.02,.03,0,0,0]) @ cg.transform([-.052,0,0,0,0,3.14159]) @ cg.transform([.01165,0,0,-1.570796,0,-1.570796])
    np.testing.assert_allclose(tcp@grasp,world_cable@cable_tip,atol=1e-12)
    assert provenance['runtime_policy_input'] is False
    wrong=tcp@cg.transform([0,.015385,.04045,.4432,-.4838,1.3303])@cable_tip
    assert np.linalg.norm(wrong[:3,3]-(tcp@grasp)[:3,3]) > .01


def test_quarter_turn_inverse_preserves_pixel_landmark_identity():
    # Exercise mapping independently, without loading neural network packages.
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'diagnose_image_orientation.py'
    tree=ast.parse(source.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inverse_quarter_turn')
    namespace={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    grid=np.arange(5*7).reshape(5,7)
    for turns in range(4):
        rotated=np.rot90(grid,turns)
        for label in [0,9,24,34]:
            y,x=np.argwhere(rotated==label)[0]
            original=namespace['inverse_quarter_turn'](np.array([[x,y]]),turns,5,7)[0]
            assert grid[original[1],original[0]] == label
