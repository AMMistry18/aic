"""Evaluator reset geometry from the challenge's authored held-SC setup.

Never imported by the production perception/controller. The engine adds the
configured offset in WORLD axes and uses absolute cable RPY at robot HOME;
interpreting that offset as TCP-local would be incorrect.
"""
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from .lastinch_geometry import HOME, MODELS, parse_pose, robot_fk, sdf_frames, source_model, transform


def sc_challenge_grasp(robot):
    # SDFormat accepts its experimental prefix without an XML namespace; the
    # stdlib XML parser requires one. Bind it for parsing, without editing SDF.
    raw=(MODELS/'sfp_sc_cable_reversed/model.sdf').read_text()
    cable=ET.fromstring(raw.replace('<sdf ', '<sdf xmlns:experimental="urn:sdformat:experimental" ',1)).find('model')
    include = next(e for e in cable.findall('include') if e.findtext('uri') == 'model://SC Plug')
    connection_name = include.find('pose').get('relative_to')
    base_frames = sdf_frames(source_model('cable_base_c_rotated_reversed'))
    plug_frames = sdf_frames(source_model('SC Plug'))
    cable_from_tip = base_frames[connection_name] @ parse_pose(include) @ plug_frames['sc_tip_link']
    world_from_tcp = robot_fk(robot, HOME)['gripper/tcp']
    # Nominal documented qualification grasp, before trial-specific jitter.
    world_from_cable = transform([0., .015385, .04045, .4432, -.4838, 1.3303])
    world_from_cable[:3, 3] += world_from_tcp[:3, 3]
    grasp = np.linalg.inv(world_from_tcp) @ world_from_cable @ cable_from_tip
    provenance = {'profile': 'challenge_sc', 'world_offset_m': [0., .015385, .04045],
                  'cable_world_rpy_rad': [.4432, -.4838, 1.3303],
                  'home_joint_rad': HOME.tolist(), 'derivation': 'inverse(T_world_tcp_HOME) @ T_world_cable_HOME @ T_cable_sc_tip',
                  'runtime_policy_input': False,
                  'source_sha256': {name: hashlib.sha256((MODELS/name/'model.sdf').read_bytes()).hexdigest()
                     for name in ['sfp_sc_cable_reversed', 'cable_base_c_rotated_reversed', 'SC Plug']}}
    return grasp, provenance
