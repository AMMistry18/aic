"""Frame resolution and reset kinematics for the last-inch testbed."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / '.artifacts/upstream-phase1/aic_assets/models'
JOINTS = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
HOME = np.array([-.1597, -1.3542, -1.6648, -1.6933, 1.571, 1.411])


def transform(values=None):
    vals = np.zeros(6) if values is None else np.asarray(values, float)
    t = np.eye(4)
    t[:3, 3] = vals[:3]
    t[:3, :3] = Rotation.from_euler('xyz', vals[3:]).as_matrix()
    return t


def parse_pose(element):
    return transform(list(map(float, element.findtext('pose', '0 0 0 0 0 0').split())))


def pose_string(t):
    vals = [*t[:3, 3], *Rotation.from_matrix(t[:3, :3]).as_euler('xyz')]
    return ' '.join(f'{v:.12g}' for v in vals)


def sdf_frames(model):
    elements = {e.attrib['name']: e for e in list(model) if e.tag in ('link', 'frame')}
    resolved = {'__model__': np.eye(4), model.attrib['name']: np.eye(4)}
    def resolve(name):
        if name not in resolved:
            e = elements[name]
            pose = e.find('pose')
            relative = (pose.get('relative_to') if pose is not None else None) or '__model__'
            resolved[name] = resolve(relative) @ parse_pose(e)
        return resolved[name]
    for name in elements:
        resolve(name)
    return resolved


def robot_fk(robot, q):
    values = dict(zip(JOINTS, q))
    values.update({'gripper/left_finger_joint': .00655, 'gripper/right_finger_joint': .00655})
    frames = {'world': np.eye(4)}
    remaining = list(robot.findall('joint'))
    while remaining:
        progress = False
        for joint in remaining[:]:
            parent = joint.find('parent').get('link')
            if parent not in frames:
                continue
            origin = joint.find('origin')
            origin_values = ([*map(float, origin.get('xyz', '0 0 0').split()),
                              *map(float, origin.get('rpy', '0 0 0').split())]
                             if origin is not None else None)
            t = transform(origin_values)
            axis = joint.find('axis')
            v = np.array(list(map(float, axis.get('xyz').split()))) if axis is not None else np.array([1., 0, 0])
            if joint.get('type') in ('revolute', 'continuous'):
                t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(v * values.get(joint.get('name'), 0.)).as_matrix()
            elif joint.get('type') == 'prismatic':
                t[:3, 3] += t[:3, :3] @ (v * values.get(joint.get('name'), .00655))
            frames[joint.find('child').get('link')] = frames[parent] @ t
            remaining.remove(joint)
            progress = True
        if not progress:
            raise ValueError('Unresolved robot joint graph')
    return frames


def solve_robot_pose(robot, target, link='gripper/tcp'):
    def residual(q):
        actual = robot_fk(robot, q)[link]
        return np.r_[actual[:3, 3] - target[:3, 3],
                     .2 * Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).as_rotvec()]
    candidates = []
    for start in (HOME, HOME + np.array([.5, 0, 0, 0, 0, 0]), -HOME):
        solution = least_squares(residual, start, bounds=(-2*np.pi, 2*np.pi), max_nfev=250,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        norm = np.linalg.norm(residual(solution.x))
        frames = robot_fk(robot, solution.x)
        clearance = min(frames[name][2,3] for name in ('upper_arm_link','forearm_link','wrist_1_link'))
        travel = np.linalg.norm(np.angle(np.exp(1j*(solution.x-HOME))))
        candidates.append((norm, travel + max(0, .02-clearance)*100, solution.x))
    valid = [row for row in candidates if row[0] < 1e-5]
    if not valid:
        raise ValueError(f'Unreachable handoff pose: residual={min(r[0] for r in candidates)}')
    return min(valid, key=lambda row: row[1])[2]


def source_model(directory):
    return ET.parse(MODELS / directory / 'model.sdf').getroot().find('model')


if __name__ == '__main__':
    model = ET.parse(ROOT / '.artifacts/lastinch_resources/cable.sdf').getroot().find('model')
    frames = sdf_frames(model)
    for key in ('sfp_module_link', 'sfp_tip_link', 'sc_plug_link', 'sc_tip_link'):
        print(key, pose_string(frames[key]))
    print('endpoint separation', np.linalg.norm(frames['sc_tip_link'][:3, 3]-frames['sfp_tip_link'][:3, 3]))
    robot = ET.parse(ROOT / '.artifacts/lastinch_resources/ur5e.urdf').getroot()
    print('tcp home', pose_string(robot_fk(robot, HOME)['gripper/tcp']))
