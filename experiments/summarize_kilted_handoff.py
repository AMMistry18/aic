"""Independent reset audit from observed cable transforms, never policy input."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('.artifacts/kilted_handoff'))
    args=parser.parse_args();root=args.run
    spec=json.loads((root/'scene.json').read_text())
    samples=json.loads((root/'cable_transforms.json').read_text())
    transforms={}
    for item in samples[0]['transforms']:
        t=item['transform'];matrix=np.eye(4)
        matrix[:3,3]=[t['translation'][k] for k in ('x','y','z')]
        matrix[:3,:3]=Rotation.from_quat([t['rotation'][k] for k in ('x','y','z','w')]).as_matrix()
        transforms[item['child_frame_id']]=matrix
    name=f"cable_{spec['reset']['target_index']}"
    tip='sc_tip_link' if spec['connector']=='sc' else 'sfp_tip_link'
    actual=transforms[name]@transforms[f'{name}/{tip}']
    expected=np.array(spec['reset']['tip_world'])
    error=np.linalg.norm(actual[:3,3]-expected[:3,3])*1000
    angle=Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3]).magnitude()*180/np.pi
    result={'purpose':'evaluator_only_observed_reset_not_insertion','tip_world_observed':actual.tolist(),
        'world_sha256':hashlib.sha256((root/'world.sdf').read_bytes()).hexdigest(),
        'tip_position_error_mm':float(error),'tip_rotation_error_deg':float(angle),
        'position_threshold_mm':.25,'rotation_threshold_deg':.25,
        'reset_gate_passed':bool(error<.25 and angle<.25),
        'sim_time':samples[0]['transforms'][0]['header']['stamp'],
        'one_reset_only':True,'policy_receives_this_artifact':False}
    (root/'observed_reset_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
