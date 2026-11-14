"""Freeze matched last-inch cases across all 15 physical target ports."""
import argparse
import hashlib
import json
from pathlib import Path
import random

METHODS=('initial_pose_guarded','visual_without_recovery','full_visual_force_recovery')
TARGETS=[('sc',i,0) for i in range(5)]+[('sfp',i,p) for i in range(5) for p in (0,1)]


def cases(split):
    output=[]
    if split=='pilot':
        selections=[(connector,i,p,gap) for connector,i,p in TARGETS for gap in (5,15)]
        base=1000
    elif split=='final':
        selections=[(connector,i,p,10) for connector in ('sc','sfp')
                    for _,i,p in [t for t in TARGETS if t[0]==connector]
                    for _ in range(20 if connector=='sc' else 10)]
        base=100000
    else:
        selections=[(connector,i,p,10) for connector,i,p in TARGETS for _ in range(4)]
        base=200000
    for index,(connector,target,port,gap) in enumerate(selections):
        seed=base+index
        rng=random.Random(seed)
        order=list(METHODS)
        rng.shuffle(order)
        if split=='pilot':
            lateral=[0.,0.]
            angles=[0.,0.,0.]
        elif split=='final':
            lateral=[rng.uniform(-.5,.5) for _ in range(2)]
            angles=[rng.uniform(-1,1) for _ in range(3)]
        else:
            severity=(.25,.5,1.,2.)[index%4]
            lateral=[severity,0.]
            angles=[0.,0.,0.]
        output.append({'case_id':f'{split}_{seed}','split':split,'seed':seed,'scene_seed':seed,
            'connector':connector,'target_index':target,'sfp_port':port,'gap_mm':gap,
            'lateral_error_port_mm':lateral,'angular_error_port_deg':angles,
            'method_order':order,'simulator_seed':None,'automatic_latch':False,
            'force_abort_n':12 if connector=='sc' else 18,'timeout_wall_s':120,
            'scope':'grasped_connector_to_verified_seating',
            'execution_status':'not_run','method_implementation_status':'requires_validated_policy_backend'})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('experiments/configs/lastinch_cases.json'))
    a=p.parse_args()
    all_cases=[r for split in ('pilot','final','robustness') for r in cases(split)]
    payload={'schema_version':1,'methods':list(METHODS),'case_count':len(all_cases),
             'trial_count':len(all_cases)*len(METHODS),'cases':all_cases}
    text=json.dumps(payload,indent=2)+'\n'
    if a.out.exists() and a.out.read_text()!=text:
        p.error('Frozen cases changed; write a versioned new manifest instead of overwriting')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(text)
    print(json.dumps({'cases':len(all_cases),'trials':len(all_cases)*3,
                      'sha256':hashlib.sha256(text.encode()).hexdigest()}))


if __name__=='__main__': main()
