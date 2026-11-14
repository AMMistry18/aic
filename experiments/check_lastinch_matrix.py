"""Execute reset geometry/IK preflight for every pilot target and a perturbed case."""
import argparse
import json
from pathlib import Path
from .lastinch_cases import cases
from .lastinch_scene import build
from .validate_lastinch import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('.artifacts/lastinch_matrix'))
    a=p.parse_args()
    results=[]
    selected=cases('pilot')+[cases('final')[7]]
    for row in selected:
        dest=a.out/row['case_id']
        try:
            build(dest,row['connector'],row['target_index'],row['sfp_port'],row['gap_mm']/1000,
                  row['scene_seed'],lateral_mm=row['lateral_error_port_mm'],angular_deg=row['angular_error_port_deg'])
            report=validate(dest)
            results.append({'case_id':row['case_id'],'passed':True,'connector':row['connector'],
                            'target':row['target_index'],'port':row['sfp_port'],'gap_mm':row['gap_mm']})
        except Exception as exc:
            results.append({'case_id':row['case_id'],'passed':False,'error':str(exc)})
    data={'purpose':'executed_geometry_and_ik_preflight_not_policy_trials','cases':results,
          'passed':sum(r['passed'] for r in results),'total':len(results)}
    (a.out/'geometry_preflight.json').write_text(json.dumps(data,indent=2)+'\n')
    print(f"Reset preflight: {data['passed']}/{data['total']} passed")
    raise SystemExit(0 if data['passed']==data['total'] else 1)


if __name__=='__main__': main()
