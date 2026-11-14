"""Actual static development errors before/after rim repair, with source hashes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    baseline=ROOT/'.artifacts/perception_data29/evaluation/mouth_transfer_original_plug_multiview_val.json'
    refined=ROOT/'.artifacts/reference30/visibility_iterations/diagnostic_scores.json'
    predictions={r['case_id']:r for r in json.loads(baseline.read_text())['scenes']}
    rows=[]
    for fit in json.loads(refined.read_text()):
        case=fit['case_id'];old=predictions[case]
        truth=json.loads((ROOT/'.artifacts/perception_data29/qualified/cases'/case/'scene.json').read_text())['reset']
        mouth,tip=np.array(truth['mouth_world']),np.array(truth['tip_world'])
        error=mouth[:3,:3].T@((np.array(old['plug_position'])-fit['position'])-(tip[:3,3]-mouth[:3,3]))
        rows.append({'case_id':case,'port_before_mm':old['diagnostic_port_error_mm'],'port_after_mm':fit['diagnostic_error_mm'],
                     'relative_lateral_before_mm':old['diagnostic_relative_lateral_error_mm'],
                     'relative_lateral_after_mm':float(np.linalg.norm(error[:2])*1000),
                     'plug_angle_deg':old['diagnostic_plug_angle_error_deg']})
    summary={key:float(np.median([r[key] for r in rows])) for key in rows[0] if key!='case_id'}
    out=ROOT/'experiments/evidence/original_repair/static_rim';out.mkdir(parents=True,exist_ok=True)
    payload={'scope':'six static development scenes used during repair; no insertion or held-out generalization result',
             'rows':rows,'medians':summary,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [baseline,refined]}}
    (out/'measurements.json').write_text(json.dumps(payload,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
    for ax,before,after,title in [(axes[0],'port_before_mm','port_after_mm','Mouth position error'),
                                 (axes[1],'relative_lateral_before_mm','relative_lateral_after_mm','Relative lateral error: plug to mouth')]:
        for i,row in enumerate(rows):
            ax.plot([0,1],[row[before],row[after]],'-o',alpha=.75,label=row['case_id'].split('_')[-1],lw=1.3,ms=4)
        ax.set_xticks([0,1],['Learned mouth','CAD rim repair']);ax.set_yscale('log');ax.set_ylabel('Error (mm)');ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=.2)
    axes[1].legend(title='Scene',fontsize=7,ncol=2,loc='upper right')
    fig.suptitle('Static development replay (6 scenes used during repair)',fontsize=11)
    for suffix in ['png','svg']:fig.savefig(out/f'error_comparison.{suffix}',dpi=180)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
