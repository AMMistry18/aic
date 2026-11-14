"""Freeze fresh, wider conditional-translation DEVELOPMENT cases, not finals."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'experiments/configs/translation_development30.json'
    if path.exists():raise FileExistsError('Preserve frozen manifest')
    cases=[]
    for target in range(5):
        seed=40000+target;rng=np.random.default_rng(seed);theta=rng.uniform(-np.pi,np.pi);radius=rng.uniform(.5,1.5)
        shared=dict(seed=seed,connector='sc',target_index=target,sfp_port=0,gap_mm=float(rng.uniform(5,15)),
            lateral_mm=(radius*np.array([np.cos(theta),np.sin(theta)])).tolist(),angular_deg=rng.uniform(-.1,.1,3).tolist())
        cases.append(dict(case_id=f'dev_{seed}',grasp_jitter_seed=None,**shared))
        cases.append(dict(case_id=f'dev_{41000+target}',grasp_jitter_seed=41000+target,**shared))
    config=dict(purpose='development_only_fresh_conditional_translation_screen_not_final',frozen_before_execution=True,
        hypotheses=['Wider translation correction than prior <=0.307 mm cases','Unknown rigid grasp variation remains observable from RGB under upstream angular alignment'],
        condition='upstream aligned SC handoff, actual rigid grasp translation estimated from images',
        cases=cases,lateral_norm_sampling_mm=[.5,1.5],gap_sampling_mm=[5,15],angular_component_sampling_deg=[-.1,.1],
        jitter_sampling='Existing scene generator: tip-frame grasp translation +/-2 mm, xyz Euler components +/-0.04 rad; physical initial tip alignment retained by reset IK',
        pairing='Each target/layout/handoff perturbation repeated with and without grasp jitter; distinct identities, same layout seed',
        independent_scoring_unchanged=True,counts_as_guarded_production_policy=False,
        policy_object_or_target_truth_inputs=False,final_100000_series_used=False,
        planned=10,status='frozen_not_executed',failure_accounting='All selected cases; report environment/setup failures separately, no tuning within batch')
    path.write_text(json.dumps(config,indent=2)+'\n');print(path)


if __name__=='__main__':main()
