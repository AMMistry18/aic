"""Frozen final evaluation utilities. Tests use separate development seed ranges."""
import math
import random
import statistics
import numpy as np


def connector_cases(connector,seed_start):
    if connector not in ('sc','sfp'):raise ValueError('Unknown connector')
    cases=[];jittered=set(random.Random(2026111244).sample(range(50),25))
    for i in range(50):
        seed=seed_start+i;rng=np.random.default_rng(seed)
        if connector=='sc':
            angle=rng.uniform(-np.pi,np.pi);radius=rng.uniform(.5,1.5)
            lateral=[float(radius*np.cos(angle)),float(radius*np.sin(angle))]
            index=i%5;slot=0;gap=float(rng.uniform(5,15));angular=rng.uniform(-.1,.1,3).tolist()
            jitter=seed+10000 if i in jittered else None
        else:
            index=(i%10)//2;slot=i%2;gap=float(rng.uniform(7,15))
            lateral=rng.uniform(-.8,.8,2).tolist();angular=rng.uniform(-.03,.03,3).tolist();jitter=seed+20000
        cases.append(dict(case_id=f'final_{connector}_{seed}',seed=seed,connector=connector,
            target_index=index,sfp_port=slot,gap_mm=gap,lateral_mm=lateral,angular_deg=angular,
            grasp_jitter_seed=jitter))
    random.Random(2026111245+(connector=='sfp')).shuffle(cases)
    return cases


def wilson(successes,total):
    if not total:return None
    z=1.959963984540054;p=successes/total;denom=1+z*z/total
    mid=(p+z*z/(2*total))/denom
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [mid-half,mid+half]


def summarize(rows):
    out={}
    for connector in ('sc','sfp'):
        selected=[r for r in rows if r['case']['connector']==connector]
        eligible=[r for r in selected if r.get('status')=='completed']
        ok=[r for r in selected if r.get('verified_success')]
        failures={}
        for r in selected:
            if not r.get('verified_success'):
                cause=r.get('reason',r.get('error','unclassified'));failures[cause]=failures.get(cause,0)+1
        audits=[r['high_rate_wrist_audit'] for r in eligible if 'high_rate_wrist_audit' in r]
        out[connector]=dict(attempted=len(selected),policy_eligible=len(eligible),verified=len(ok),
            all_attempt_wilson95=wilson(len(ok),len(selected)),policy_eligible_wilson95=wilson(len(ok),len(eligible)),
            median_success_sim_s=statistics.median(r['sim_s'] for r in ok) if ok else None,
            median_success_wall_s=statistics.median(r['wall_s'] for r in ok) if ok else None,
            maximum_raw_peak_N=max((a['peak_tared_force_N'] for a in audits),default=None),
            complete_force_audits=sum(bool(a.get('complete_action_coverage')) for a in audits),
            failure_causes=failures)
    return out


def paired_exact_mcnemar(rows,component):
    groups={}
    for r in rows:groups.setdefault(r['base_case_id'],{})[r['ablation']]=bool(r.get('verified_success'))
    pairs=[g for g in groups.values() if 'full' in g and component in g]
    full_only=sum(g['full'] and not g[component] for g in pairs)
    ablation_only=sum(g[component] and not g['full'] for g in pairs)
    discordant=full_only+ablation_only
    p=min(1.,2*sum(math.comb(discordant,i) for i in range(min(full_only,ablation_only)+1))/2**discordant) if discordant else 1.
    return dict(pairs=len(pairs),full_only=full_only,ablation_only=ablation_only,exact_two_sided_p=p)
